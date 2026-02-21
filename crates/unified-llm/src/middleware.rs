// middleware.rs -- Middleware trait and onion-pattern chain execution (Layer 3).
//
// Spec reference: S2.3 Middleware / Interceptor Pattern
// DoD: 8.1.6 -- Middleware chain executes in correct order
//   (request: registration order, response: reverse order)

use unified_llm_types::{
    BoxFuture, BoxStream, Error, Request, Response, StreamEvent, StreamEventType,
};

/// Handle for delegating to the next middleware in the chain, or to the provider.
pub struct Next<'a> {
    /// Handler for non-streaming (complete) requests.
    pub(crate) complete_fn:
        Box<dyn FnOnce(Request) -> BoxFuture<'a, Result<Response, Error>> + Send + 'a>,
    /// Handler for streaming requests.
    #[allow(clippy::type_complexity)]
    pub(crate) stream_fn: Box<
        dyn FnOnce(
                Request,
            )
                -> BoxFuture<'a, Result<BoxStream<'a, Result<StreamEvent, Error>>, Error>>
            + Send
            + 'a,
    >,
}

impl<'a> Next<'a> {
    /// Delegate a non-streaming request to the next handler in the chain.
    pub fn run(self, request: Request) -> BoxFuture<'a, Result<Response, Error>> {
        (self.complete_fn)(request)
    }

    /// Delegate a streaming request to the next handler in the chain.
    pub fn run_stream(
        self,
        request: Request,
    ) -> BoxFuture<'a, Result<BoxStream<'a, Result<StreamEvent, Error>>, Error>> {
        (self.stream_fn)(request)
    }
}

/// Middleware trait for cross-cutting concerns (logging, caching, rate limiting, etc.).
///
/// **Execution order** (spec S2.3):
/// - Request phase: registration order (first registered = first to execute)
/// - Response phase: reverse order (onion/chain-of-responsibility pattern)
///
/// Implementors MUST call `next.run(request)` (or `next.run_stream(request)`) to
/// delegate to the next middleware. Failing to call `next` short-circuits the chain.
pub trait Middleware: Send + Sync {
    /// Process a non-streaming (complete) request. Default: pass through.
    fn process<'a>(
        &'a self,
        request: Request,
        next: Next<'a>,
    ) -> BoxFuture<'a, Result<Response, Error>> {
        next.run(request)
    }

    /// Process a streaming request. Default: pass through.
    fn process_stream<'a>(
        &'a self,
        request: Request,
        next: Next<'a>,
    ) -> BoxFuture<'a, Result<BoxStream<'a, Result<StreamEvent, Error>>, Error>> {
        next.run_stream(request)
    }
}

/// A simple logging middleware that logs request/response metadata using `tracing`.
///
/// Logs at `info` level:
/// - Request: model, provider
/// - Response: finish reason, token usage, duration
///
/// # Example
/// ```ignore
/// use std::sync::Arc;
/// use unified_llm::middleware::LoggingMiddleware;
///
/// let client = Client::builder()
///     .provider("anthropic", adapter)
///     .middleware(Arc::new(LoggingMiddleware))
///     .build()?;
/// ```
pub struct LoggingMiddleware;

impl Middleware for LoggingMiddleware {
    fn process<'a>(
        &'a self,
        request: Request,
        next: Next<'a>,
    ) -> BoxFuture<'a, Result<Response, Error>> {
        Box::pin(async move {
            let model = request.model.clone();
            let provider = request.provider.clone().unwrap_or_default();
            tracing::info!(model = %model, provider = %provider, "LLM request");

            let start = std::time::Instant::now();
            let response = next.run(request).await?;

            tracing::info!(
                model = %model,
                provider = %response.provider,
                finish_reason = %response.finish_reason.reason,
                input_tokens = response.usage.input_tokens,
                output_tokens = response.usage.output_tokens,
                duration_ms = start.elapsed().as_millis() as u64,
                "LLM response"
            );
            Ok(response)
        })
    }

    fn process_stream<'a>(
        &'a self,
        request: Request,
        next: Next<'a>,
    ) -> BoxFuture<'a, Result<BoxStream<'a, Result<StreamEvent, Error>>, Error>> {
        Box::pin(async move {
            let model = request.model.clone();
            let provider = request.provider.clone().unwrap_or_default();
            tracing::info!(model = %model, provider = %provider, "LLM stream request");

            let start = std::time::Instant::now();
            let inner_stream = next.run_stream(request).await?;

            let wrapped: BoxStream<'a, Result<StreamEvent, Error>> =
                Box::pin(async_stream::stream! {
                    use futures::StreamExt;
                    let mut inner = inner_stream;
                    let mut final_usage: Option<unified_llm_types::Usage> = None;
                    let mut final_finish_reason: Option<unified_llm_types::FinishReason> = None;

                    while let Some(event) = inner.next().await {
                        if let Ok(ref evt) = event {
                            if evt.event_type == StreamEventType::Finish {
                                final_usage = evt.usage.clone();
                                final_finish_reason = evt.finish_reason.clone();
                            }
                        }
                        yield event;
                    }

                    // Log at end of stream
                    let usage = final_usage.unwrap_or_default();
                    let finish = final_finish_reason
                        .map(|f| f.reason)
                        .unwrap_or_else(|| "unknown".to_string());

                    tracing::info!(
                        model = %model,
                        provider = %provider,
                        finish_reason = %finish,
                        input_tokens = usage.input_tokens,
                        output_tokens = usage.output_tokens,
                        duration_ms = start.elapsed().as_millis() as u64,
                        "LLM stream complete"
                    );
                });

            Ok(wrapped)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::Mutex as StdMutex;
    use unified_llm_types::*;

    use crate::client::Client;
    use crate::testing::{make_test_response, MockProvider};

    // --- Test middleware: records the order it was called ---

    struct OrderRecordingMiddleware {
        id: &'static str,
        log: Arc<StdMutex<Vec<String>>>,
    }

    impl Middleware for OrderRecordingMiddleware {
        fn process<'a>(
            &'a self,
            request: Request,
            next: Next<'a>,
        ) -> BoxFuture<'a, Result<Response, Error>> {
            let id = self.id;
            let log = self.log.clone();
            Box::pin(async move {
                log.lock().unwrap().push(format!("{id}:request"));
                let response = next.run(request).await?;
                log.lock().unwrap().push(format!("{id}:response"));
                Ok(response)
            })
        }
    }

    // --- Test middleware: modifies request and response ---

    struct ModifyingMiddleware;

    impl Middleware for ModifyingMiddleware {
        fn process<'a>(
            &'a self,
            mut request: Request,
            next: Next<'a>,
        ) -> BoxFuture<'a, Result<Response, Error>> {
            Box::pin(async move {
                request.temperature = Some(0.42); // modify a real field to prove middleware runs
                let mut response = next.run(request).await?;
                response.warnings.push(Warning {
                    message: "modified by middleware".to_string(),
                    code: Some("MW001".to_string()),
                });
                Ok(response)
            })
        }
    }

    // --- Test middleware: returns error without calling next ---

    struct ErrorMiddleware;

    impl Middleware for ErrorMiddleware {
        fn process<'a>(
            &'a self,
            _request: Request,
            _next: Next<'a>,
        ) -> BoxFuture<'a, Result<Response, Error>> {
            Box::pin(async move { Err(Error::configuration("middleware rejected request")) })
        }
    }

    #[tokio::test]
    async fn test_no_middleware_passthrough() {
        let mock = MockProvider::new("test").with_response(make_test_response("hello", "test"));
        let client = Client::builder()
            .provider("test", Box::new(mock))
            .build()
            .unwrap();

        let request = Request {
            model: "m".into(),
            messages: vec![Message::user("Hi")],
            ..Default::default()
        };
        let response = client.complete(request).await.unwrap();
        assert_eq!(response.text(), "hello");
    }

    #[tokio::test]
    async fn test_single_middleware_modifies_request_and_response() {
        let mock = MockProvider::new("test").with_response(make_test_response("hello", "test"));
        let client = Client::builder()
            .provider("test", Box::new(mock))
            .middleware(Arc::new(ModifyingMiddleware))
            .build()
            .unwrap();

        let request = Request {
            model: "m".into(),
            messages: vec![Message::user("Hi")],
            ..Default::default()
        };
        let response = client.complete(request).await.unwrap();
        assert_eq!(response.warnings.len(), 1);
        assert_eq!(response.warnings[0].message, "modified by middleware");
    }

    #[tokio::test]
    async fn test_middleware_onion_order_two_middlewares() {
        // Spec S2.3: registration order for request, reverse for response.
        // [A, B] => A:request, B:request, B:response, A:response
        let log = Arc::new(StdMutex::new(Vec::new()));
        let mw_a = Arc::new(OrderRecordingMiddleware {
            id: "A",
            log: log.clone(),
        });
        let mw_b = Arc::new(OrderRecordingMiddleware {
            id: "B",
            log: log.clone(),
        });

        let mock = MockProvider::new("test").with_response(make_test_response("hello", "test"));
        let client = Client::builder()
            .provider("test", Box::new(mock))
            .middleware(mw_a)
            .middleware(mw_b)
            .build()
            .unwrap();

        let request = Request {
            model: "m".into(),
            messages: vec![Message::user("Hi")],
            ..Default::default()
        };
        client.complete(request).await.unwrap();

        let entries = log.lock().unwrap().clone();
        assert_eq!(
            entries,
            vec!["A:request", "B:request", "B:response", "A:response"]
        );
    }

    #[tokio::test]
    async fn test_middleware_onion_order_three_middlewares() {
        // [A, B, C] => A:req, B:req, C:req, C:res, B:res, A:res
        let log = Arc::new(StdMutex::new(Vec::new()));
        let mw_a = Arc::new(OrderRecordingMiddleware {
            id: "A",
            log: log.clone(),
        });
        let mw_b = Arc::new(OrderRecordingMiddleware {
            id: "B",
            log: log.clone(),
        });
        let mw_c = Arc::new(OrderRecordingMiddleware {
            id: "C",
            log: log.clone(),
        });

        let mock = MockProvider::new("test").with_response(make_test_response("hello", "test"));
        let client = Client::builder()
            .provider("test", Box::new(mock))
            .middleware(mw_a)
            .middleware(mw_b)
            .middleware(mw_c)
            .build()
            .unwrap();

        let request = Request {
            model: "m".into(),
            messages: vec![Message::user("Hi")],
            ..Default::default()
        };
        client.complete(request).await.unwrap();

        let entries = log.lock().unwrap().clone();
        assert_eq!(
            entries,
            vec![
                "A:request",
                "B:request",
                "C:request",
                "C:response",
                "B:response",
                "A:response"
            ]
        );
    }

    #[tokio::test]
    async fn test_middleware_error_propagation() {
        let mock = MockProvider::new("test").with_response(make_test_response("hello", "test"));
        let client = Client::builder()
            .provider("test", Box::new(mock))
            .middleware(Arc::new(ErrorMiddleware))
            .build()
            .unwrap();

        let request = Request {
            model: "m".into(),
            messages: vec![Message::user("Hi")],
            ..Default::default()
        };
        let err = client.complete(request).await.unwrap_err();
        assert_eq!(err.kind, ErrorKind::Configuration);
        assert!(err.message.contains("middleware rejected"));
    }

    #[tokio::test]
    async fn test_middleware_streaming_default_passthrough() {
        use futures::StreamExt;

        let mock = MockProvider::new("test").with_stream_events(vec![
            StreamEvent {
                event_type: StreamEventType::StreamStart,
                ..Default::default()
            },
            StreamEvent {
                event_type: StreamEventType::TextDelta,
                delta: Some("hi".into()),
                ..Default::default()
            },
            StreamEvent {
                event_type: StreamEventType::Finish,
                ..Default::default()
            },
        ]);
        // ModifyingMiddleware only overrides process(), not process_stream().
        let client = Client::builder()
            .provider("test", Box::new(mock))
            .middleware(Arc::new(ModifyingMiddleware))
            .build()
            .unwrap();

        let request = Request {
            model: "m".into(),
            messages: vec![Message::user("Hi")],
            ..Default::default()
        };
        let event_stream = client.stream(request).unwrap();
        let events: Vec<_> = event_stream
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .filter_map(|r| r.ok())
            .collect();

        assert_eq!(events.len(), 3);
        assert_eq!(events[0].event_type, StreamEventType::StreamStart);
        assert_eq!(events[1].delta.as_deref(), Some("hi"));
        assert_eq!(events[2].event_type, StreamEventType::Finish);
    }

    // --- FP-12: LoggingMiddleware ---

    #[test]
    fn test_logging_middleware_implements_trait() {
        // Verify LoggingMiddleware can be used as Arc<dyn Middleware>
        let mw: Arc<dyn Middleware> = Arc::new(LoggingMiddleware);
        let _ = mw; // compiles = pass
    }

    #[tokio::test]
    async fn test_logging_middleware_processes_stream() {
        use futures::StreamExt;

        let mock = MockProvider::new("test").with_stream_events(vec![
            StreamEvent {
                event_type: StreamEventType::StreamStart,
                ..Default::default()
            },
            StreamEvent {
                event_type: StreamEventType::TextDelta,
                delta: Some("hello".into()),
                ..Default::default()
            },
            StreamEvent {
                event_type: StreamEventType::Finish,
                finish_reason: Some(unified_llm_types::FinishReason::stop()),
                usage: Some(unified_llm_types::Usage {
                    input_tokens: 5,
                    output_tokens: 3,
                    total_tokens: 8,
                    ..Default::default()
                }),
                ..Default::default()
            },
        ]);
        let client = Client::builder()
            .provider("test", Box::new(mock))
            .middleware(Arc::new(LoggingMiddleware))
            .build()
            .unwrap();

        let request = Request {
            model: "m".into(),
            messages: vec![Message::user("Hi")],
            ..Default::default()
        };
        let event_stream = client.stream(request).unwrap();
        let events: Vec<_> = event_stream
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .filter_map(|r| r.ok())
            .collect();

        // LoggingMiddleware should pass through all events transparently
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].event_type, StreamEventType::StreamStart);
        assert_eq!(events[1].event_type, StreamEventType::TextDelta);
        assert_eq!(events[1].delta.as_deref(), Some("hello"));
        assert_eq!(events[2].event_type, StreamEventType::Finish);
        assert_eq!(events[2].usage.as_ref().unwrap().input_tokens, 5);
    }
}
