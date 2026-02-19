// MockProvider — testing utility for unit tests.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Duration;

use unified_llm_types::{
    BoxFuture, BoxStream, Error, FinishReason, Message, ProviderAdapter, Request, Response,
    StreamEvent, Usage,
};

/// A stream that immediately returns None (empty).
struct EmptyStream;

impl futures_core::Stream for EmptyStream {
    type Item = Result<StreamEvent, Error>;
    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        std::task::Poll::Ready(None)
    }
}

/// Build an SSE body string from event type/data tuples.
/// Shared helper to avoid duplication across unit and integration tests.
pub fn build_sse_body(events: &[(&str, &str)]) -> String {
    let mut body = String::new();
    for (event_type, data) in events {
        body.push_str(&format!("event: {event_type}\ndata: {data}\n\n"));
    }
    body
}

/// Create a minimal test Response with the given text and provider name.
pub fn make_test_response(text: &str, provider: &str) -> Response {
    Response {
        id: "resp_test".into(),
        model: "test-model".into(),
        provider: provider.into(),
        message: Message::assistant(text),
        finish_reason: FinishReason::stop(),
        usage: Usage::default(),
        raw: None,
        warnings: vec![],
        rate_limit: None,
    }
}

/// A mock provider for testing. Returns pre-configured responses or errors
/// in the order they were queued (unified FIFO queue).
pub struct MockProvider {
    name: String,
    /// Unified queue: Ok(Response) or Err(Error), consumed in insertion order.
    actions: Mutex<Vec<Result<Response, Error>>>,
    /// Stream queue: each entry is a sequence of Result<StreamEvent, Error>.
    /// Ok items are yielded normally; Err items are yielded as stream errors.
    /// pub(crate) for advanced test scenarios (e.g., partial-then-error streams).
    pub(crate) stream_actions: Mutex<Vec<Vec<Result<StreamEvent, Error>>>>,
    recorded: Mutex<Vec<Request>>,
    call_count: AtomicUsize,
    /// Optional delay before yielding each stream event (for timeout testing).
    stream_delay: Mutex<Option<Duration>>,
}

impl MockProvider {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            actions: Mutex::new(Vec::new()),
            stream_actions: Mutex::new(Vec::new()),
            recorded: Mutex::new(Vec::new()),
            call_count: AtomicUsize::new(0),
            stream_delay: Mutex::new(None),
        }
    }

    /// Queue a successful response. Returned by the next `complete()` call
    /// after all previously queued items have been consumed.
    pub fn with_response(self, response: Response) -> Self {
        self.actions.lock().unwrap().push(Ok(response));
        self
    }

    /// Queue an error. Returned by the next `complete()` call
    /// after all previously queued items have been consumed.
    pub fn with_error(self, error: Error) -> Self {
        self.actions.lock().unwrap().push(Err(error));
        self
    }

    /// Queue a set of stream events to be returned by the next `stream()` call.
    /// All events are wrapped in Ok() automatically.
    pub fn with_stream_events(self, events: Vec<StreamEvent>) -> Self {
        let items: Vec<Result<StreamEvent, Error>> = events.into_iter().map(Ok).collect();
        self.stream_actions.lock().unwrap().push(items);
        self
    }

    /// Queue a stream that immediately yields an error (simulates connection failure).
    /// The stream yields `Err(error)` as its first and only item.
    pub fn with_stream_error(self, error: Error) -> Self {
        self.stream_actions.lock().unwrap().push(vec![Err(error)]);
        self
    }

    /// Set a delay before each stream event (for timeout testing).
    /// Use with `tokio::time::pause()` for deterministic timing.
    pub fn with_stream_delay(self, delay: Duration) -> Self {
        *self.stream_delay.lock().unwrap() = Some(delay);
        self
    }

    pub fn call_count(&self) -> usize {
        self.call_count.load(Ordering::SeqCst)
    }

    /// Return a clone of all requests recorded by `complete()` calls.
    pub fn recorded_requests(&self) -> Vec<Request> {
        self.recorded.lock().unwrap().clone()
    }
}

impl ProviderAdapter for MockProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn complete(&self, request: Request) -> BoxFuture<'_, Result<Response, Error>> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        self.recorded.lock().unwrap().push(request.clone());
        Box::pin(async {
            let mut actions = self.actions.lock().unwrap();
            if !actions.is_empty() {
                return actions.remove(0);
            }
            Err(Error::configuration("MockProvider: no actions configured"))
        })
    }

    fn stream(&self, _request: Request) -> BoxStream<'_, Result<StreamEvent, Error>> {
        let mut queue = self.stream_actions.lock().unwrap();
        let delay = *self.stream_delay.lock().unwrap();
        if !queue.is_empty() {
            let items = queue.remove(0);
            Box::pin(async_stream::stream! {
                for item in items {
                    if let Some(d) = delay {
                        tokio::time::sleep(d).await;
                    }
                    yield item;
                }
            })
        } else {
            Box::pin(EmptyStream)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use unified_llm_types::*;

    // --- Existing tests (preserved) ---

    #[tokio::test]
    async fn test_mock_provider_returns_response() {
        let mock =
            MockProvider::new("mock").with_response(make_test_response("Hello from mock", "mock"));
        let req = Request::default().model("test");
        let resp = mock.complete(req).await.unwrap();
        assert_eq!(resp.text(), "Hello from mock");
    }

    #[tokio::test]
    async fn test_mock_provider_returns_error() {
        let mock = MockProvider::new("mock").with_error(Error::from_http_status(
            429,
            "rate limited".into(),
            "mock",
            None,
            None,
        ));
        let req = Request::default().model("test");
        let err = mock.complete(req).await.unwrap_err();
        assert_eq!(err.kind, ErrorKind::RateLimit);
    }

    #[tokio::test]
    async fn test_mock_provider_name() {
        let mock = MockProvider::new("test-provider");
        assert_eq!(mock.name(), "test-provider");
    }

    #[tokio::test]
    async fn test_mock_provider_call_count() {
        let mock = MockProvider::new("mock")
            .with_response(make_test_response("r1", "mock"))
            .with_response(make_test_response("r2", "mock"));
        assert_eq!(mock.call_count(), 0);
        mock.complete(Request::default()).await.unwrap();
        assert_eq!(mock.call_count(), 1);
        mock.complete(Request::default()).await.unwrap();
        assert_eq!(mock.call_count(), 2);
    }

    // --- P2A-T04 tests ---

    #[tokio::test]
    async fn test_mock_provider_with_stream_events() {
        let events = vec![
            StreamEvent {
                event_type: StreamEventType::StreamStart,
                ..Default::default()
            },
            StreamEvent {
                event_type: StreamEventType::TextDelta,
                delta: Some("Hello".into()),
                ..Default::default()
            },
            StreamEvent {
                event_type: StreamEventType::Finish,
                ..Default::default()
            },
        ];
        let mock = MockProvider::new("mock").with_stream_events(events);
        let req = Request::default().model("test");
        let stream = mock.stream(req);
        let collected: Vec<_> = futures::StreamExt::collect::<Vec<_>>(stream).await;
        assert_eq!(collected.len(), 3);
        assert!(collected[0].is_ok());
        assert_eq!(
            collected[0].as_ref().unwrap().event_type,
            StreamEventType::StreamStart
        );
        assert_eq!(collected[1].as_ref().unwrap().delta, Some("Hello".into()));
        assert_eq!(
            collected[2].as_ref().unwrap().event_type,
            StreamEventType::Finish
        );
    }

    #[tokio::test]
    async fn test_mock_provider_tracks_requests() {
        let mock = MockProvider::new("mock").with_response(make_test_response("r1", "mock"));
        let req = Request::default()
            .model("test-model")
            .messages(vec![Message::user("hello")]);
        mock.complete(req).await.unwrap();
        let requests = mock.recorded_requests();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].model, "test-model");
    }

    #[test]
    fn test_build_sse_body() {
        let body = build_sse_body(&[("event_type1", "data1"), ("event_type2", "data2")]);
        assert!(body.contains("event: event_type1\ndata: data1\n\n"));
        assert!(body.contains("event: event_type2\ndata: data2\n\n"));
    }

    #[test]
    fn test_make_test_response() {
        let resp = make_test_response("Hello", "test-provider");
        assert_eq!(resp.text(), "Hello");
        assert_eq!(resp.provider, "test-provider");
        assert_eq!(resp.finish_reason.reason, "stop");
    }

    #[tokio::test]
    async fn test_mock_provider_stream_empty_without_events() {
        let mock = MockProvider::new("mock");
        let stream = mock.stream(Request::default());
        let collected: Vec<_> = futures::StreamExt::collect::<Vec<_>>(stream).await;
        assert!(collected.is_empty());
    }

    // --- P3-T02 tests: unified actions queue ---

    #[tokio::test]
    async fn test_mock_provider_interleaved_response_then_error() {
        // First call: success, second call: error
        let mock = MockProvider::new("mock")
            .with_response(make_test_response("first", "mock"))
            .with_error(Error::from_http_status(
                500,
                "boom".into(),
                "mock",
                None,
                None,
            ));
        let r1 = mock.complete(Request::default()).await;
        assert!(r1.is_ok());
        assert_eq!(r1.unwrap().text(), "first");
        let r2 = mock.complete(Request::default()).await;
        assert!(r2.is_err());
        assert_eq!(r2.unwrap_err().kind, ErrorKind::Server);
    }

    #[tokio::test]
    async fn test_mock_provider_three_step_tool_loop_pattern() {
        // Simulates: Step 1 → tool call response, Step 2 → 500 error (retry),
        //            Step 2 retry → final success
        let mock = MockProvider::new("mock")
            .with_response(make_test_response("tool_call_step", "mock"))
            .with_error(Error::from_http_status(
                500,
                "server error".into(),
                "mock",
                None,
                None,
            ))
            .with_response(make_test_response("recovered", "mock"));
        let r1 = mock.complete(Request::default()).await.unwrap();
        assert_eq!(r1.text(), "tool_call_step");
        let r2 = mock.complete(Request::default()).await;
        assert!(r2.is_err());
        let r3 = mock.complete(Request::default()).await.unwrap();
        assert_eq!(r3.text(), "recovered");
    }

    #[tokio::test]
    async fn test_mock_provider_exhausted_queue_returns_error() {
        // After all queued items consumed, subsequent calls return a configuration error
        let mock = MockProvider::new("mock").with_response(make_test_response("only_one", "mock"));
        let r1 = mock.complete(Request::default()).await.unwrap();
        assert_eq!(r1.text(), "only_one");
        let r2 = mock.complete(Request::default()).await;
        assert!(r2.is_err());
        assert_eq!(r2.unwrap_err().kind, ErrorKind::Configuration);
    }

    // --- P3-T06 tests: stream error support ---

    #[tokio::test]
    async fn test_mock_provider_stream_error() {
        let mock = MockProvider::new("mock").with_stream_error(Error::from_http_status(
            429,
            "rate limited".into(),
            "mock",
            None,
            None,
        ));
        let stream = mock.stream(Request::default());
        let collected: Vec<_> = futures::StreamExt::collect::<Vec<_>>(stream).await;
        assert_eq!(collected.len(), 1);
        assert!(collected[0].is_err());
        assert_eq!(
            collected[0].as_ref().unwrap_err().kind,
            ErrorKind::RateLimit
        );
    }

    #[tokio::test]
    async fn test_mock_provider_stream_error_then_success() {
        // First stream() call: error stream, second: success stream
        let mock = MockProvider::new("mock")
            .with_stream_error(Error::from_http_status(
                429,
                "rate limited".into(),
                "mock",
                None,
                None,
            ))
            .with_stream_events(vec![
                StreamEvent {
                    event_type: StreamEventType::StreamStart,
                    ..Default::default()
                },
                StreamEvent {
                    event_type: StreamEventType::TextDelta,
                    delta: Some("Hello".into()),
                    ..Default::default()
                },
                StreamEvent {
                    event_type: StreamEventType::Finish,
                    ..Default::default()
                },
            ]);

        // First stream: error
        let stream1 = mock.stream(Request::default());
        let collected1: Vec<_> = futures::StreamExt::collect::<Vec<_>>(stream1).await;
        assert_eq!(collected1.len(), 1);
        assert!(collected1[0].is_err());

        // Second stream: success
        let stream2 = mock.stream(Request::default());
        let collected2: Vec<_> = futures::StreamExt::collect::<Vec<_>>(stream2).await;
        assert_eq!(collected2.len(), 3);
        assert!(collected2[0].is_ok());
    }
}
