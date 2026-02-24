// api/stream.rs — stream() function + StreamResult (Layer 4).
//
// Wraps Client.stream() with prompt standardization, StreamAccumulator,
// tool execution loops, retry logic, and cancellation support.

use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

use futures::StreamExt;
use futures_core::Stream;
use unified_llm_types::*;

use crate::client::Client;
use crate::util::retry::with_retry;
use crate::util::stream_accumulator::StreamAccumulator;

use super::generate::{build_request, build_step_result, standardize_messages};
use super::generate_types::StepResult;
use super::tool_loop::execute_all_tools;
use super::types::GenerateOptions;

/// The primary streaming generation function.
///
/// Wraps `Client.stream()` with prompt standardization, tool execution loops,
/// and retry logic. When active tools are provided and the model makes tool
/// calls, the stream pauses while tools execute, then resumes streaming the
/// next LLM response. The consumer sees a single flat stream of events.
///
/// Cancellation (DoD 8.4.9): If `abort_signal` is set and cancelled, yields
/// `Err(Error::abort())` and terminates.
///
/// Retry rules (DoD 8.8.9): Only the initial connection of each step is
/// retried. Once events have been yielded, errors are emitted as Error
/// events without retry.
///
/// Spec reference: §4.4, §6.2, §6.6
///
/// See also [`stream_with_default`] for a convenience wrapper that uses the
/// module-level default client and returns a `'static` stream.
pub fn stream<'a>(options: GenerateOptions, client: &'a Client) -> Result<StreamResult<'a>, Error> {
    // M-9: Validate tools and tool_choice upfront before any API calls.
    if let Some(ref tools) = options.tools {
        for tool in tools {
            tool.definition.validate()?;
        }
    }
    if let Some(ref tc) = options.tool_choice {
        tc.validate()?;
    }

    // Validate upfront (before entering async)
    let messages = standardize_messages(
        options.prompt.as_deref(),
        options.messages.clone(),
        options.system.as_deref(),
    )?;

    let tool_definitions: Vec<ToolDefinition> = options
        .tools
        .as_ref()
        .map(|tools| tools.iter().map(|t| t.definition.clone()).collect())
        .unwrap_or_default();

    let max_retries = options.max_retries;
    let max_tool_rounds = options.max_tool_rounds;
    let abort_signal = options.abort_signal.clone();
    let stop_when = options.stop_when.clone();
    let timeout_config = options.timeout.clone();

    // Build the multi-step stream using async_stream.
    // This handles retry, tool execution loops, cancellation, and multi-step streaming.
    let event_stream = async_stream::stream! {
        let mut conversation = messages;
        let mut steps: Vec<StepResult> = Vec::new();
        let retry_policy = RetryPolicy {
            max_retries,
            ..Default::default()
        };

        // Compute total deadline once (DoD 8.4.10)
        let total_deadline = timeout_config
            .as_ref()
            .and_then(|t| t.total)
            .map(|secs| tokio::time::Instant::now() + Duration::from_secs_f64(secs));
        let per_step_duration = timeout_config
            .as_ref()
            .and_then(|t| t.per_step)
            .map(Duration::from_secs_f64);

        // P0-1: Shared retry counter — inner first-read retry shares budget with outer
        let mut retries_remaining: u32 = max_retries;

        for round in 0..=max_tool_rounds {
            // Check cancellation before each step (DoD 8.4.9)
            if let Some(ref token) = abort_signal {
                if token.is_cancelled() {
                    yield Err(Error::abort());
                    return;
                }
            }

            // Check total timeout before each step
            if let Some(deadline) = total_deadline {
                if tokio::time::Instant::now() >= deadline {
                    yield Ok(StreamEvent {
                        event_type: StreamEventType::Error,
                        error: Some(Box::new(StreamError::timeout(format!(
                            "Total stream timeout of {}s exceeded",
                            timeout_config.as_ref().and_then(|t| t.total).unwrap_or(0.0)
                        )))),
                        ..Default::default()
                    });
                    return;
                }
            }

            let request = build_request(&options, &conversation, &tool_definitions);

            // Retry the stream connection for this step.
            let mut provider_stream = match with_retry(&retry_policy, || {
                let req = request.clone();
                async move {
                    let s = client.stream(req)?;
                    Ok(s)
                }
            }).await {
                Ok(s) => s,
                Err(e) => {
                    yield Err(e);
                    return;
                }
            };

            // Compute per-step deadline for this round
            let step_deadline = per_step_duration
                .map(|d| tokio::time::Instant::now() + d);

            // Yield all events from this step, accumulating as we go.
            let mut accumulator = StreamAccumulator::new();
            let mut events_yielded = false;

            loop {
                // Compute the nearest deadline (total or per-step)
                let effective_deadline = match (total_deadline, step_deadline) {
                    (Some(t), Some(s)) => Some(if t < s { t } else { s }),
                    (Some(t), None) => Some(t),
                    (None, Some(s)) => Some(s),
                    (None, None) => None,
                };

                // Race between next event, cancellation (DoD 8.4.9), and timeouts
                #[allow(clippy::large_enum_variant)]
                enum PollResult {
                    Event(Option<Result<StreamEvent, Error>>),
                    Abort,
                    Timeout,
                }

                let poll = match (&abort_signal, effective_deadline) {
                    (Some(token), Some(deadline)) => {
                        tokio::select! {
                            next = provider_stream.next() => PollResult::Event(next),
                            _ = token.cancelled() => PollResult::Abort,
                            _ = tokio::time::sleep_until(deadline) => PollResult::Timeout,
                        }
                    }
                    (Some(token), None) => {
                        tokio::select! {
                            next = provider_stream.next() => PollResult::Event(next),
                            _ = token.cancelled() => PollResult::Abort,
                        }
                    }
                    (None, Some(deadline)) => {
                        tokio::select! {
                            next = provider_stream.next() => PollResult::Event(next),
                            _ = tokio::time::sleep_until(deadline) => PollResult::Timeout,
                        }
                    }
                    (None, None) => {
                        PollResult::Event(provider_stream.next().await)
                    }
                };

                let item = match poll {
                    PollResult::Abort => {
                        yield Err(Error::abort());
                        return;
                    }
                    PollResult::Timeout => {
                        // Determine which timeout fired for the error message
                        let msg = if total_deadline.is_some()
                            && total_deadline == effective_deadline
                        {
                            format!(
                                "Total stream timeout of {}s exceeded",
                                timeout_config.as_ref().and_then(|t| t.total).unwrap_or(0.0)
                            )
                        } else {
                            format!(
                                "Per-step stream timeout of {}s exceeded",
                                per_step_duration.map(|d| d.as_secs_f64()).unwrap_or(0.0)
                            )
                        };
                        yield Ok(StreamEvent {
                            event_type: StreamEventType::Error,
                            error: Some(Box::new(StreamError::timeout(msg))),
                            ..Default::default()
                        });
                        return;
                    }
                    PollResult::Event(event) => event,
                };

                let item = match item {
                    Some(item) => item,
                    None => break,
                };
                match item {
                    Ok(event) => {
                        accumulator.process(&event);
                        events_yielded = true;
                        yield Ok(event);
                    }
                    Err(e) => {
                        // P0-1: Single shared retry budget. The outer with_retry (line 118)
                        // may have consumed some retries for connection failures.
                        // This inner path handles first-read errors (stream created OK
                        // but first event is Err). We use a shared counter to ensure
                        // total attempts never exceed max_retries + 1.
                        if !events_yielded && e.retryable && retries_remaining > 0 {
                            retries_remaining -= 1;
                            // L-21: Backoff delay before retry to prevent rapid-fire retries
                            tokio::time::sleep(Duration::from_millis(500)).await;
                            // Re-create the stream for this step
                            match client.stream(request.clone()) {
                                Ok(new_stream) => {
                                    provider_stream = new_stream;
                                    continue;
                                }
                                Err(reconnect_err) => {
                                    yield Err(reconnect_err);
                                    return;
                                }
                            }
                        } else if events_yielded {
                            // Error after events yielded — emit as Error event,
                            // do NOT retry (DoD 8.8.9)
                            yield Ok(StreamEvent {
                                event_type: StreamEventType::Error,
                                error: Some(Box::new(StreamError::from_error(&e))),
                                ..Default::default()
                            });
                            return;
                        } else {
                            // Error before events, not retryable or max_retries=0
                            yield Err(e);
                            return;
                        }
                    }
                }
            }

            // After the stream completes, check if we need to continue with tools.
            // S-3 NOTE: The accumulator is reset implicitly because a new one is created
            // each iteration. If follow-up steps' TextDelta events are not reaching the
            // consumer, investigate whether the provider adapter's stream translator
            // re-emits TextStart on continuation responses (OpenAI Responses API may
            // require the text_started flag to be reset between separate HTTP streams).
            let response = match accumulator.response() {
                Some(r) => r,
                None => return, // Stream ended without Finish event
            };

            // Check exit conditions
            let tool_calls = response.tool_calls();
            let is_tool_call = !tool_calls.is_empty()
                && response.finish_reason.reason == "tool_calls";

            if !is_tool_call {
                break; // Natural completion — done
            }
            if round >= max_tool_rounds {
                break; // Budget exhausted
            }

            // Check if any called tools are active (have execute handlers)
            let tools = options.tools.as_deref().unwrap_or(&[]);
            let has_active = tool_calls.iter().any(|tc| {
                tools.iter().any(|t| t.definition.name == tc.name && t.is_active())
            });
            if !has_active {
                break; // All passive — return to caller
            }

            // Execute active tools concurrently
            let call_refs: Vec<&content::ToolCallData> = tool_calls.into_iter().collect();
            let tool_results = execute_all_tools(
                tools,
                &call_refs,
                &conversation,
                &abort_signal,
                options.repair_tool_call.as_ref(),
                options.validate_tool_args,
            )
            .await;

            // Extend conversation for next round
            conversation.push(response.message.clone()); // assistant message with tool calls
            for result in &tool_results {
                conversation.push(Message::tool_result(
                    &result.tool_call_id,
                    result.content.to_string(),
                    result.is_error,
                ));
            }

            // Build StepResult for step tracking and stop_when callback
            let step = build_step_result(&response, tool_results);
            steps.push(step);

            // Emit StepFinish event to mark end of this tool round
            yield Ok(StreamEvent {
                event_type: StreamEventType::StepFinish,
                raw: Some(serde_json::json!({
                    "step": round,
                    "tool_calls": call_refs.len(),
                })),
                usage: Some(response.usage.clone()),
                finish_reason: Some(response.finish_reason.clone()),
                ..Default::default()
            });

            // Check stop_when callback after each tool round
            if let Some(ref stop_fn) = stop_when {
                if stop_fn(&steps) {
                    yield Ok(StreamEvent {
                        event_type: StreamEventType::Finish,
                        finish_reason: Some(FinishReason {
                            reason: "stop_when".to_string(),
                            raw: Some("stop_when_callback".to_string()),
                        }),
                        ..Default::default()
                    });
                    return;
                }
            }

            // Loop continues — next iteration starts a new stream
        }
    };

    Ok(StreamResult::new(Box::pin(event_stream)))
}

/// Stream text using the default client.
///
/// Returns a `StreamResult<'static>` by capturing `Arc<Client>` internally
/// via channel decoupling. Falls back to `Client::from_env()` if no default
/// client has been set.
///
/// # Errors
/// Returns `ConfigurationError` if no default client is set and no API keys
/// are found in the environment.
///
/// Spec reference: §4.4
pub fn stream_with_default(options: GenerateOptions) -> Result<StreamResult<'static>, Error> {
    let client = crate::default_client::get_default_client()?;

    // Validate synchronously before entering the async stream (same as stream()).
    if let Some(ref tools) = options.tools {
        for tool in tools {
            tool.definition.validate()?;
        }
    }
    if let Some(ref tc) = options.tool_choice {
        tc.validate()?;
    }

    // Channel-decouple: spawn a task that owns the Arc<Client> and forwards
    // events through an mpsc channel so the returned stream is 'static.
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<StreamEvent, Error>>(32);

    tokio::spawn(async move {
        let mut result = match stream(options, &client) {
            Ok(r) => r,
            Err(e) => {
                let _ = tx.send(Err(e)).await;
                return;
            }
        };
        use futures::StreamExt;
        while let Some(event) = result.next().await {
            if tx.send(event).await.is_err() {
                break; // Receiver dropped
            }
        }
    });

    let rx_stream = tokio_stream::wrappers::ReceiverStream::new(rx);
    Ok(StreamResult::new(Box::pin(rx_stream)))
}

/// Result of a streaming generation operation.
///
/// Implements `Stream<Item = Result<StreamEvent, Error>>` for async iteration.
/// After the stream completes, call `.response()` for the accumulated Response.
pub struct StreamResult<'a> {
    /// The inner event stream.
    inner: BoxStream<'a, Result<StreamEvent, Error>>,
    /// Accumulates events into a complete Response.
    accumulator: StreamAccumulator,
    /// Whether the stream has finished.
    finished: bool,
}

impl<'a> StreamResult<'a> {
    pub(crate) fn new(inner: BoxStream<'a, Result<StreamEvent, Error>>) -> Self {
        Self {
            inner,
            accumulator: StreamAccumulator::new(),
            finished: false,
        }
    }

    /// Get the accumulated Response after the stream ends.
    /// Returns None if the stream has not finished.
    pub fn response(&self) -> Option<Response> {
        if self.finished {
            self.accumulator.response()
        } else {
            None
        }
    }

    /// Returns the response accumulated so far, even if the stream hasn't finished.
    /// Returns None if no events have been processed yet.
    pub fn partial_response(&self) -> Option<Response> {
        self.accumulator.current_response()
    }

    /// Create a filtered stream that yields only text deltas as strings.
    ///
    /// Borrows `self` mutably, so the `StreamResult` remains accessible after
    /// the `TextDeltaStream` is dropped. This lets callers use `text_stream()`
    /// for convenient iteration and then call `response()` afterwards.
    pub fn text_stream(&mut self) -> TextDeltaStream<'_, 'a> {
        TextDeltaStream { inner: self }
    }
}

// StreamResult is Unpin because BoxStream (Pin<Box<dyn Stream + Send>>) is Unpin,
// StreamAccumulator is Unpin, and bool is Unpin.
impl<'a> Stream for StreamResult<'a> {
    type Item = Result<StreamEvent, Error>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        match this.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(mut event))) => {
                this.accumulator.process(&event);
                if event.event_type == StreamEventType::Finish {
                    this.finished = true;
                    // C-3: Attach the accumulated response to the Finish event
                    // so consumers can access the complete response directly
                    // from the event without calling .response() separately.
                    if event.response.is_none() {
                        if let Some(response) = this.accumulator.response() {
                            event.response = Some(Box::new(response));
                        }
                    }
                }
                Poll::Ready(Some(Ok(event)))
            }
            Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(e))),
            Poll::Ready(None) => {
                this.finished = true;
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

/// A filtered stream that yields only text delta strings.
///
/// Borrows a `StreamResult` mutably, so the `StreamResult` remains usable
/// (e.g. for `.response()`) after this stream is dropped.
pub struct TextDeltaStream<'b, 'a: 'b> {
    inner: &'b mut StreamResult<'a>,
}

impl<'b, 'a: 'b> Stream for TextDeltaStream<'b, 'a> {
    type Item = Result<String, Error>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            match Pin::new(&mut *this.inner).poll_next(cx) {
                Poll::Ready(Some(Ok(event))) => {
                    if event.event_type == StreamEventType::TextDelta {
                        if let Some(delta) = event.delta {
                            return Poll::Ready(Some(Ok(delta)));
                        }
                    }
                    // Skip non-text-delta events
                    continue;
                }
                Poll::Ready(Some(Err(e))) => return Poll::Ready(Some(Err(e))),
                Poll::Ready(None) => return Poll::Ready(None),
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::MockProvider;

    fn make_client_with_mock(mock: MockProvider) -> Client {
        Client::builder()
            .provider("mock", Box::new(mock))
            .build()
            .unwrap()
    }

    /// Helper: make a standard set of stream events for text streaming.
    fn make_text_stream_events(text_deltas: &[&str]) -> Vec<StreamEvent> {
        let mut events = vec![
            StreamEvent {
                event_type: StreamEventType::StreamStart,
                id: Some("stream_1".into()),
                ..Default::default()
            },
            StreamEvent {
                event_type: StreamEventType::TextStart,
                ..Default::default()
            },
        ];
        for delta in text_deltas {
            events.push(StreamEvent {
                event_type: StreamEventType::TextDelta,
                delta: Some((*delta).into()),
                ..Default::default()
            });
        }
        events.push(StreamEvent {
            event_type: StreamEventType::TextEnd,
            ..Default::default()
        });
        events.push(StreamEvent {
            event_type: StreamEventType::Finish,
            finish_reason: Some(FinishReason::stop()),
            usage: Some(Usage {
                input_tokens: 10,
                output_tokens: 5,
                total_tokens: 15,
                ..Default::default()
            }),
            ..Default::default()
        });
        events
    }

    // --- DoD 8.4.4: stream() yields TEXT_DELTA events ---

    #[tokio::test]
    async fn test_stream_yields_text_deltas() {
        let events = make_text_stream_events(&["Hello", " ", "world"]);
        let mock = MockProvider::new("mock").with_stream_events(events);
        let client = make_client_with_mock(mock);
        let opts = GenerateOptions::new("test-model").prompt("Hi");

        let mut result = stream(opts, &client).unwrap();
        let mut deltas = Vec::new();
        while let Some(event) = result.next().await {
            let event = event.unwrap();
            if event.event_type == StreamEventType::TextDelta {
                deltas.push(event.delta.unwrap());
            }
        }
        assert_eq!(deltas, vec!["Hello", " ", "world"]);
    }

    // --- DoD 8.4.5: stream() has STREAM_START and FINISH events ---

    #[tokio::test]
    async fn test_stream_has_start_and_finish() {
        let events = make_text_stream_events(&["Hi"]);
        let mock = MockProvider::new("mock").with_stream_events(events);
        let client = make_client_with_mock(mock);
        let opts = GenerateOptions::new("test-model").prompt("Hello");

        let mut result = stream(opts, &client).unwrap();
        let mut event_types = Vec::new();
        while let Some(event) = result.next().await {
            event_types.push(event.unwrap().event_type);
        }
        assert!(event_types.contains(&StreamEventType::StreamStart));
        assert!(event_types.contains(&StreamEventType::Finish));
    }

    // --- DoD 8.4.6: TextStart/Delta/End lifecycle ---

    #[tokio::test]
    async fn test_stream_start_delta_end_pattern() {
        let events = make_text_stream_events(&["Hello"]);
        let mock = MockProvider::new("mock").with_stream_events(events);
        let client = make_client_with_mock(mock);
        let opts = GenerateOptions::new("test-model").prompt("Hi");

        let mut result = stream(opts, &client).unwrap();
        let mut event_types = Vec::new();
        while let Some(event) = result.next().await {
            event_types.push(event.unwrap().event_type);
        }

        // Verify the order: StreamStart, TextStart, TextDelta, TextEnd, Finish
        let text_start_idx = event_types
            .iter()
            .position(|t| *t == StreamEventType::TextStart);
        let text_delta_idx = event_types
            .iter()
            .position(|t| *t == StreamEventType::TextDelta);
        let text_end_idx = event_types
            .iter()
            .position(|t| *t == StreamEventType::TextEnd);

        assert!(text_start_idx.is_some(), "TextStart should be present");
        assert!(text_delta_idx.is_some(), "TextDelta should be present");
        assert!(text_end_idx.is_some(), "TextEnd should be present");

        assert!(
            text_start_idx.unwrap() < text_delta_idx.unwrap(),
            "TextStart should come before TextDelta"
        );
        assert!(
            text_delta_idx.unwrap() < text_end_idx.unwrap(),
            "TextDelta should come before TextEnd"
        );
    }

    // --- StreamResult.response() returns accumulated Response ---

    #[tokio::test]
    async fn test_stream_result_response() {
        let events = make_text_stream_events(&["Hello", " world"]);
        let mock = MockProvider::new("mock").with_stream_events(events);
        let client = make_client_with_mock(mock);
        let opts = GenerateOptions::new("test-model").prompt("Hi");

        let mut result = stream(opts, &client).unwrap();

        // Before consuming, response should be None
        assert!(result.response().is_none());

        // Consume all events
        while let Some(_event) = result.next().await {}

        // After consuming, response should be available
        let response = result
            .response()
            .expect("response should be available after stream finishes");
        assert_eq!(response.text(), "Hello world");
        assert_eq!(response.finish_reason.reason, "stop");
        assert_eq!(response.usage.input_tokens, 10);
        assert_eq!(response.usage.output_tokens, 5);
        assert_eq!(response.usage.total_tokens, 15);
    }

    // --- C-3: FINISH event carries accumulated response ---

    #[tokio::test]
    async fn test_finish_event_has_response() {
        let events = make_text_stream_events(&["Hello", " world"]);
        let mock = MockProvider::new("mock").with_stream_events(events);
        let client = make_client_with_mock(mock);
        let opts = GenerateOptions::new("test-model").prompt("Hi");

        let mut result = stream(opts, &client).unwrap();
        let mut finish_event = None;

        while let Some(event) = result.next().await {
            let event = event.unwrap();
            if event.event_type == StreamEventType::Finish {
                finish_event = Some(event);
            }
        }

        let finish = finish_event.expect("Should have a Finish event");
        assert!(
            finish.response.is_some(),
            "FINISH event must have response field populated (spec §3.13)"
        );
        let response = finish.response.unwrap();
        assert_eq!(response.text(), "Hello world");
        assert_eq!(response.finish_reason.reason, "stop");
        assert_eq!(response.usage.input_tokens, 10);
        assert_eq!(response.usage.output_tokens, 5);
        assert_eq!(response.usage.total_tokens, 15);
    }

    // --- text_stream() convenience yields only text strings ---

    #[tokio::test]
    async fn test_stream_text_stream_convenience() {
        let events = make_text_stream_events(&["Hello", " ", "world"]);
        let mock = MockProvider::new("mock").with_stream_events(events);
        let client = make_client_with_mock(mock);
        let opts = GenerateOptions::new("test-model").prompt("Hi");

        let mut result = stream(opts, &client).unwrap();
        let mut text_stream = result.text_stream();

        let mut texts = Vec::new();
        while let Some(text) = text_stream.next().await {
            texts.push(text.unwrap());
        }
        assert_eq!(texts, vec!["Hello", " ", "world"]);
    }

    // --- C-4: text_stream() does not consume StreamResult ---

    #[tokio::test]
    async fn test_text_stream_does_not_consume_stream_result() {
        // After using text_stream(), response() should still work.
        let events = make_text_stream_events(&["Hello", " world"]);
        let mock = MockProvider::new("mock").with_stream_events(events);
        let client = make_client_with_mock(mock);
        let opts = GenerateOptions::new("test-model").prompt("Hi");

        let mut result = stream(opts, &client).unwrap();

        // Use text_stream() for convenient iteration
        {
            let mut text_stream = result.text_stream();
            let mut texts = Vec::new();
            while let Some(text) = text_stream.next().await {
                texts.push(text.unwrap());
            }
            assert_eq!(texts, vec!["Hello", " world"]);
        } // text_stream dropped here

        // After text_stream is dropped, we should still be able to access response()
        let response = result
            .response()
            .expect("response should be available after text_stream consumed all events");
        assert_eq!(response.text(), "Hello world");
        assert_eq!(response.finish_reason.reason, "stop");
        assert_eq!(response.usage.input_tokens, 10);
    }

    // --- DoD 8.8.9: Stream retry rules ---

    #[tokio::test]
    async fn test_stream_retries_initial_connection_error() {
        // First stream() call: immediate error (simulates connection failure)
        // Second stream() call: success
        // The retry should transparently recover.
        let mock = MockProvider::new("mock")
            .with_stream_error(Error::from_http_status(
                429,
                "rate limited".into(),
                "mock",
                None,
                None,
            ))
            .with_stream_events(make_text_stream_events(&["Hello"]));
        let client = make_client_with_mock(mock);
        let opts = GenerateOptions::new("test-model")
            .prompt("Hi")
            .max_retries(2);

        let mut result = stream(opts, &client).unwrap();
        let mut deltas = Vec::new();
        while let Some(event) = result.next().await {
            let event = event.unwrap();
            if event.event_type == StreamEventType::TextDelta {
                deltas.push(event.delta.unwrap());
            }
        }
        assert_eq!(deltas, vec!["Hello"]);
    }

    #[tokio::test]
    async fn test_stream_no_retry_after_partial_data() {
        // DoD 8.8.9: Some events yielded, then error → Error event emitted, no retry
        // Use a stream that yields: StreamStart, TextDelta("Hello"), then Err(500)
        // Followed by a success stream that should NOT be consumed (no retry).
        let partial_events: Vec<Result<StreamEvent, Error>> = vec![
            Ok(StreamEvent {
                event_type: StreamEventType::StreamStart,
                ..Default::default()
            }),
            Ok(StreamEvent {
                event_type: StreamEventType::TextDelta,
                delta: Some("Hello".into()),
                ..Default::default()
            }),
            Err(Error::from_http_status(
                500,
                "mid-stream error".into(),
                "mock",
                None,
                None,
            )),
        ];

        // Build mock with partial events manually (using the raw stream_actions queue)
        let mock = MockProvider::new("mock")
            // The second stream (should NOT be reached due to no-retry-after-partial)
            .with_stream_events(make_text_stream_events(&["Should not reach"]));

        // Push the partial+error stream directly
        mock.stream_actions
            .lock()
            .unwrap()
            .insert(0, partial_events);

        let client = make_client_with_mock(mock);
        let opts = GenerateOptions::new("test-model")
            .prompt("Hi")
            .max_retries(3); // Retries available, but should NOT be used after partial data

        let mut result = stream(opts, &client).unwrap();
        let mut collected_types = Vec::new();
        let mut had_error_event = false;

        while let Some(item) = result.next().await {
            match item {
                Ok(event) => {
                    collected_types.push(event.event_type.clone());
                    if event.event_type == StreamEventType::Error {
                        had_error_event = true;
                    }
                }
                Err(_) => {
                    // Should NOT happen — errors after partial data become Error events
                    panic!("Should not get Err after partial data, should get Error event instead");
                }
            }
        }

        // Verify we got the partial events before the error
        assert!(collected_types.contains(&StreamEventType::StreamStart));
        assert!(collected_types.contains(&StreamEventType::TextDelta));
        // Verify an Error event was emitted (not an Err return)
        assert!(had_error_event, "Should have emitted an Error event");
        // Verify we did NOT get any events from the second stream
        assert!(
            !collected_types.contains(&StreamEventType::Finish),
            "Should NOT have reached the success stream"
        );
    }

    // --- P3-T13: Streaming with Tools ---

    /// Helper: make stream events that simulate a tool call response.
    /// Produces: StreamStart, ToolCallStart, ToolCallDelta, ToolCallEnd, Finish(tool_calls)
    fn make_tool_call_stream_events(
        tool_name: &str,
        call_id: &str,
        args_json: &str,
    ) -> Vec<StreamEvent> {
        vec![
            StreamEvent {
                event_type: StreamEventType::StreamStart,
                id: Some(format!("stream_{}", call_id)),
                ..Default::default()
            },
            StreamEvent {
                event_type: StreamEventType::ToolCallStart,
                tool_call: Some(crate::ToolCall {
                    id: call_id.to_string(),
                    name: tool_name.to_string(),
                    arguments: serde_json::Map::new(),
                    raw_arguments: None,
                }),
                ..Default::default()
            },
            StreamEvent {
                event_type: StreamEventType::ToolCallDelta,
                delta: Some(args_json.into()),
                ..Default::default()
            },
            StreamEvent {
                event_type: StreamEventType::ToolCallEnd,
                ..Default::default()
            },
            StreamEvent {
                event_type: StreamEventType::Finish,
                finish_reason: Some(FinishReason::tool_calls()),
                usage: Some(Usage {
                    input_tokens: 10,
                    output_tokens: 5,
                    total_tokens: 15,
                    ..Default::default()
                }),
                ..Default::default()
            },
        ]
    }

    #[tokio::test]
    async fn test_stream_with_active_tool_executes_and_continues() {
        // DoD 8.7.1 streaming: Mock returns stream with tool calls, then stream with final text.
        // Verify: first stream events yielded, tools executed, second stream events yielded.
        let tool = super::super::types::Tool::active(
            "echo",
            "echoes input",
            serde_json::json!({"type": "object"}),
            |args| Box::pin(async move { Ok(args) }),
        );

        let mock = MockProvider::new("mock")
            .with_stream_events(make_tool_call_stream_events(
                "echo",
                "call_1",
                r#"{"text":"hello"}"#,
            ))
            .with_stream_events(make_text_stream_events(&["Echoed: hello"]));
        let client = make_client_with_mock(mock);

        let opts = GenerateOptions::new("test-model")
            .prompt("echo hello")
            .tools(vec![tool])
            .max_tool_rounds(1);

        let mut result = stream(opts, &client).unwrap();
        let mut text_deltas = Vec::new();
        let mut event_types = Vec::new();
        while let Some(event) = result.next().await {
            let event = event.unwrap();
            event_types.push(event.event_type.clone());
            if event.event_type == StreamEventType::TextDelta {
                if let Some(d) = &event.delta {
                    text_deltas.push(d.clone());
                }
            }
        }
        // Should have events from BOTH streams
        assert!(
            text_deltas.contains(&"Echoed: hello".to_string()),
            "Should have text from the second stream after tool execution"
        );
        // Should have tool call events from first stream
        assert!(
            event_types.contains(&StreamEventType::ToolCallStart),
            "Should have tool call events from first stream"
        );
        // Should have Finish events (at least the final one)
        let finish_count = event_types
            .iter()
            .filter(|t| **t == StreamEventType::Finish)
            .count();
        assert!(finish_count >= 1, "Should have at least one Finish event");
    }

    #[tokio::test]
    async fn test_stream_with_passive_tool_no_loop() {
        // Passive tool → stream returns tool calls, no second stream started.
        let tool = super::super::types::Tool::passive(
            "search",
            "search web",
            serde_json::json!({"type": "object"}),
        );

        let mock = MockProvider::new("mock")
            .with_stream_events(make_tool_call_stream_events(
                "search",
                "call_1",
                r#"{"query":"rust"}"#,
            ))
            .with_stream_events(make_text_stream_events(&["Should not reach"]));
        let client = make_client_with_mock(mock);

        let opts = GenerateOptions::new("test-model")
            .prompt("search something")
            .tools(vec![tool])
            .max_tool_rounds(3);

        let mut result = stream(opts, &client).unwrap();
        let mut event_types = Vec::new();
        while let Some(event) = result.next().await {
            let event = event.unwrap();
            event_types.push(event.event_type.clone());
        }
        // Should NOT have text deltas from the second stream
        assert!(
            !event_types.contains(&StreamEventType::TextStart),
            "Should NOT have reached the second stream (passive tool, no loop)"
        );
        // Should have tool call events from first stream
        assert!(event_types.contains(&StreamEventType::ToolCallStart));
        // Should have exactly one Finish event (from the first stream only)
        let finish_count = event_types
            .iter()
            .filter(|t| **t == StreamEventType::Finish)
            .count();
        assert_eq!(finish_count, 1, "Should have exactly one Finish event");
    }

    #[tokio::test]
    async fn test_stream_multi_step_events_continuous() {
        // Verify events from both steps appear in a single continuous iteration.
        // The consumer should see tool call events from step 1, then text events from step 2,
        // all from a single stream iteration.
        let tool = super::super::types::Tool::active(
            "echo",
            "echoes input",
            serde_json::json!({"type": "object"}),
            |args| Box::pin(async move { Ok(args) }),
        );

        let mock = MockProvider::new("mock")
            .with_stream_events(make_tool_call_stream_events(
                "echo",
                "call_1",
                r#"{"text":"hi"}"#,
            ))
            .with_stream_events(make_text_stream_events(&["Final", " answer"]));
        let client = make_client_with_mock(mock);

        let opts = GenerateOptions::new("test-model")
            .prompt("go")
            .tools(vec![tool])
            .max_tool_rounds(1);

        let mut result = stream(opts, &client).unwrap();
        let mut all_events = Vec::new();
        while let Some(event) = result.next().await {
            all_events.push(event.unwrap());
        }

        // Verify events from step 1 (tool call) appear before events from step 2 (text)
        let tool_call_start_idx = all_events
            .iter()
            .position(|e| e.event_type == StreamEventType::ToolCallStart);
        let final_text_delta_idx = all_events
            .iter()
            .rposition(|e| e.event_type == StreamEventType::TextDelta);

        assert!(
            tool_call_start_idx.is_some(),
            "Should have ToolCallStart from step 1"
        );
        assert!(
            final_text_delta_idx.is_some(),
            "Should have TextDelta from step 2"
        );
        assert!(
            tool_call_start_idx.unwrap() < final_text_delta_idx.unwrap(),
            "Step 1 events should come before step 2 events"
        );

        // Verify text deltas from step 2
        let text_deltas: Vec<String> = all_events
            .iter()
            .filter(|e| e.event_type == StreamEventType::TextDelta)
            .filter_map(|e| e.delta.clone())
            .collect();
        assert_eq!(text_deltas, vec!["Final", " answer"]);
    }

    #[tokio::test]
    async fn test_stream_max_retries_zero_no_retry() {
        // max_retries=0 → error propagated immediately, no retry attempt
        let mock = MockProvider::new("mock")
            .with_stream_error(Error::from_http_status(
                429,
                "rate limited".into(),
                "mock",
                None,
                None,
            ))
            .with_stream_events(make_text_stream_events(&["Should not reach"]));
        let client = make_client_with_mock(mock);
        let opts = GenerateOptions::new("test-model")
            .prompt("Hi")
            .max_retries(0);

        let mut result = stream(opts, &client).unwrap();
        let first_item = result.next().await;
        assert!(first_item.is_some());
        let err = first_item.unwrap().unwrap_err();
        assert_eq!(err.kind, ErrorKind::RateLimit);
    }

    // --- DoD 8.4.9: Cancellation via abort signal ---

    #[tokio::test]
    async fn test_stream_abort() {
        // 8.4.9: Pre-cancelled token → AbortError event for stream()
        use tokio_util::sync::CancellationToken;
        let token = CancellationToken::new();
        token.cancel(); // Cancel immediately
        let mock = MockProvider::new("mock")
            .with_stream_events(make_text_stream_events(&["Should not reach"]));
        let client = make_client_with_mock(mock);
        let opts = GenerateOptions::new("test")
            .prompt("hi")
            .abort_signal(token);
        let mut result = stream(opts, &client).unwrap();
        let mut saw_abort = false;
        while let Some(event) = result.next().await {
            if let Err(e) = event {
                assert_eq!(e.kind, ErrorKind::Abort);
                saw_abort = true;
                break;
            }
        }
        assert!(saw_abort, "Expected abort error in stream");
    }

    // --- AF-10: StepFinish emitted between tool loop rounds ---

    #[tokio::test]
    async fn test_stream_step_finish_emitted_between_rounds() {
        // Two-round tool loop: tool call stream → tool execution → text stream.
        // Should emit a StepFinish event between the two rounds.
        let tool = super::super::types::Tool::active(
            "echo",
            "echoes input",
            serde_json::json!({"type": "object"}),
            |args| Box::pin(async move { Ok(args) }),
        );

        let mock = MockProvider::new("mock")
            .with_stream_events(make_tool_call_stream_events(
                "echo",
                "call_1",
                r#"{"text":"hi"}"#,
            ))
            .with_stream_events(make_text_stream_events(&["Final answer"]));
        let client = make_client_with_mock(mock);

        let opts = GenerateOptions::new("test-model")
            .prompt("go")
            .tools(vec![tool])
            .max_tool_rounds(1);

        let mut result = stream(opts, &client).unwrap();
        let mut event_types = Vec::new();
        while let Some(event) = result.next().await {
            event_types.push(event.unwrap().event_type.clone());
        }

        // StepFinish should appear between the first Finish (from tool call stream)
        // and the second stream's events
        assert!(
            event_types.contains(&StreamEventType::StepFinish),
            "StepFinish event should be emitted between tool loop rounds. Got: {:?}",
            event_types
        );

        // Verify ordering: StepFinish comes after the first Finish but before the final Finish
        let step_finish_idx = event_types
            .iter()
            .position(|t| *t == StreamEventType::StepFinish)
            .unwrap();
        let final_finish_idx = event_types
            .iter()
            .rposition(|t| *t == StreamEventType::Finish)
            .unwrap();
        assert!(
            step_finish_idx < final_finish_idx,
            "StepFinish should come before the final Finish event"
        );
    }

    // --- AF-11: stop_when halts streaming tool loop ---

    #[tokio::test]
    async fn test_stream_stop_when_terminates_early() {
        // 3-round tool loop with stop_when that triggers after 1 step.
        // Should see only 1 StepFinish, then Finish with reason "stop_when".
        let tool = super::super::types::Tool::active(
            "echo",
            "echoes input",
            serde_json::json!({"type": "object"}),
            |args| Box::pin(async move { Ok(args) }),
        );

        let mock = MockProvider::new("mock")
            .with_stream_events(make_tool_call_stream_events(
                "echo",
                "call_1",
                r#"{"text":"hi"}"#,
            ))
            // Second stream would be another tool call, but stop_when should prevent reaching it
            .with_stream_events(make_tool_call_stream_events(
                "echo",
                "call_2",
                r#"{"text":"bye"}"#,
            ))
            .with_stream_events(make_text_stream_events(&["Should not reach"]));
        let client = make_client_with_mock(mock);

        let opts = GenerateOptions::new("test-model")
            .prompt("go")
            .tools(vec![tool])
            .max_tool_rounds(5)
            .stop_when(|_steps| true); // Stop after any step

        let mut result = stream(opts, &client).unwrap();
        let mut event_types = Vec::new();
        let mut finish_reasons = Vec::new();
        while let Some(event) = result.next().await {
            let event = event.unwrap();
            if let Some(ref fr) = event.finish_reason {
                finish_reasons.push(fr.reason.clone());
            }
            event_types.push(event.event_type.clone());
        }

        // Should have StepFinish from the first round
        let step_finish_count = event_types
            .iter()
            .filter(|t| **t == StreamEventType::StepFinish)
            .count();
        assert_eq!(
            step_finish_count, 1,
            "Should have exactly 1 StepFinish (stop_when fired after first round)"
        );

        // The last Finish event should indicate stop_when
        assert!(
            finish_reasons.iter().any(|r| r == "stop_when"),
            "Final Finish should have reason 'stop_when'. Got reasons: {:?}",
            finish_reasons
        );
    }

    // --- AF-12: Stream timeouts ---

    #[tokio::test]
    async fn test_stream_total_timeout_fires() {
        // Stream with a slow mock (each event delayed 100ms) and a total timeout of 50ms.
        // With paused time, we advance time to trigger the timeout deterministically.
        tokio::time::pause();

        let events = make_text_stream_events(&["Hello", " world"]);
        let mock = MockProvider::new("mock")
            .with_stream_events(events)
            .with_stream_delay(std::time::Duration::from_millis(100));
        let client = make_client_with_mock(mock);

        let opts = GenerateOptions::new("test-model")
            .prompt("Hi")
            .timeout(TimeoutConfig {
                total: Some(0.05), // 50ms total timeout
                per_step: None,
            });

        let mut result = stream(opts, &client).unwrap();
        let mut had_timeout_error = false;
        while let Some(event) = result.next().await {
            match event {
                Ok(evt) if evt.event_type == StreamEventType::Error => {
                    if let Some(ref err) = evt.error {
                        if err.message.contains("timeout") || err.message.contains("Timeout") {
                            had_timeout_error = true;
                        }
                    }
                }
                Err(e) if e.message.contains("timeout") || e.message.contains("Timeout") => {
                    had_timeout_error = true;
                }
                _ => {}
            }
        }
        assert!(
            had_timeout_error,
            "Total timeout should fire when mock stream exceeds total duration"
        );
    }

    #[tokio::test]
    async fn test_stream_no_timeout_completes_normally() {
        // Stream without timeout config. Should complete normally.
        let events = make_text_stream_events(&["Hello"]);
        let mock = MockProvider::new("mock").with_stream_events(events);
        let client = make_client_with_mock(mock);
        let opts = GenerateOptions::new("test-model").prompt("Hi");
        // No timeout set

        let mut result = stream(opts, &client).unwrap();
        let mut completed = false;
        while let Some(event) = result.next().await {
            let event = event.unwrap();
            if event.event_type == StreamEventType::Finish {
                completed = true;
            }
        }
        assert!(completed, "Stream should complete normally without timeout");
    }

    // --- AF-13: partial_response() on StreamResult ---

    #[tokio::test]
    async fn test_stream_result_partial_response() {
        let events = make_text_stream_events(&["Hello", " world"]);
        let mock = MockProvider::new("mock").with_stream_events(events);
        let client = make_client_with_mock(mock);
        let opts = GenerateOptions::new("test-model").prompt("Hi");

        let mut result = stream(opts, &client).unwrap();

        // Before any events: partial_response should be None
        assert!(result.partial_response().is_none());

        // Consume first few events (StreamStart, TextStart, TextDelta("Hello"))
        let mut consumed = 0;
        while let Some(event) = result.next().await {
            let event = event.unwrap();
            consumed += 1;
            if event.event_type == StreamEventType::TextDelta && consumed >= 3 {
                break;
            }
        }

        // Mid-stream: partial_response should return current accumulated state
        let partial = result.partial_response();
        assert!(
            partial.is_some(),
            "partial_response should return Some mid-stream"
        );
        let partial = partial.unwrap();
        assert_eq!(
            partial.text(),
            "Hello",
            "Partial response should contain accumulated text so far"
        );
    }

    #[tokio::test]
    async fn test_stream_result_partial_response_matches_final() {
        let events = make_text_stream_events(&["Hello", " world"]);
        let mock = MockProvider::new("mock").with_stream_events(events);
        let client = make_client_with_mock(mock);
        let opts = GenerateOptions::new("test-model").prompt("Hi");

        let mut result = stream(opts, &client).unwrap();

        // Consume all events
        while let Some(_) = result.next().await {}

        // After stream ends, partial_response and response should both work
        let partial = result.partial_response();
        let full = result.response();
        assert!(partial.is_some());
        assert!(full.is_some());
        assert_eq!(partial.unwrap().text(), full.unwrap().text());
    }

    #[tokio::test]
    async fn test_stream_per_step_timeout_fires() {
        // Per-step timeout wraps each individual client.stream() call.
        // A slow mock should trigger the per-step timeout before the stream starts.
        tokio::time::pause();

        let events = make_text_stream_events(&["This should timeout"]);
        let mock = MockProvider::new("mock")
            .with_stream_events(events)
            .with_stream_delay(std::time::Duration::from_millis(500));
        let client = make_client_with_mock(mock);

        let opts = GenerateOptions::new("test-model")
            .prompt("Hi")
            .timeout(TimeoutConfig {
                total: None,
                per_step: Some(0.01), // 10ms — mock's 500ms delay will exceed this
            });

        let mut result = stream(opts, &client).unwrap();
        let mut had_timeout_error = false;
        while let Some(event) = result.next().await {
            match event {
                Ok(evt) if evt.event_type == StreamEventType::Error => {
                    if let Some(ref err) = evt.error {
                        if err.message.contains("timeout") || err.message.contains("Timeout") {
                            had_timeout_error = true;
                        }
                    }
                }
                Err(e) if e.message.contains("timeout") || e.message.contains("Timeout") => {
                    had_timeout_error = true;
                }
                _ => {}
            }
        }
        assert!(
            had_timeout_error,
            "Per-step timeout should fire when mock stream exceeds per_step duration"
        );
    }

    // --- M-9: validate() on tools/ToolChoice called automatically in stream() ---

    #[tokio::test]
    async fn test_stream_rejects_invalid_tool_name() {
        let mock = MockProvider::new("mock")
            .with_stream_events(make_text_stream_events(&["Should not reach"]));
        let client = make_client_with_mock(mock);

        let tool = super::super::types::Tool::passive(
            "1invalid",
            "bad name",
            serde_json::json!({"type": "object"}),
        );
        let opts = GenerateOptions::new("test").prompt("hi").tools(vec![tool]);
        // Extract the error before the StreamResult borrow of `client` is dropped.
        let err = stream(opts, &client).map(|_| ()).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Configuration);
        assert!(
            err.message.contains("1invalid"),
            "Error should mention the invalid name, got: {}",
            err.message
        );
    }

    #[tokio::test]
    async fn test_stream_rejects_invalid_tool_choice_mode() {
        let mock = MockProvider::new("mock")
            .with_stream_events(make_text_stream_events(&["Should not reach"]));
        let client = make_client_with_mock(mock);

        let opts = GenerateOptions::new("test")
            .prompt("hi")
            .tool_choice(ToolChoice {
                mode: "bogus_mode".into(),
                tool_name: None,
            });
        // Extract the error before the StreamResult borrow of `client` is dropped.
        let err = stream(opts, &client).map(|_| ()).unwrap_err();
        assert_eq!(err.kind, ErrorKind::UnsupportedToolChoice);
        assert!(
            err.message.contains("bogus_mode"),
            "Error should mention the invalid mode, got: {}",
            err.message
        );
    }

    // --- YELLOW-1: Mid-flight cancellation (cancel after data flows) ---

    #[tokio::test]
    async fn test_stream_abort_mid_flight() {
        // 8.4.9: Cancel AFTER receiving multiple TextDelta events (not before streaming).
        // Verifies: abort terminates cleanly mid-stream, no events after cancellation.
        use tokio_util::sync::CancellationToken;
        tokio::time::pause();

        let events = make_text_stream_events(&[
            "chunk1", "chunk2", "chunk3", "chunk4", "chunk5", "chunk6", "chunk7", "chunk8",
            "chunk9", "chunk10",
        ]);
        let mock = MockProvider::new("mock")
            .with_stream_events(events)
            .with_stream_delay(std::time::Duration::from_millis(100));
        let client = make_client_with_mock(mock);

        let token = CancellationToken::new();
        let opts = GenerateOptions::new("test-model")
            .prompt("Hi")
            .abort_signal(token.clone());

        let mut result = stream(opts, &client).unwrap();
        let mut text_deltas = Vec::new();
        let mut saw_abort = false;

        while let Some(item) = result.next().await {
            match item {
                Ok(event) => {
                    if event.event_type == StreamEventType::TextDelta {
                        text_deltas.push(event.delta.unwrap());
                        // Cancel mid-flight after receiving 3 TextDelta events
                        if text_deltas.len() == 3 {
                            token.cancel();
                        }
                    }
                }
                Err(e) => {
                    assert_eq!(e.kind, ErrorKind::Abort);
                    saw_abort = true;
                    break;
                }
            }
        }

        // Must have received at least 3 TextDelta events before the abort
        assert!(
            text_deltas.len() >= 3,
            "Expected at least 3 TextDeltas before abort, got {}",
            text_deltas.len()
        );
        assert!(
            saw_abort,
            "Stream should have yielded AbortError after mid-flight cancellation"
        );

        // After abort, stream must be terminated — no more events
        let trailing = result.next().await;
        assert!(
            trailing.is_none(),
            "No events should be delivered after abort"
        );
    }

    // --- YELLOW-8: No retry after many events (strengthens 1-event invariant) ---

    #[tokio::test]
    async fn test_stream_no_retry_after_many_events() {
        // DoD 8.8.9: Strengthen the no-retry-after-partial invariant beyond "1 event".
        // 10+ events yielded before error → Error event emitted, NOT retried.
        let mut partial_events: Vec<Result<StreamEvent, Error>> = vec![
            Ok(StreamEvent {
                event_type: StreamEventType::StreamStart,
                ..Default::default()
            }),
            Ok(StreamEvent {
                event_type: StreamEventType::TextStart,
                ..Default::default()
            }),
        ];
        // Add 10 TextDelta events before the error
        for i in 0..10 {
            partial_events.push(Ok(StreamEvent {
                event_type: StreamEventType::TextDelta,
                delta: Some(format!("chunk{}", i)),
                ..Default::default()
            }));
        }
        // Then an error mid-stream
        partial_events.push(Err(Error::from_http_status(
            500,
            "mid-stream error after many events".into(),
            "mock",
            None,
            None,
        )));

        let mock = MockProvider::new("mock")
            // Second stream (should NOT be reached due to no-retry-after-partial)
            .with_stream_events(make_text_stream_events(&["Should not reach"]));

        // Push the partial+error stream as the first stream action
        mock.stream_actions
            .lock()
            .unwrap()
            .insert(0, partial_events);

        let client = make_client_with_mock(mock);
        let opts = GenerateOptions::new("test-model")
            .prompt("Hi")
            .max_retries(3); // Retries available, but must NOT be used after partial data

        let mut result = stream(opts, &client).unwrap();
        let mut collected_types = Vec::new();
        let mut text_delta_count = 0;
        let mut had_error_event = false;

        while let Some(item) = result.next().await {
            match item {
                Ok(event) => {
                    collected_types.push(event.event_type.clone());
                    if event.event_type == StreamEventType::TextDelta {
                        text_delta_count += 1;
                    }
                    if event.event_type == StreamEventType::Error {
                        had_error_event = true;
                    }
                }
                Err(_) => {
                    panic!("Should not get Err after partial data, should get Error event instead");
                }
            }
        }

        // Verify we received all 10 TextDelta events before the error
        assert_eq!(
            text_delta_count, 10,
            "Should have received all 10 TextDelta events before the error"
        );
        // Verify an Error event was emitted (not an Err return)
        assert!(
            had_error_event,
            "Should have emitted an Error event, not retried"
        );
        // Verify we did NOT get any events from the retry stream
        assert!(
            !collected_types.contains(&StreamEventType::Finish),
            "Should NOT have retried — no Finish event from the retry stream"
        );
    }

    // --- YELLOW-12: Combined total + per-step timeout (shorter wins) ---

    #[tokio::test]
    async fn test_combined_total_and_per_step_timeout() {
        // Both total and per_step configured; the shorter one should fire first.
        // per_step = 20ms, total = 100ms, mock delay = 500ms → per_step wins.
        tokio::time::pause();

        let events = make_text_stream_events(&["This should timeout"]);
        let mock = MockProvider::new("mock")
            .with_stream_events(events)
            .with_stream_delay(std::time::Duration::from_millis(500));
        let client = make_client_with_mock(mock);

        let opts = GenerateOptions::new("test-model")
            .prompt("Hi")
            .timeout(TimeoutConfig {
                total: Some(0.10),    // 100ms — the longer timeout
                per_step: Some(0.02), // 20ms — the shorter timeout (should win)
            });

        let mut result = stream(opts, &client).unwrap();
        let mut had_timeout_error = false;
        let mut timeout_message = String::new();

        while let Some(event) = result.next().await {
            match event {
                Ok(evt) if evt.event_type == StreamEventType::Error => {
                    if let Some(ref err) = evt.error {
                        if err.message.contains("timeout") || err.message.contains("Timeout") {
                            had_timeout_error = true;
                            timeout_message.clone_from(&err.message);
                        }
                    }
                }
                Err(e) if e.message.contains("timeout") || e.message.contains("Timeout") => {
                    had_timeout_error = true;
                    timeout_message.clone_from(&e.message);
                }
                _ => {}
            }
        }

        assert!(
            had_timeout_error,
            "Should have received a timeout error when both timeouts are configured"
        );
        // The shorter timeout (per_step = 20ms) should win over the longer (total = 100ms)
        assert!(
            timeout_message.contains("Per-step"),
            "Per-step timeout (shorter) should fire first, got: {}",
            timeout_message
        );
    }
}
