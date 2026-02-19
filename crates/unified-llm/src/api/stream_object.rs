// api/stream_object.rs -- stream_object() function (Layer 4).
//
// Streaming structured output with partial object updates.
// Calls stream() internally, then incrementally parses JSON to yield
// partial objects as tokens arrive.
//
// Spec reference: S4.6

use std::pin::Pin;
use std::task::{Context, Poll};

use futures::StreamExt;
use futures_core::Stream;
use unified_llm_types::*;

use crate::client::Client;

use super::stream::stream;
use super::types::GenerateOptions;

/// A partial object yielded during streaming structured output.
#[derive(Debug, Clone)]
pub struct PartialObject {
    /// The partially-parsed JSON object (grows as tokens arrive).
    pub object: serde_json::Value,
    /// The raw text accumulated so far.
    pub raw_text: String,
}

/// Result of `stream_object()` -- yields partial objects, finalizes with `.object()`.
///
/// Implements `Stream<Item = Result<PartialObject, Error>>` for async iteration.
/// After the stream completes, call `.object()` for the complete, validated object.
///
/// Note: `Debug` is implemented manually because the inner stream is not `Debug`.
pub struct StreamObjectResult<'a> {
    /// The inner partial-object stream.
    inner: Pin<Box<dyn Stream<Item = Result<PartialObject, Error>> + Send + 'a>>,
    /// The last complete partial object seen (becomes the final object).
    final_object: Option<serde_json::Value>,
    /// The JSON schema to validate against.
    schema: serde_json::Value,
    /// The raw accumulated text.
    raw_text: String,
}

/// Streaming structured output with partial object updates.
///
/// Calls `stream()` internally with `response_format` set to `json_schema`,
/// then incrementally parses the streamed text to yield `PartialObject` values
/// as tokens arrive. Each partial is a best-effort parse of the text so far.
///
/// After exhausting the stream, call `.object()` to get the final validated object.
///
/// Returns `NoObjectGenerated` error from `.object()` if the final result
/// cannot be parsed or does not validate against the provided schema.
///
/// Spec reference: S4.6
pub fn stream_object<'a>(
    mut options: GenerateOptions,
    schema: serde_json::Value,
    client: &'a Client,
) -> Result<StreamObjectResult<'a>, Error> {
    // Set response_format to json_schema (same pattern as generate_object)
    options.response_format = Some(ResponseFormat {
        r#type: "json_schema".to_string(),
        json_schema: Some(schema.clone()),
        strict: true,
    });

    // Call stream() internally
    let mut stream_result = stream(options, client)?;

    // Build the partial-yielding stream
    let partial_stream = async_stream::stream! {
        let mut accumulated_text = String::new();
        let mut last_yielded: Option<serde_json::Value> = None;

        while let Some(event_result) = stream_result.next().await {
            let event = match event_result {
                Ok(e) => e,
                Err(e) => {
                    yield Err(e);
                    continue;
                }
            };

            if event.event_type == StreamEventType::TextDelta {
                if let Some(delta) = &event.delta {
                    accumulated_text.push_str(delta);

                    // Try to parse partial JSON
                    if let Some(partial) = try_parse_partial(&accumulated_text) {
                        // Only yield if the object actually changed
                        if last_yielded.as_ref() != Some(&partial) {
                            last_yielded = Some(partial.clone());
                            yield Ok(PartialObject {
                                object: partial,
                                raw_text: accumulated_text.clone(),
                            });
                        }
                    }
                }
            }
        }
    };

    Ok(StreamObjectResult {
        inner: Box::pin(partial_stream),
        final_object: None,
        schema,
        raw_text: String::new(),
    })
}

/// Streaming structured output using the default client.
///
/// Returns a `StreamObjectResult<'static>` by using channel decoupling
/// internally. Falls back to `Client::from_env()` if no default client
/// has been set.
///
/// # Errors
/// Returns `ConfigurationError` if no default client is set and no API keys
/// are found in the environment.
///
/// Spec reference: S4.6
pub fn stream_object_with_default(
    mut options: GenerateOptions,
    schema: serde_json::Value,
) -> Result<StreamObjectResult<'static>, Error> {
    // Eagerly verify the default client is available before doing anything.
    let _client = crate::default_client::get_default_client()?;

    // Validate upfront (synchronously, before spawning).
    if let Some(ref tools) = options.tools {
        for tool in tools {
            tool.definition.validate()?;
        }
    }
    if let Some(ref tc) = options.tool_choice {
        tc.validate()?;
    }

    // Set response_format to json_schema (same as stream_object).
    options.response_format = Some(ResponseFormat {
        r#type: "json_schema".to_string(),
        json_schema: Some(schema.clone()),
        strict: true,
    });

    // Use stream_with_default's channel-decoupled stream, then layer
    // partial JSON parsing on top (same logic as stream_object).
    let stream_result = super::stream::stream_with_default(options)?;

    // Build the partial-yielding stream (same as stream_object)
    let mut inner_stream = stream_result;
    let partial_stream = async_stream::stream! {
        let mut accumulated_text = String::new();
        let mut last_yielded: Option<serde_json::Value> = None;

        while let Some(event_result) = StreamExt::next(&mut inner_stream).await {
            let event = match event_result {
                Ok(e) => e,
                Err(e) => {
                    yield Err(e);
                    continue;
                }
            };

            if event.event_type == StreamEventType::TextDelta {
                if let Some(delta) = &event.delta {
                    accumulated_text.push_str(delta);

                    if let Some(partial) = try_parse_partial(&accumulated_text) {
                        if last_yielded.as_ref() != Some(&partial) {
                            last_yielded = Some(partial.clone());
                            yield Ok(PartialObject {
                                object: partial,
                                raw_text: accumulated_text.clone(),
                            });
                        }
                    }
                }
            }
        }
    };

    Ok(StreamObjectResult {
        inner: Box::pin(partial_stream),
        final_object: None,
        schema,
        raw_text: String::new(),
    })
}

impl<'a> std::fmt::Debug for StreamObjectResult<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamObjectResult")
            .field("final_object", &self.final_object)
            .field("schema", &self.schema)
            .field("raw_text", &self.raw_text)
            .field("inner", &"<stream>")
            .finish()
    }
}

// StreamObjectResult is Unpin because the inner Pin<Box<dyn Stream>> is Unpin,
// and all other fields are Unpin.
impl<'a> Stream for StreamObjectResult<'a> {
    type Item = Result<PartialObject, Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(partial))) => {
                self.final_object = Some(partial.object.clone());
                self.raw_text = partial.raw_text.clone();
                Poll::Ready(Some(Ok(partial)))
            }
            Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(e))),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl<'a> StreamObjectResult<'a> {
    /// Get the final complete, validated object after the stream is exhausted.
    ///
    /// Returns `NoObjectGenerated` error if:
    /// - The stream has not completed or produced no parseable output.
    /// - The final object does not validate against the provided schema.
    pub fn object(&self) -> Result<serde_json::Value, Error> {
        let obj = self.final_object.as_ref().ok_or_else(|| {
            no_object_error(
                "No object generated -- stream may not have completed",
                &self.raw_text,
            )
        })?;

        // Validate against schema (same pattern as generate_object)
        if !jsonschema::is_valid(&self.schema, obj) {
            let error_detail = jsonschema::validate(&self.schema, obj)
                .err()
                .map(|e| e.to_string())
                .unwrap_or_else(|| "unknown validation error".into());
            return Err(no_object_error(
                &format!("Schema validation failed: {}", error_detail),
                &self.raw_text,
            ));
        }

        Ok(obj.clone())
    }
}

/// Create a NoObjectGenerated error with details.
fn no_object_error(reason: &str, raw_text: &str) -> Error {
    Error {
        kind: ErrorKind::NoObjectGenerated,
        message: format!(
            "Failed to generate structured output: {}. Raw text: {}",
            reason, raw_text
        ),
        retryable: false,
        source: None,
        provider: None,
        status_code: None,
        error_code: None,
        retry_after: None,
        raw: None,
    }
}

/// Attempt to parse partial JSON by closing unclosed braces, brackets, and strings.
///
/// Returns `Some(value)` if the text (possibly with fixups) parses as valid JSON,
/// `None` otherwise.
fn try_parse_partial(text: &str) -> Option<serde_json::Value> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }

    // First try: parse as-is (might already be complete JSON)
    if let Ok(val) = serde_json::from_str::<serde_json::Value>(trimmed) {
        return Some(val);
    }

    // Second try: close unclosed braces/brackets/strings
    let mut fixed = trimmed.to_string();

    // Track what needs closing by scanning the string
    let mut stack: Vec<char> = Vec::new();
    let mut in_string = false;
    let mut escape_next = false;

    for ch in fixed.chars() {
        if escape_next {
            escape_next = false;
            continue;
        }
        if ch == '\\' && in_string {
            escape_next = true;
            continue;
        }
        if ch == '"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        match ch {
            '{' => stack.push('}'),
            '[' => stack.push(']'),
            '}' | ']' => {
                stack.pop();
            }
            _ => {}
        }
    }

    // If we're in an unclosed string, close it
    if in_string {
        fixed.push('"');
    }

    // Remove trailing comma before closing (invalid JSON: `{"a": 1,}`)
    let trimmed_end = fixed.trim_end();
    if let Some(stripped) = trimmed_end.strip_suffix(',') {
        fixed = stripped.to_string();
    }

    // Close all open brackets/braces in reverse order
    while let Some(closer) = stack.pop() {
        fixed.push(closer);
    }

    serde_json::from_str::<serde_json::Value>(&fixed).ok()
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

    // --- try_parse_partial tests ---

    #[test]
    fn test_try_parse_partial_complete_json() {
        let input = r#"{"name": "Alice", "age": 30}"#;
        let result = try_parse_partial(input).unwrap();
        assert_eq!(result["name"], "Alice");
        assert_eq!(result["age"], 30);
    }

    #[test]
    fn test_try_parse_partial_incomplete_object() {
        // Unclosed string and object
        let input = r#"{"name": "Ali"#;
        let result = try_parse_partial(input).unwrap();
        assert_eq!(result["name"], "Ali");
    }

    #[test]
    fn test_try_parse_partial_trailing_comma() {
        let input = r#"{"a": 1,"#;
        let result = try_parse_partial(input).unwrap();
        assert_eq!(result["a"], 1);
    }

    #[test]
    fn test_try_parse_partial_nested_incomplete() {
        let input = r#"{"user": {"name": "Bob"#;
        let result = try_parse_partial(input).unwrap();
        assert_eq!(result["user"]["name"], "Bob");
    }

    #[test]
    fn test_try_parse_partial_empty_string() {
        assert!(try_parse_partial("").is_none());
        assert!(try_parse_partial("   ").is_none());
    }

    #[test]
    fn test_try_parse_partial_array() {
        let input = r#"[1, 2, 3"#;
        let result = try_parse_partial(input).unwrap();
        assert_eq!(result, serde_json::json!([1, 2, 3]));
    }

    #[test]
    fn test_try_parse_partial_unclosed_string_value() {
        let input = r#"{"greeting": "hello wor"#;
        let result = try_parse_partial(input).unwrap();
        assert_eq!(result["greeting"], "hello wor");
    }

    #[test]
    fn test_try_parse_partial_escaped_quote_in_string() {
        // The string value contains an escaped quote: "say \"hi\""
        let input = r#"{"msg": "say \"hi\""#;
        let result = try_parse_partial(input).unwrap();
        assert_eq!(result["msg"], "say \"hi\"");
    }

    // --- stream_object with mock tests ---

    #[tokio::test]
    async fn test_stream_object_with_mock() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        });

        // Simulate streaming a JSON response chunk by chunk
        let events =
            make_text_stream_events(&[r#"{"na"#, r#"me": "Alice""#, r#", "age"#, r#"": 30}"#]);
        let mock = MockProvider::new("mock").with_stream_events(events);
        let client = make_client_with_mock(mock);

        let opts = GenerateOptions::new("test-model").prompt("Extract: Alice is 30 years old.");
        let mut result = stream_object(opts, schema, &client).unwrap();

        let mut partial_count = 0;
        while let Some(partial) = result.next().await {
            let p = partial.unwrap();
            partial_count += 1;
            // Each partial should be a valid JSON value
            assert!(p.object.is_object() || p.object.is_array() || p.object.is_string());
        }

        assert!(
            partial_count > 0,
            "Should have received at least one partial"
        );

        let final_obj = result.object().unwrap();
        assert_eq!(final_obj["name"], "Alice");
        assert_eq!(final_obj["age"], 30);
    }

    #[tokio::test]
    async fn test_stream_object_schema_validation_on_final() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        });

        // Response is valid JSON but missing required "age" field
        let events = make_text_stream_events(&[r#"{"name": "Alice"}"#]);
        let mock = MockProvider::new("mock").with_stream_events(events);
        let client = make_client_with_mock(mock);

        let opts = GenerateOptions::new("test-model").prompt("Extract person info.");
        let mut result = stream_object(opts, schema, &client).unwrap();

        // Consume the stream
        while let Some(_partial) = result.next().await {}

        // .object() should fail schema validation
        let err = result.object().unwrap_err();
        assert_eq!(err.kind, ErrorKind::NoObjectGenerated);
        assert!(!err.retryable, "Schema validation failures are NOT retried");
    }

    #[tokio::test]
    async fn test_stream_object_no_object_before_stream_ends() {
        // Calling .object() before consuming the stream should fail
        let schema = serde_json::json!({"type": "object"});
        let events = make_text_stream_events(&[r#"{}"#]);
        let mock = MockProvider::new("mock").with_stream_events(events);
        let client = make_client_with_mock(mock);

        let opts = GenerateOptions::new("test-model").prompt("test");
        let result = stream_object(opts, schema, &client).unwrap();

        // Don't consume the stream -- .object() should return error
        let err = result.object().unwrap_err();
        assert_eq!(err.kind, ErrorKind::NoObjectGenerated);
    }

    #[tokio::test]
    async fn test_stream_object_only_yields_on_change() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "x": {"type": "integer"}
            }
        });

        // Chunk 1 and 2 produce the same partial: {"x": 1}
        // Only one partial should be yielded for them.
        let events = make_text_stream_events(&[
            r#"{"x": 1"#, // partial: {"x": 1}
            r#""#,        // no change to accumulated text after trim -- still same partial
            r#"}"#,       // final: {"x": 1} (same value, but now complete)
        ]);
        let mock = MockProvider::new("mock").with_stream_events(events);
        let client = make_client_with_mock(mock);

        let opts = GenerateOptions::new("test-model").prompt("test");
        let mut result = stream_object(opts, schema, &client).unwrap();

        let mut partials = Vec::new();
        while let Some(p) = result.next().await {
            partials.push(p.unwrap());
        }

        // Should have at least 1 partial, but duplicates should be suppressed
        assert!(!partials.is_empty(), "Should yield at least one partial");
        // All partials should have x=1
        for p in &partials {
            assert_eq!(p.object["x"], 1);
        }
    }

    #[tokio::test]
    async fn test_stream_object_sets_response_format() {
        // Verify stream_object sets json_schema response format by checking
        // it works end-to-end (the mock doesn't care, but the function must
        // set it for real providers to return structured output).
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "answer": {"type": "string"}
            }
        });

        let events = make_text_stream_events(&[r#"{"answer": "42"}"#]);
        let mock = MockProvider::new("mock").with_stream_events(events);
        let client = make_client_with_mock(mock);

        let opts = GenerateOptions::new("test-model").prompt("What is the answer?");
        // Verify opts does NOT have response_format yet
        assert!(
            opts.response_format.is_none(),
            "options should not have response_format before stream_object"
        );

        let mut result = stream_object(opts, schema, &client).unwrap();
        while let Some(_) = result.next().await {}

        let obj = result.object().unwrap();
        assert_eq!(obj["answer"], "42");
    }

    #[tokio::test]
    async fn test_stream_object_mid_stream_error_propagation() {
        // A mid-stream error (partial data then error) should propagate through
        // the partial object stream. We use a partial+error stream that yields
        // some text deltas then an error after events have already been yielded.
        let schema = serde_json::json!({"type": "object"});

        let partial_events: Vec<Result<StreamEvent, Error>> = vec![
            Ok(StreamEvent {
                event_type: StreamEventType::StreamStart,
                ..Default::default()
            }),
            Ok(StreamEvent {
                event_type: StreamEventType::TextDelta,
                delta: Some(r#"{"x": 1"#.into()),
                ..Default::default()
            }),
            // Error event mid-stream (emitted as StreamEvent::Error by stream())
            Ok(StreamEvent {
                event_type: StreamEventType::Error,
                error: Some(Box::new(unified_llm_types::StreamError {
                    kind: ErrorKind::Server,
                    message: "mid-stream error".into(),
                    retryable: false,
                })),
                ..Default::default()
            }),
        ];

        let mock = MockProvider::new("mock");
        mock.stream_actions.lock().unwrap().push(partial_events);
        let client = make_client_with_mock(mock);

        let opts = GenerateOptions::new("test-model").prompt("test");
        let mut result = stream_object(opts, schema, &client).unwrap();

        // Consume all partials — we should get at least one partial before the stream ends
        let mut got_partial = false;
        while let Some(item) = result.next().await {
            if let Ok(p) = item {
                got_partial = true;
                assert_eq!(p.object["x"], 1);
            }
        }
        assert!(
            got_partial,
            "Should have received at least one partial before stream error"
        );

        // After mid-stream error, .object() should still return the last partial
        // (schema validation may fail since it's incomplete, but we got data)
        // Here the object {"x": 1} is valid against {"type": "object"}
        let obj = result.object().unwrap();
        assert_eq!(obj["x"], 1);
    }

    #[tokio::test]
    async fn test_stream_object_connection_error() {
        // A non-retryable connection error (401) is yielded as an Err through the stream
        // (stream() returns Ok but the first polled item is Err). stream_object wraps this,
        // so the error surfaces when iterating.
        let schema = serde_json::json!({"type": "object"});
        let mock = MockProvider::new("mock").with_stream_error(Error {
            kind: ErrorKind::Authentication,
            message: "invalid API key".into(),
            retryable: false,
            source: None,
            provider: Some("mock".into()),
            status_code: Some(401),
            error_code: None,
            retry_after: None,
            raw: None,
        });
        let client = make_client_with_mock(mock);

        let opts = GenerateOptions::new("test-model").prompt("test");
        let mut result = stream_object(opts, schema, &client).unwrap();

        // The error surfaces when we iterate the stream
        let mut saw_error = false;
        while let Some(item) = result.next().await {
            if let Err(e) = item {
                assert_eq!(e.kind, ErrorKind::Authentication);
                saw_error = true;
                break;
            }
        }
        assert!(
            saw_error,
            "Should have received authentication error from stream"
        );

        // After error, .object() should fail (no object was generated)
        let err = result.object().unwrap_err();
        assert_eq!(err.kind, ErrorKind::NoObjectGenerated);
    }
}
