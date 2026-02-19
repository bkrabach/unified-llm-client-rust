//! Phase 1 Integration Test: Full Anthropic stack validation.
//!
//! Tests the complete request/response cycle through Client → AnthropicAdapter
//! using wiremock to mock the Anthropic API.

use futures::StreamExt;
use secrecy::SecretString;
use unified_llm::client::ClientBuilder;
use unified_llm::providers::anthropic::AnthropicAdapter;
use unified_llm_types::*;

/// Helper: build an SSE body from event type/data pairs.
fn build_sse_body(events: &[(&str, &str)]) -> String {
    let mut body = String::new();
    for (event_type, data) in events {
        body.push_str(&format!("event: {}\ndata: {}\n\n", event_type, data));
    }
    body
}

// === Test 1: complete() through Client ===

#[tokio::test]
async fn test_phase1_complete_through_client() {
    let server = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .and(wiremock::matchers::path("/v1/messages"))
        .and(wiremock::matchers::header("x-api-key", "test-key"))
        .and(wiremock::matchers::header_exists("anthropic-version"))
        .respond_with(
            wiremock::ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "msg_integration",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Integration test passed!"}],
                "model": "claude-sonnet-4-20250514",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 15, "output_tokens": 8}
            })),
        )
        .mount(&server)
        .await;

    let adapter = AnthropicAdapter::new_with_base_url(
        SecretString::from("test-key".to_string()),
        server.uri(),
    );
    let client = ClientBuilder::new()
        .provider("anthropic", Box::new(adapter))
        .build()
        .unwrap();

    let req = Request::default()
        .model("claude-sonnet-4-20250514")
        .messages(vec![
            Message::system("You are a test assistant."),
            Message::user("Say the magic words"),
        ]);

    let resp = client.complete(req).await.unwrap();

    assert_eq!(resp.text(), "Integration test passed!");
    assert_eq!(resp.id, "msg_integration");
    assert_eq!(resp.model, "claude-sonnet-4-20250514");
    assert_eq!(resp.provider, "anthropic");
    assert_eq!(resp.finish_reason.reason, "stop");
    assert_eq!(resp.usage.input_tokens, 15);
    assert_eq!(resp.usage.output_tokens, 8);
    assert_eq!(resp.usage.total_tokens, 23);
}

// === Test 2: stream() through Client ===

#[tokio::test]
async fn test_phase1_stream_through_client() {
    let sse_body = build_sse_body(&[
        (
            "message_start",
            r#"{"type":"message_start","message":{"id":"msg_stream","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-20250514","usage":{"input_tokens":12,"output_tokens":0}}}"#,
        ),
        (
            "content_block_start",
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
        ),
        (
            "content_block_delta",
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#,
        ),
        (
            "content_block_delta",
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" from"}}"#,
        ),
        (
            "content_block_delta",
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" streaming!"}}"#,
        ),
        (
            "content_block_stop",
            r#"{"type":"content_block_stop","index":0}"#,
        ),
        (
            "message_delta",
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":6}}"#,
        ),
        ("message_stop", r#"{"type":"message_stop"}"#),
    ]);

    let server = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .and(wiremock::matchers::path("/v1/messages"))
        .respond_with(
            wiremock::ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse_body),
        )
        .mount(&server)
        .await;

    let adapter = AnthropicAdapter::new_with_base_url(
        SecretString::from("test-key".to_string()),
        server.uri(),
    );
    let client = ClientBuilder::new()
        .provider("anthropic", Box::new(adapter))
        .build()
        .unwrap();

    let req = Request::default()
        .model("claude-sonnet-4-20250514")
        .messages(vec![Message::user("Stream me")]);

    let stream = client.stream(req).unwrap();
    let events: Vec<Result<StreamEvent, Error>> = stream.collect().await;

    // Verify event sequence
    let types: Vec<StreamEventType> = events
        .iter()
        .filter_map(|e| e.as_ref().ok())
        .map(|e| e.event_type.clone())
        .collect();

    assert_eq!(
        types,
        vec![
            StreamEventType::StreamStart,
            StreamEventType::TextStart,
            StreamEventType::TextDelta,
            StreamEventType::TextDelta,
            StreamEventType::TextDelta,
            StreamEventType::TextEnd,
            StreamEventType::Finish,
        ]
    );

    // Verify text concatenation
    let text: String = events
        .iter()
        .filter_map(|e| e.as_ref().ok())
        .filter(|e| e.event_type == StreamEventType::TextDelta)
        .filter_map(|e| e.delta.clone())
        .collect();
    assert_eq!(text, "Hello from streaming!");

    // Verify StreamStart has id
    let start = events
        .iter()
        .filter_map(|e| e.as_ref().ok())
        .find(|e| e.event_type == StreamEventType::StreamStart)
        .unwrap();
    assert_eq!(start.id, Some("msg_stream".to_string()));

    // Verify Finish has usage
    let finish = events
        .iter()
        .filter_map(|e| e.as_ref().ok())
        .find(|e| e.event_type == StreamEventType::Finish)
        .unwrap();
    assert_eq!(finish.finish_reason.as_ref().unwrap().reason, "stop");
    let usage = finish.usage.as_ref().unwrap();
    assert_eq!(usage.input_tokens, 12);
    assert_eq!(usage.output_tokens, 6);
}

// === Test 3: Error handling (429 RateLimitError) ===

#[tokio::test]
async fn test_phase1_error_handling_rate_limit() {
    let server = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .and(wiremock::matchers::path("/v1/messages"))
        .respond_with(
            wiremock::ResponseTemplate::new(429)
                .insert_header("retry-after", "10")
                .set_body_json(serde_json::json!({
                    "type": "error",
                    "error": {
                        "type": "rate_limit_error",
                        "message": "Rate limit exceeded"
                    }
                })),
        )
        .mount(&server)
        .await;

    let adapter = AnthropicAdapter::new_with_base_url(
        SecretString::from("test-key".to_string()),
        server.uri(),
    );
    let client = ClientBuilder::new()
        .provider("anthropic", Box::new(adapter))
        .build()
        .unwrap();

    let req = Request::default()
        .model("test")
        .messages(vec![Message::user("Hi")]);

    let err = client.complete(req).await.unwrap_err();
    assert_eq!(err.kind, ErrorKind::RateLimit);
    assert!(err.retryable);
    assert_eq!(err.retry_after, Some(std::time::Duration::from_secs(10)));
    assert_eq!(err.provider, Some("anthropic".to_string()));
}

// === Test 4: Thinking block round-trip ===

#[tokio::test]
async fn test_phase1_thinking_block_roundtrip() {
    let server = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .and(wiremock::matchers::path("/v1/messages"))
        .respond_with(
            wiremock::ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "msg_think",
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "The user asked about the meaning of life. Let me consider various philosophical perspectives...",
                        "signature": "sig_abc123_preserve_verbatim"
                    },
                    {
                        "type": "text",
                        "text": "The answer is 42."
                    }
                ],
                "model": "claude-sonnet-4-20250514",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 25, "output_tokens": 45}
            })),
        )
        .mount(&server)
        .await;

    let adapter = AnthropicAdapter::new_with_base_url(
        SecretString::from("test-key".to_string()),
        server.uri(),
    );
    let client = ClientBuilder::new()
        .provider("anthropic", Box::new(adapter))
        .build()
        .unwrap();

    let req = Request::default()
        .model("claude-sonnet-4-20250514")
        .messages(vec![Message::user("What is the meaning of life?")])
        .provider_options(Some(serde_json::json!({
            "anthropic": {
                "thinking": {"type": "enabled", "budget_tokens": 2048}
            }
        })));

    let resp = client.complete(req).await.unwrap();

    // Verify thinking content
    assert_eq!(
        resp.reasoning(),
        Some("The user asked about the meaning of life. Let me consider various philosophical perspectives...".to_string())
    );

    // Verify text content
    assert_eq!(resp.text(), "The answer is 42.");

    // Verify signature preserved verbatim (DoD 8.5.4)
    match &resp.message.content[0] {
        ContentPart::Thinking { thinking } => {
            assert_eq!(
                thinking.signature,
                Some("sig_abc123_preserve_verbatim".to_string()),
                "Signature must round-trip verbatim"
            );
            assert!(!thinking.redacted);
        }
        other => panic!("Expected Thinking content part, got {:?}", other),
    }
}

// === Test 5: Stream with tool calls ===

#[tokio::test]
async fn test_phase1_stream_with_tool_calls() {
    let sse_body = build_sse_body(&[
        (
            "message_start",
            r#"{"type":"message_start","message":{"id":"msg_tools","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-20250514","usage":{"input_tokens":20,"output_tokens":0}}}"#,
        ),
        // Text block
        (
            "content_block_start",
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
        ),
        (
            "content_block_delta",
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Let me check the weather."}}"#,
        ),
        (
            "content_block_stop",
            r#"{"type":"content_block_stop","index":0}"#,
        ),
        // Tool call block
        (
            "content_block_start",
            r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_abc","name":"get_weather"}}"#,
        ),
        (
            "content_block_delta",
            r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"city\":\"San Francisco\",\"units\":\"fahrenheit\"}"}}"#,
        ),
        (
            "content_block_stop",
            r#"{"type":"content_block_stop","index":1}"#,
        ),
        (
            "message_delta",
            r#"{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":30}}"#,
        ),
        ("message_stop", r#"{"type":"message_stop"}"#),
    ]);

    let server = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .respond_with(
            wiremock::ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse_body),
        )
        .mount(&server)
        .await;

    let adapter = AnthropicAdapter::new_with_base_url(
        SecretString::from("test-key".to_string()),
        server.uri(),
    );
    let client = ClientBuilder::new()
        .provider("anthropic", Box::new(adapter))
        .build()
        .unwrap();

    let req = Request::default()
        .model("claude-sonnet-4-20250514")
        .messages(vec![Message::user("What's the weather in SF?")])
        .tools(vec![ToolDefinition {
            name: "get_weather".into(),
            description: "Get current weather".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "units": {"type": "string"}
                }
            }),
            strict: None,
        }]);

    let stream = client.stream(req).unwrap();
    let events: Vec<Result<StreamEvent, Error>> = stream.collect().await;

    let types: Vec<StreamEventType> = events
        .iter()
        .filter_map(|e| e.as_ref().ok())
        .map(|e| e.event_type.clone())
        .collect();

    // Verify correct event ordering
    assert_eq!(
        types,
        vec![
            StreamEventType::StreamStart,
            StreamEventType::TextStart,
            StreamEventType::TextDelta,
            StreamEventType::TextEnd,
            StreamEventType::ToolCallStart,
            StreamEventType::ToolCallDelta,
            StreamEventType::ToolCallEnd,
            StreamEventType::Finish,
        ]
    );

    // Verify tool call data
    let tc_start = events
        .iter()
        .filter_map(|e| e.as_ref().ok())
        .find(|e| e.event_type == StreamEventType::ToolCallStart)
        .unwrap();
    let tc = tc_start.tool_call.as_ref().unwrap();
    assert_eq!(tc.id, "toolu_abc");
    assert_eq!(tc.name, "get_weather");

    // Verify finish reason
    let finish = events
        .iter()
        .filter_map(|e| e.as_ref().ok())
        .find(|e| e.event_type == StreamEventType::Finish)
        .unwrap();
    assert_eq!(finish.finish_reason.as_ref().unwrap().reason, "tool_calls");
}

// === Test 6: Cache usage tokens ===

#[tokio::test]
async fn test_phase1_cache_usage_tokens() {
    let server = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .and(wiremock::matchers::path("/v1/messages"))
        .respond_with(
            wiremock::ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "msg_cache",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Cached!"}],
                "model": "claude-sonnet-4-20250514",
                "stop_reason": "end_turn",
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 5,
                    "cache_read_input_tokens": 80,
                    "cache_creation_input_tokens": 20
                }
            })),
        )
        .mount(&server)
        .await;

    let adapter = AnthropicAdapter::new_with_base_url(
        SecretString::from("test-key".to_string()),
        server.uri(),
    );
    let client = ClientBuilder::new()
        .provider("anthropic", Box::new(adapter))
        .build()
        .unwrap();

    let req = Request::default()
        .model("claude-sonnet-4-20250514")
        .messages(vec![
            Message::system("Long system prompt that benefits from caching"),
            Message::user("Hi"),
        ]);

    let resp = client.complete(req).await.unwrap();

    assert_eq!(resp.usage.cache_read_tokens, Some(80));
    assert_eq!(resp.usage.cache_write_tokens, Some(20));
    assert_eq!(resp.usage.input_tokens, 100);
    assert_eq!(resp.usage.output_tokens, 5);
}

// === Test 7: Stream error propagation ===

#[tokio::test]
async fn test_phase1_stream_error_propagation() {
    let server = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .respond_with(
            wiremock::ResponseTemplate::new(500).set_body_json(serde_json::json!({
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": "Internal server error"
                }
            })),
        )
        .mount(&server)
        .await;

    let adapter = AnthropicAdapter::new_with_base_url(
        SecretString::from("test-key".to_string()),
        server.uri(),
    );
    let client = ClientBuilder::new()
        .provider("anthropic", Box::new(adapter))
        .build()
        .unwrap();

    let req = Request::default()
        .model("test")
        .messages(vec![Message::user("Hi")]);

    let stream = client.stream(req).unwrap();
    let events: Vec<Result<StreamEvent, Error>> = stream.collect().await;

    assert_eq!(events.len(), 1);
    let err = events[0].as_ref().unwrap_err();
    assert_eq!(err.kind, ErrorKind::Server);
    assert!(err.retryable);
}
