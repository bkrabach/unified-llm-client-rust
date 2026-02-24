//! Phase 2D: Cross-Provider Conformance Tests
//!
//! These tests run the SAME assertions against all 3 provider adapters (Anthropic, OpenAI, Gemini)
//! using wiremock, covering DoD 8.9 items that work at the adapter level.
//!
//! DoD items covered:
//! - 8.9.1:  Simple text generation (×3)
//! - 8.9.2:  Streaming text generation (×3)
//! - 8.9.5:  Single tool call (×3)
//! - 8.9.10: Reasoning/thinking token reporting (×3)
//! - 8.9.11: Error handling invalid API key → 401 (×3)
//! - 8.9.12: Error handling rate limit → 429 (×3)
//! - 8.9.13: Usage token counts accurate (×3)
//! - 8.9.15: Provider-specific options pass through (×3)

use futures::StreamExt;
use secrecy::SecretString;
use unified_llm::client::ClientBuilder;
use unified_llm::providers::anthropic::AnthropicAdapter;
use unified_llm::providers::gemini::GeminiAdapter;
use unified_llm::providers::openai::OpenAiAdapter;
use unified_llm_types::*;

// ============================================================================
// Test Harness: Per-provider wiremock setup
// ============================================================================

struct ProviderTestHarness {
    client: unified_llm::client::Client,
    server: wiremock::MockServer,
    provider_name: String,
}

impl ProviderTestHarness {
    async fn anthropic() -> Self {
        let server = wiremock::MockServer::start().await;
        let adapter = AnthropicAdapter::new_with_base_url(
            SecretString::from("test-key".to_string()),
            server.uri(),
        );
        let client = ClientBuilder::new()
            .provider("anthropic", Box::new(adapter))
            .build()
            .unwrap();
        Self {
            client,
            server,
            provider_name: "anthropic".to_string(),
        }
    }

    async fn openai() -> Self {
        let server = wiremock::MockServer::start().await;
        let adapter = OpenAiAdapter::new_with_base_url(
            SecretString::from("sk-test".to_string()),
            server.uri(),
        );
        let client = ClientBuilder::new()
            .provider("openai", Box::new(adapter))
            .build()
            .unwrap();
        Self {
            client,
            server,
            provider_name: "openai".to_string(),
        }
    }

    async fn gemini() -> Self {
        let server = wiremock::MockServer::start().await;
        let adapter = GeminiAdapter::new_with_base_url(
            SecretString::from("test-key".to_string()),
            server.uri(),
        );
        let client = ClientBuilder::new()
            .provider("gemini", Box::new(adapter))
            .build()
            .unwrap();
        Self {
            client,
            server,
            provider_name: "gemini".to_string(),
        }
    }
}

// ============================================================================
// Mock response helpers: provider-specific JSON shapes
// ============================================================================

fn anthropic_text_response(text: &str) -> serde_json::Value {
    serde_json::json!({
        "id": "msg_conf",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": "claude-sonnet-4-20250514",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 12, "output_tokens": 8}
    })
}

fn openai_text_response(text: &str) -> serde_json::Value {
    serde_json::json!({
        "id": "resp_conf",
        "object": "response",
        "status": "completed",
        "model": "gpt-4o",
        "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}],
        "usage": {"input_tokens": 12, "output_tokens": 8, "total_tokens": 20}
    })
}

fn gemini_text_response(text: &str) -> serde_json::Value {
    serde_json::json!({
        "responseId": "resp_conf",
        "modelVersion": "gemini-2.0-flash",
        "candidates": [{"content": {"parts": [{"text": text}], "role": "model"}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 8, "totalTokenCount": 20}
    })
}

fn anthropic_tool_call_response() -> serde_json::Value {
    serde_json::json!({
        "id": "msg_tc",
        "type": "message",
        "role": "assistant",
        "content": [{
            "type": "tool_use",
            "id": "toolu_abc",
            "name": "get_weather",
            "input": {"city": "SF"}
        }],
        "model": "claude-sonnet-4-20250514",
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 20, "output_tokens": 15}
    })
}

fn openai_tool_call_response() -> serde_json::Value {
    serde_json::json!({
        "id": "resp_tc",
        "object": "response",
        "status": "completed",
        "model": "gpt-4o",
        "output": [{
            "type": "function_call",
            "call_id": "call_abc",
            "name": "get_weather",
            "arguments": "{\"city\":\"SF\"}"
        }],
        "usage": {"input_tokens": 20, "output_tokens": 15, "total_tokens": 35}
    })
}

fn gemini_tool_call_response() -> serde_json::Value {
    serde_json::json!({
        "responseId": "resp_tc",
        "modelVersion": "gemini-2.0-flash",
        "candidates": [{"content": {"parts": [{"functionCall": {"name": "get_weather", "args": {"city": "SF"}}}], "role": "model"}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 15, "totalTokenCount": 35}
    })
}

fn anthropic_error_response(error_type: &str, message: &str) -> serde_json::Value {
    serde_json::json!({
        "type": "error",
        "error": {"type": error_type, "message": message}
    })
}

fn openai_error_response(error_type: &str, message: &str) -> serde_json::Value {
    serde_json::json!({
        "error": {"message": message, "type": error_type, "code": error_type}
    })
}

fn gemini_error_response(code: u16, status: &str, message: &str) -> serde_json::Value {
    serde_json::json!({
        "error": {"code": code, "message": message, "status": status}
    })
}

fn anthropic_usage_response(input: u32, output: u32) -> serde_json::Value {
    serde_json::json!({
        "id": "msg_usage",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "ok"}],
        "model": "claude-sonnet-4-20250514",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": input, "output_tokens": output}
    })
}

fn openai_usage_response(input: u32, output: u32) -> serde_json::Value {
    serde_json::json!({
        "id": "resp_usage",
        "object": "response",
        "status": "completed",
        "model": "gpt-4o",
        "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]}],
        "usage": {"input_tokens": input, "output_tokens": output, "total_tokens": input + output}
    })
}

fn gemini_usage_response(input: u32, output: u32) -> serde_json::Value {
    serde_json::json!({
        "responseId": "resp_usage",
        "modelVersion": "gemini-2.0-flash",
        "candidates": [{"content": {"parts": [{"text": "ok"}], "role": "model"}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": input, "candidatesTokenCount": output, "totalTokenCount": input + output}
    })
}

fn anthropic_reasoning_response() -> serde_json::Value {
    serde_json::json!({
        "id": "msg_reason",
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "Let me think about this...", "signature": "sig_abc"},
            {"type": "text", "text": "42"}
        ],
        "model": "claude-sonnet-4-20250514",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 25, "output_tokens": 50}
    })
}

fn openai_reasoning_response() -> serde_json::Value {
    serde_json::json!({
        "id": "resp_reason",
        "object": "response",
        "status": "completed",
        "model": "o3",
        "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "42"}]}],
        "usage": {
            "input_tokens": 25,
            "output_tokens": 50,
            "total_tokens": 75,
            "output_tokens_details": {"reasoning_tokens": 40}
        }
    })
}

fn gemini_reasoning_response() -> serde_json::Value {
    serde_json::json!({
        "responseId": "resp_reason",
        "modelVersion": "gemini-2.5-flash",
        "candidates": [{"content": {"parts": [
            {"thought": true, "text": "Let me think...", "thoughtSignature": "sig_gem"},
            {"text": "42"}
        ], "role": "model"}, "finishReason": "STOP"}],
        "usageMetadata": {
            "promptTokenCount": 25,
            "candidatesTokenCount": 50,
            "totalTokenCount": 75,
            "thoughtsTokenCount": 40
        }
    })
}

// ============================================================================
// SSE body helpers
// ============================================================================

fn build_sse_body(events: &[(&str, &str)]) -> String {
    events
        .iter()
        .map(|(t, d)| format!("event: {t}\ndata: {d}\n\n"))
        .collect()
}

fn anthropic_stream_text_sse(text: &str) -> String {
    // Split text in half for multiple deltas
    let mid = text.len() / 2;
    let (first, second) = text.split_at(mid);
    let first_escaped = first.replace('\"', "\\\"");
    let second_escaped = second.replace('\"', "\\\"");
    let full_escaped = text.replace('\"', "\\\"");
    build_sse_body(&[
        (
            "message_start",
            &format!(
                r#"{{"type":"message_start","message":{{"id":"msg_stream","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-20250514","usage":{{"input_tokens":10,"output_tokens":0}}}}}}"#
            ),
        ),
        (
            "content_block_start",
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
        ),
        (
            "content_block_delta",
            &format!(
                r#"{{"type":"content_block_delta","index":0,"delta":{{"type":"text_delta","text":"{first_escaped}"}}}}"#
            ),
        ),
        (
            "content_block_delta",
            &format!(
                r#"{{"type":"content_block_delta","index":0,"delta":{{"type":"text_delta","text":"{second_escaped}"}}}}"#
            ),
        ),
        (
            "content_block_stop",
            r#"{"type":"content_block_stop","index":0}"#,
        ),
        (
            "message_delta",
            &format!(
                r#"{{"type":"message_delta","delta":{{"stop_reason":"end_turn"}},"usage":{{"output_tokens":{}}}}}"#,
                full_escaped.len()
            ),
        ),
        ("message_stop", r#"{"type":"message_stop"}"#),
    ])
}

fn openai_stream_text_sse(text: &str) -> String {
    let mid = text.len() / 2;
    let (first, second) = text.split_at(mid);
    let first_escaped = first.replace('\"', "\\\"");
    let second_escaped = second.replace('\"', "\\\"");
    let full_escaped = text.replace('\"', "\\\"");
    build_sse_body(&[
        (
            "response.created",
            r#"{"response":{"id":"resp_stream","object":"response","status":"in_progress","model":"gpt-4o","output":[]},"sequence_number":0}"#,
        ),
        (
            "response.output_item.added",
            r#"{"item":{"type":"message","role":"assistant","content":[]},"output_index":0}"#,
        ),
        (
            "response.content_part.added",
            r#"{"part":{"type":"output_text","text":""},"output_index":0,"content_index":0}"#,
        ),
        (
            "response.output_text.delta",
            &format!(r#"{{"delta":"{first_escaped}","output_index":0,"content_index":0}}"#),
        ),
        (
            "response.output_text.delta",
            &format!(r#"{{"delta":"{second_escaped}","output_index":0,"content_index":0}}"#),
        ),
        (
            "response.output_text.done",
            &format!(r#"{{"text":"{full_escaped}","output_index":0,"content_index":0}}"#),
        ),
        (
            "response.output_item.done",
            &format!(
                r#"{{"item":{{"type":"message","role":"assistant","content":[{{"type":"output_text","text":"{full_escaped}"}}]}},"output_index":0}}"#
            ),
        ),
        (
            "response.completed",
            &format!(
                r#"{{"response":{{"id":"resp_stream","object":"response","status":"completed","model":"gpt-4o","output":[{{"type":"message","role":"assistant","content":[{{"type":"output_text","text":"{full_escaped}"}}]}}],"usage":{{"input_tokens":10,"output_tokens":5,"total_tokens":15}}}},"sequence_number":5}}"#
            ),
        ),
    ])
}

fn gemini_stream_text_sse(text: &str) -> String {
    let mid = text.len() / 2;
    let (first, second) = text.split_at(mid);
    let first_escaped = first.replace('\"', "\\\"");
    let second_escaped = second.replace('\"', "\\\"");
    [
        format!("data: {{\"responseId\":\"resp_stream\",\"modelVersion\":\"gemini-2.0-flash\",\"candidates\":[{{\"content\":{{\"parts\":[{{\"text\":\"{first_escaped}\"}}],\"role\":\"model\"}}}}]}}\n\n"),
        format!("data: {{\"candidates\":[{{\"content\":{{\"parts\":[{{\"text\":\"{second_escaped}\"}}],\"role\":\"model\"}},\"finishReason\":\"STOP\"}}],\"usageMetadata\":{{\"promptTokenCount\":10,\"candidatesTokenCount\":5,\"totalTokenCount\":15}}}}\n\n"),
    ].join("")
}

// ============================================================================
// Mount helpers
// ============================================================================

async fn mount_text_response(h: &ProviderTestHarness, text: &str) {
    match h.provider_name.as_str() {
        "anthropic" => {
            wiremock::Mock::given(wiremock::matchers::method("POST"))
                .and(wiremock::matchers::path("/v1/messages"))
                .respond_with(
                    wiremock::ResponseTemplate::new(200)
                        .set_body_json(anthropic_text_response(text)),
                )
                .mount(&h.server)
                .await;
        }
        "openai" => {
            wiremock::Mock::given(wiremock::matchers::method("POST"))
                .and(wiremock::matchers::path("/v1/responses"))
                .respond_with(
                    wiremock::ResponseTemplate::new(200).set_body_json(openai_text_response(text)),
                )
                .mount(&h.server)
                .await;
        }
        "gemini" => {
            wiremock::Mock::given(wiremock::matchers::method("POST"))
                .and(wiremock::matchers::path_regex(
                    r"/v1beta/models/.+:generateContent",
                ))
                .respond_with(
                    wiremock::ResponseTemplate::new(200).set_body_json(gemini_text_response(text)),
                )
                .mount(&h.server)
                .await;
        }
        _ => panic!("Unknown provider: {}", h.provider_name),
    }
}

async fn mount_stream_response(h: &ProviderTestHarness, text: &str) {
    let (sse_body, path_pattern) = match h.provider_name.as_str() {
        "anthropic" => (anthropic_stream_text_sse(text), "/v1/messages".to_string()),
        "openai" => (openai_stream_text_sse(text), "/v1/responses".to_string()),
        "gemini" => (
            gemini_stream_text_sse(text),
            "REGEX".to_string(), // handled below
        ),
        _ => panic!("Unknown provider"),
    };

    if h.provider_name == "gemini" {
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path_regex(
                r"/v1beta/models/.+:streamGenerateContent",
            ))
            .respond_with(
                wiremock::ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse_body),
            )
            .mount(&h.server)
            .await;
    } else {
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path(path_pattern))
            .respond_with(
                wiremock::ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse_body),
            )
            .mount(&h.server)
            .await;
    }
}

async fn mount_error_response(h: &ProviderTestHarness, status: u16, retry_after: Option<&str>) {
    let body = match h.provider_name.as_str() {
        "anthropic" => {
            let err_type = if status == 401 {
                "authentication_error"
            } else {
                "rate_limit_error"
            };
            anthropic_error_response(err_type, "Test error")
        }
        "openai" => {
            let err_type = if status == 401 {
                "invalid_api_key"
            } else {
                "rate_limit_error"
            };
            openai_error_response(err_type, "Test error")
        }
        "gemini" => {
            let gstatus = if status == 401 {
                "UNAUTHENTICATED"
            } else {
                "RESOURCE_EXHAUSTED"
            };
            gemini_error_response(status, gstatus, "Test error")
        }
        _ => panic!("Unknown provider"),
    };

    let mut template = wiremock::ResponseTemplate::new(status).set_body_json(body);
    if let Some(ra) = retry_after {
        template = template.insert_header("retry-after", ra);
    }

    match h.provider_name.as_str() {
        "anthropic" => {
            wiremock::Mock::given(wiremock::matchers::method("POST"))
                .and(wiremock::matchers::path("/v1/messages"))
                .respond_with(template)
                .mount(&h.server)
                .await;
        }
        "openai" => {
            wiremock::Mock::given(wiremock::matchers::method("POST"))
                .and(wiremock::matchers::path("/v1/responses"))
                .respond_with(template)
                .mount(&h.server)
                .await;
        }
        "gemini" => {
            wiremock::Mock::given(wiremock::matchers::method("POST"))
                .respond_with(template)
                .mount(&h.server)
                .await;
        }
        _ => panic!("Unknown provider: {}", h.provider_name),
    }
}

async fn mount_tool_call_response(h: &ProviderTestHarness) {
    let body = match h.provider_name.as_str() {
        "anthropic" => anthropic_tool_call_response(),
        "openai" => openai_tool_call_response(),
        "gemini" => gemini_tool_call_response(),
        _ => panic!("Unknown provider"),
    };

    match h.provider_name.as_str() {
        "anthropic" => {
            wiremock::Mock::given(wiremock::matchers::method("POST"))
                .and(wiremock::matchers::path("/v1/messages"))
                .respond_with(wiremock::ResponseTemplate::new(200).set_body_json(body))
                .mount(&h.server)
                .await;
        }
        "openai" => {
            wiremock::Mock::given(wiremock::matchers::method("POST"))
                .and(wiremock::matchers::path("/v1/responses"))
                .respond_with(wiremock::ResponseTemplate::new(200).set_body_json(body))
                .mount(&h.server)
                .await;
        }
        "gemini" => {
            wiremock::Mock::given(wiremock::matchers::method("POST"))
                .respond_with(wiremock::ResponseTemplate::new(200).set_body_json(body))
                .mount(&h.server)
                .await;
        }
        _ => panic!("Unknown provider: {}", h.provider_name),
    }
}

async fn mount_usage_response(h: &ProviderTestHarness, input: u32, output: u32) {
    let body = match h.provider_name.as_str() {
        "anthropic" => anthropic_usage_response(input, output),
        "openai" => openai_usage_response(input, output),
        "gemini" => gemini_usage_response(input, output),
        _ => panic!("Unknown provider"),
    };

    match h.provider_name.as_str() {
        "anthropic" => {
            wiremock::Mock::given(wiremock::matchers::method("POST"))
                .and(wiremock::matchers::path("/v1/messages"))
                .respond_with(wiremock::ResponseTemplate::new(200).set_body_json(body))
                .mount(&h.server)
                .await;
        }
        "openai" => {
            wiremock::Mock::given(wiremock::matchers::method("POST"))
                .and(wiremock::matchers::path("/v1/responses"))
                .respond_with(wiremock::ResponseTemplate::new(200).set_body_json(body))
                .mount(&h.server)
                .await;
        }
        "gemini" => {
            wiremock::Mock::given(wiremock::matchers::method("POST"))
                .respond_with(wiremock::ResponseTemplate::new(200).set_body_json(body))
                .mount(&h.server)
                .await;
        }
        _ => panic!("Unknown provider: {}", h.provider_name),
    }
}

async fn mount_reasoning_response(h: &ProviderTestHarness) {
    let body = match h.provider_name.as_str() {
        "anthropic" => anthropic_reasoning_response(),
        "openai" => openai_reasoning_response(),
        "gemini" => gemini_reasoning_response(),
        _ => panic!("Unknown provider"),
    };

    match h.provider_name.as_str() {
        "anthropic" => {
            wiremock::Mock::given(wiremock::matchers::method("POST"))
                .and(wiremock::matchers::path("/v1/messages"))
                .respond_with(wiremock::ResponseTemplate::new(200).set_body_json(body))
                .mount(&h.server)
                .await;
        }
        "openai" => {
            wiremock::Mock::given(wiremock::matchers::method("POST"))
                .and(wiremock::matchers::path("/v1/responses"))
                .respond_with(wiremock::ResponseTemplate::new(200).set_body_json(body))
                .mount(&h.server)
                .await;
        }
        "gemini" => {
            wiremock::Mock::given(wiremock::matchers::method("POST"))
                .respond_with(wiremock::ResponseTemplate::new(200).set_body_json(body))
                .mount(&h.server)
                .await;
        }
        _ => panic!("Unknown provider: {}", h.provider_name),
    }
}

async fn mount_provider_options_response(h: &ProviderTestHarness) {
    // Use the text response; we verify the request body for provider_options passthrough
    mount_text_response(h, "options ok").await;
}

// ============================================================================
// DoD 8.9.1: Simple text generation (×3)
// ============================================================================

async fn verify_simple_text(h: &ProviderTestHarness) {
    mount_text_response(h, "Hello from the model!").await;
    let req = Request::default()
        .model("test-model")
        .messages(vec![Message::user("Say hello")]);
    let resp = h.client.complete(req).await.unwrap();

    assert!(
        !resp.text().is_empty(),
        "{}: text should not be empty",
        h.provider_name
    );
    assert_eq!(
        resp.text(),
        "Hello from the model!",
        "{}: text mismatch",
        h.provider_name
    );
    assert_eq!(
        resp.finish_reason.reason, "stop",
        "{}: finish_reason should be stop",
        h.provider_name
    );
    assert!(
        resp.usage.input_tokens > 0,
        "{}: input_tokens should be > 0",
        h.provider_name
    );
    assert!(
        resp.usage.output_tokens > 0,
        "{}: output_tokens should be > 0",
        h.provider_name
    );
    assert_eq!(
        resp.provider, h.provider_name,
        "{}: provider name mismatch",
        h.provider_name
    );
}

#[tokio::test]
async fn test_simple_text_anthropic() {
    verify_simple_text(&ProviderTestHarness::anthropic().await).await;
}

#[tokio::test]
async fn test_simple_text_openai() {
    verify_simple_text(&ProviderTestHarness::openai().await).await;
}

#[tokio::test]
async fn test_simple_text_gemini() {
    verify_simple_text(&ProviderTestHarness::gemini().await).await;
}

// ============================================================================
// DoD 8.9.2: Streaming text generation (×3)
// ============================================================================

async fn verify_streaming_text(h: &ProviderTestHarness) {
    mount_stream_response(h, "Hello world").await;
    let req = Request::default()
        .model("test-model")
        .messages(vec![Message::user("Say hello")]);
    let stream = h.client.stream(req).unwrap();
    let events: Vec<StreamEvent> = stream
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .filter_map(|r| r.ok())
        .collect();

    let types: Vec<_> = events.iter().map(|e| e.event_type.clone()).collect();

    // All providers must emit these event types
    assert!(
        types.contains(&StreamEventType::StreamStart),
        "{}: missing StreamStart",
        h.provider_name
    );
    assert!(
        types.contains(&StreamEventType::TextStart),
        "{}: missing TextStart",
        h.provider_name
    );
    assert!(
        types.contains(&StreamEventType::TextDelta),
        "{}: missing TextDelta",
        h.provider_name
    );
    assert!(
        types.contains(&StreamEventType::TextEnd),
        "{}: missing TextEnd",
        h.provider_name
    );
    assert!(
        types.contains(&StreamEventType::Finish),
        "{}: missing Finish",
        h.provider_name
    );

    // Text deltas concatenate to the full response
    let text: String = events
        .iter()
        .filter(|e| e.event_type == StreamEventType::TextDelta)
        .filter_map(|e| e.delta.as_ref())
        .cloned()
        .collect();
    assert_eq!(
        text, "Hello world",
        "{}: text deltas mismatch",
        h.provider_name
    );

    // Finish event has usage
    let finish = events
        .iter()
        .find(|e| e.event_type == StreamEventType::Finish)
        .expect("missing Finish event");
    assert!(
        finish.usage.is_some(),
        "{}: Finish should have usage",
        h.provider_name
    );
}

#[tokio::test]
async fn test_streaming_text_anthropic() {
    verify_streaming_text(&ProviderTestHarness::anthropic().await).await;
}

#[tokio::test]
async fn test_streaming_text_openai() {
    verify_streaming_text(&ProviderTestHarness::openai().await).await;
}

#[tokio::test]
async fn test_streaming_text_gemini() {
    verify_streaming_text(&ProviderTestHarness::gemini().await).await;
}

// ============================================================================
// DoD 8.9.5: Single tool call (×3)
// ============================================================================

async fn verify_single_tool_call(h: &ProviderTestHarness) {
    mount_tool_call_response(h).await;
    let req = Request::default()
        .model("test-model")
        .messages(vec![Message::user("What's the weather?")])
        .tools(vec![ToolDefinition {
            name: "get_weather".to_string(),
            description: "Get weather for a city".to_string(),
            parameters: serde_json::json!({"type": "object", "properties": {"city": {"type": "string"}}}),
            strict: None,
        }]);
    let resp = h.client.complete(req).await.unwrap();

    // Finish reason should be tool_calls
    assert_eq!(
        resp.finish_reason.reason, "tool_calls",
        "{}: finish_reason should be tool_calls",
        h.provider_name
    );

    // Should have exactly one tool call
    let tool_calls = resp.tool_calls();
    assert_eq!(
        tool_calls.len(),
        1,
        "{}: should have exactly 1 tool call",
        h.provider_name
    );

    // Tool call has correct function name
    assert_eq!(
        tool_calls[0].name, "get_weather",
        "{}: tool call name mismatch",
        h.provider_name
    );

    // Tool call has a non-empty ID
    assert!(
        !tool_calls[0].id.is_empty(),
        "{}: tool call ID should not be empty",
        h.provider_name
    );

    // Arguments contain the city (ToolCall.arguments is already a parsed Map)
    assert_eq!(
        tool_calls[0].arguments.get("city").unwrap(),
        &serde_json::json!("SF"),
        "{}: tool call argument mismatch",
        h.provider_name
    );
}

#[tokio::test]
async fn test_tool_call_anthropic() {
    verify_single_tool_call(&ProviderTestHarness::anthropic().await).await;
}

#[tokio::test]
async fn test_tool_call_openai() {
    verify_single_tool_call(&ProviderTestHarness::openai().await).await;
}

#[tokio::test]
async fn test_tool_call_gemini() {
    verify_single_tool_call(&ProviderTestHarness::gemini().await).await;
}

// ============================================================================
// DoD 8.9.11: Error handling — invalid API key → 401 (×3)
// ============================================================================

async fn verify_error_401(h: &ProviderTestHarness) {
    mount_error_response(h, 401, None).await;
    let req = Request::default()
        .model("test-model")
        .messages(vec![Message::user("Hi")]);
    let err = h.client.complete(req).await.unwrap_err();

    assert_eq!(
        err.kind,
        ErrorKind::Authentication,
        "{}: 401 should map to Authentication",
        h.provider_name
    );
    assert!(
        !err.retryable,
        "{}: auth error should not be retryable",
        h.provider_name
    );
    assert_eq!(
        err.provider,
        Some(h.provider_name.clone()),
        "{}: provider should be set",
        h.provider_name
    );
}

#[tokio::test]
async fn test_error_401_anthropic() {
    verify_error_401(&ProviderTestHarness::anthropic().await).await;
}

#[tokio::test]
async fn test_error_401_openai() {
    verify_error_401(&ProviderTestHarness::openai().await).await;
}

#[tokio::test]
async fn test_error_401_gemini() {
    verify_error_401(&ProviderTestHarness::gemini().await).await;
}

// ============================================================================
// DoD 8.9.12: Error handling — rate limit → 429 (×3)
// ============================================================================

async fn verify_error_429(h: &ProviderTestHarness) {
    mount_error_response(h, 429, Some("15")).await;
    let req = Request::default()
        .model("test-model")
        .messages(vec![Message::user("Hi")]);
    let err = h.client.complete(req).await.unwrap_err();

    assert_eq!(
        err.kind,
        ErrorKind::RateLimit,
        "{}: 429 should map to RateLimit",
        h.provider_name
    );
    assert!(
        err.retryable,
        "{}: rate limit should be retryable",
        h.provider_name
    );
    assert_eq!(
        err.retry_after,
        Some(std::time::Duration::from_secs(15)),
        "{}: retry_after should be parsed",
        h.provider_name
    );
}

#[tokio::test]
async fn test_error_429_anthropic() {
    verify_error_429(&ProviderTestHarness::anthropic().await).await;
}

#[tokio::test]
async fn test_error_429_openai() {
    verify_error_429(&ProviderTestHarness::openai().await).await;
}

#[tokio::test]
async fn test_error_429_gemini() {
    verify_error_429(&ProviderTestHarness::gemini().await).await;
}

// ============================================================================
// DoD 8.9.12: 429 retry-then-succeed — transparent retry via generate() (×3)
// ============================================================================

/// Verify that a 429 response is retried transparently by api::generate()
/// and succeeds when the retry gets a 200 response.
///
/// This complements verify_error_429 (which only tests error mapping at the
/// client.complete() level) by exercising the full retry path through the
/// high-level generate() API.
async fn verify_error_429_retry_then_succeed(h: &ProviderTestHarness) {
    // Build provider-specific error and success response bodies
    let error_body = match h.provider_name.as_str() {
        "anthropic" => anthropic_error_response("rate_limit_error", "Rate limited"),
        "openai" => openai_error_response("rate_limit_error", "Rate limited"),
        "gemini" => gemini_error_response(429, "RESOURCE_EXHAUSTED", "Rate limited"),
        _ => panic!("Unknown provider"),
    };
    let success_body = match h.provider_name.as_str() {
        "anthropic" => anthropic_text_response("Success after retry"),
        "openai" => openai_text_response("Success after retry"),
        "gemini" => gemini_text_response("Success after retry"),
        _ => panic!("Unknown provider"),
    };

    // Mount sequenced responses using wiremock priorities:
    //   Priority 1 (highest): 429 error with Retry-After: 0 — consumed first
    //   Priority 2 (lower):   200 success — consumed on retry
    let error_template = wiremock::ResponseTemplate::new(429)
        .set_body_json(error_body)
        .insert_header("retry-after", "0");
    let success_template = wiremock::ResponseTemplate::new(200).set_body_json(success_body);

    if h.provider_name == "gemini" {
        let path_re = r"/v1beta/models/.+:generateContent";
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path_regex(path_re))
            .respond_with(error_template)
            .up_to_n_times(1)
            .with_priority(1)
            .mount(&h.server)
            .await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path_regex(path_re))
            .respond_with(success_template)
            .up_to_n_times(1)
            .with_priority(2)
            .mount(&h.server)
            .await;
    } else {
        let path = match h.provider_name.as_str() {
            "anthropic" => "/v1/messages",
            "openai" => "/v1/responses",
            _ => panic!("Unknown provider"),
        };
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path(path))
            .respond_with(error_template)
            .up_to_n_times(1)
            .with_priority(1)
            .mount(&h.server)
            .await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path(path))
            .respond_with(success_template)
            .up_to_n_times(1)
            .with_priority(2)
            .mount(&h.server)
            .await;
    }

    // Use api::generate() which has built-in retry logic (via with_retry).
    // max_retries(1) = one retry attempt after the initial 429 failure.
    let opts = GenerateOptions::new("test-model")
        .messages(vec![Message::user("Hello")])
        .max_retries(1);
    let result = api::generate(opts, &h.client).await.unwrap();

    assert_eq!(
        result.text, "Success after retry",
        "{}: 429 should be retried transparently and succeed",
        h.provider_name
    );
}

#[tokio::test]
async fn test_error_429_retry_then_succeed_anthropic() {
    verify_error_429_retry_then_succeed(&ProviderTestHarness::anthropic().await).await;
}

#[tokio::test]
async fn test_error_429_retry_then_succeed_openai() {
    verify_error_429_retry_then_succeed(&ProviderTestHarness::openai().await).await;
}

#[tokio::test]
async fn test_error_429_retry_then_succeed_gemini() {
    verify_error_429_retry_then_succeed(&ProviderTestHarness::gemini().await).await;
}

// ============================================================================
// DoD 8.9.13: Usage token counts accurate (×3)
// ============================================================================

async fn verify_usage_accuracy(h: &ProviderTestHarness) {
    mount_usage_response(h, 42, 17).await;
    let req = Request::default()
        .model("test-model")
        .messages(vec![Message::user("Count tokens")]);
    let resp = h.client.complete(req).await.unwrap();

    assert_eq!(
        resp.usage.input_tokens, 42,
        "{}: input_tokens mismatch",
        h.provider_name
    );
    assert_eq!(
        resp.usage.output_tokens, 17,
        "{}: output_tokens mismatch",
        h.provider_name
    );
    assert_eq!(
        resp.usage.total_tokens, 59,
        "{}: total_tokens should be input + output",
        h.provider_name
    );
}

#[tokio::test]
async fn test_usage_accuracy_anthropic() {
    verify_usage_accuracy(&ProviderTestHarness::anthropic().await).await;
}

#[tokio::test]
async fn test_usage_accuracy_openai() {
    verify_usage_accuracy(&ProviderTestHarness::openai().await).await;
}

#[tokio::test]
async fn test_usage_accuracy_gemini() {
    verify_usage_accuracy(&ProviderTestHarness::gemini().await).await;
}

// ============================================================================
// DoD 8.9.15: Provider-specific options pass through (×3)
// ============================================================================

async fn verify_provider_options(h: &ProviderTestHarness) {
    mount_provider_options_response(h).await;

    let provider_opts = match h.provider_name.as_str() {
        "anthropic" => {
            serde_json::json!({"anthropic": {"betas": ["beta-flag"], "custom_key": "custom_val"}})
        }
        "openai" => serde_json::json!({"openai": {"store": true, "metadata": {"session": "abc"}}}),
        "gemini" => serde_json::json!({"gemini": {"extra_config": "val123"}}),
        _ => panic!("Unknown provider"),
    };

    let req = Request::default()
        .model("test-model")
        .messages(vec![Message::user("Options test")])
        .provider_options(Some(provider_opts));

    let resp = h.client.complete(req).await.unwrap();
    assert_eq!(
        resp.text(),
        "options ok",
        "{}: response text mismatch",
        h.provider_name
    );

    // Verify the request was received (which means options were included in the body)
    let requests = h.server.received_requests().await.unwrap();
    assert_eq!(
        requests.len(),
        1,
        "{}: should have exactly 1 request",
        h.provider_name
    );

    let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();

    // Provider-specific assertions
    match h.provider_name.as_str() {
        "openai" => {
            assert_eq!(
                body["store"], true,
                "openai: store option should be passed through"
            );
            assert_eq!(
                body["metadata"]["session"], "abc",
                "openai: metadata should be passed through"
            );
        }
        "gemini" => {
            assert_eq!(
                body["extra_config"], "val123",
                "gemini: extra_config should be passed through"
            );
        }
        "anthropic" => {
            // W-11: Anthropic filters internal keys (betas, beta_headers, auto_cache)
            // but passes non-internal keys through to the request body.
            assert_eq!(
                body["custom_key"], "custom_val",
                "anthropic: custom_key should be passed through to request body"
            );
            // Internal key "betas" should NOT appear at top level
            assert!(
                body.get("betas").is_none(),
                "anthropic: internal key 'betas' should be filtered out"
            );
        }
        _ => {}
    }
}

#[tokio::test]
async fn test_provider_options_anthropic() {
    verify_provider_options(&ProviderTestHarness::anthropic().await).await;
}

#[tokio::test]
async fn test_provider_options_openai() {
    verify_provider_options(&ProviderTestHarness::openai().await).await;
}

#[tokio::test]
async fn test_provider_options_gemini() {
    verify_provider_options(&ProviderTestHarness::gemini().await).await;
}

// ============================================================================
// DoD 8.9.10: Reasoning/thinking token reporting (×3)
// Also covers 8.5.6: reasoning_tokens distinct from output_tokens
// ============================================================================

async fn verify_reasoning_tokens(h: &ProviderTestHarness) {
    mount_reasoning_response(h).await;
    let req = Request::default()
        .model("test-model")
        .messages(vec![Message::user("Think hard about this")]);
    let resp = h.client.complete(req).await.unwrap();

    // All providers should return the text portion
    assert_eq!(
        resp.text(),
        "42",
        "{}: text should be '42'",
        h.provider_name
    );

    match h.provider_name.as_str() {
        "anthropic" => {
            // Anthropic reports thinking as content blocks, not usage.reasoning_tokens
            // Verify thinking content is present
            let reasoning = resp.reasoning();
            assert!(
                reasoning.is_some(),
                "anthropic: should have reasoning content"
            );
            assert!(
                reasoning.unwrap().contains("think"),
                "anthropic: reasoning should contain thinking text"
            );
        }
        "openai" => {
            // OpenAI reports reasoning_tokens in usage
            assert_eq!(
                resp.usage.reasoning_tokens,
                Some(40),
                "openai: reasoning_tokens should be 40"
            );
            // 8.5.6: reasoning_tokens distinct from output_tokens
            assert_eq!(
                resp.usage.output_tokens, 50,
                "openai: output_tokens should be 50 (includes reasoning)"
            );
            assert_ne!(
                resp.usage.reasoning_tokens.unwrap_or(0),
                resp.usage.output_tokens,
                "openai: reasoning_tokens should differ from output_tokens"
            );
        }
        "gemini" => {
            // Gemini reports thinking via thoughtsTokenCount and thinking content parts
            assert_eq!(
                resp.usage.reasoning_tokens,
                Some(40),
                "gemini: reasoning_tokens should be 40"
            );
            // Gemini also has thinking content parts
            let reasoning = resp.reasoning();
            assert!(reasoning.is_some(), "gemini: should have reasoning content");
            // 8.5.6: reasoning_tokens distinct from output_tokens
            assert_ne!(
                resp.usage.reasoning_tokens.unwrap_or(0),
                resp.usage.output_tokens,
                "gemini: reasoning_tokens should differ from output_tokens"
            );
        }
        _ => panic!("Unknown provider"),
    }
}

#[tokio::test]
async fn test_reasoning_tokens_anthropic() {
    verify_reasoning_tokens(&ProviderTestHarness::anthropic().await).await;
}

#[tokio::test]
async fn test_reasoning_tokens_openai() {
    verify_reasoning_tokens(&ProviderTestHarness::openai().await).await;
}

#[tokio::test]
async fn test_reasoning_tokens_gemini() {
    verify_reasoning_tokens(&ProviderTestHarness::gemini().await).await;
}

// ============================================================================
// Phase 3: Additional response helpers for tool conformance
// ============================================================================

fn anthropic_parallel_tool_call_response() -> serde_json::Value {
    serde_json::json!({
        "id": "msg_ptc", "type": "message", "role": "assistant",
        "content": [
            {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "SF"}},
            {"type": "tool_use", "id": "toolu_2", "name": "get_weather", "input": {"city": "NYC"}}
        ],
        "model": "claude-sonnet-4-20250514", "stop_reason": "tool_use",
        "usage": {"input_tokens": 30, "output_tokens": 20}
    })
}

fn openai_parallel_tool_call_response() -> serde_json::Value {
    serde_json::json!({
        "id": "resp_ptc", "object": "response", "status": "completed", "model": "gpt-4o",
        "output": [
            {"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": "{\"city\":\"SF\"}"},
            {"type": "function_call", "call_id": "call_2", "name": "get_weather", "arguments": "{\"city\":\"NYC\"}"}
        ],
        "usage": {"input_tokens": 30, "output_tokens": 20, "total_tokens": 50}
    })
}

fn gemini_parallel_tool_call_response() -> serde_json::Value {
    serde_json::json!({
        "responseId": "resp_ptc", "modelVersion": "gemini-2.0-flash",
        "candidates": [{"content": {"parts": [
            {"functionCall": {"name": "get_weather", "args": {"city": "SF"}}},
            {"functionCall": {"name": "get_weather", "args": {"city": "NYC"}}}
        ], "role": "model"}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 30, "candidatesTokenCount": 20, "totalTokenCount": 50}
    })
}

/// Mount a sequence of responses: each call consumes the next response in order.
/// Uses wiremock priorities — lower priority number = higher priority, with up_to_n_times(1)
/// so after a mock is consumed, the next one takes over.
async fn mount_sequenced_responses(
    h: &ProviderTestHarness,
    responses: Vec<(serde_json::Value, u16)>,
) {
    for (i, (body, status)) in responses.into_iter().enumerate() {
        let template = wiremock::ResponseTemplate::new(status).set_body_json(body);
        let priority = (i + 1) as u8;

        if h.provider_name == "gemini" {
            wiremock::Mock::given(wiremock::matchers::method("POST"))
                .and(wiremock::matchers::path_regex(
                    r"/v1beta/models/.+:generateContent",
                ))
                .respond_with(template)
                .up_to_n_times(1)
                .with_priority(priority)
                .mount(&h.server)
                .await;
        } else {
            let path = match h.provider_name.as_str() {
                "anthropic" => "/v1/messages",
                "openai" => "/v1/responses",
                _ => panic!("Unknown provider"),
            };
            wiremock::Mock::given(wiremock::matchers::method("POST"))
                .and(wiremock::matchers::path(path))
                .respond_with(template)
                .up_to_n_times(1)
                .with_priority(priority)
                .mount(&h.server)
                .await;
        }
    }
}

// ============================================================================
// DoD 8.9.5: Single tool call + execution via generate() (×3)
// ============================================================================

use unified_llm::api::{self, GenerateOptions, Tool};

async fn verify_tool_single_call(h: &ProviderTestHarness) {
    let tool_call_body = match h.provider_name.as_str() {
        "anthropic" => anthropic_tool_call_response(),
        "openai" => openai_tool_call_response(),
        "gemini" => gemini_tool_call_response(),
        _ => panic!("Unknown provider"),
    };
    let final_body = match h.provider_name.as_str() {
        "anthropic" => anthropic_text_response("Weather: sunny in SF"),
        "openai" => openai_text_response("Weather: sunny in SF"),
        "gemini" => gemini_text_response("Weather: sunny in SF"),
        _ => panic!("Unknown provider"),
    };
    mount_sequenced_responses(h, vec![(tool_call_body, 200), (final_body, 200)]).await;

    let tool = Tool::active(
        "get_weather",
        "Get weather for a city",
        serde_json::json!({"type": "object", "properties": {"city": {"type": "string"}}}),
        |args| {
            Box::pin(async move {
                let city = args
                    .get("city")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                Ok(serde_json::json!({"weather": format!("sunny in {city}")}))
            })
        },
    );

    let opts = GenerateOptions::new("test-model")
        .prompt("What's the weather in SF?")
        .tools(vec![tool])
        .max_tool_rounds(1);
    let result = api::generate(opts, &h.client).await.unwrap();

    assert_eq!(
        result.text, "Weather: sunny in SF",
        "{}: final text mismatch",
        h.provider_name
    );
    assert_eq!(
        result.steps.len(),
        2,
        "{}: should have 2 steps (tool call + final)",
        h.provider_name
    );
    assert!(
        !result.steps[0].tool_calls.is_empty(),
        "{}: step 0 should have tool calls",
        h.provider_name
    );
    assert_eq!(
        result.steps[0].tool_calls[0].name, "get_weather",
        "{}: tool name mismatch",
        h.provider_name
    );
    assert!(
        !result.steps[0].tool_results.is_empty(),
        "{}: step 0 should have tool results",
        h.provider_name
    );
    assert_eq!(
        result.finish_reason.reason, "stop",
        "{}: finish_reason should be stop",
        h.provider_name
    );
}

#[tokio::test]
async fn test_tool_single_call_anthropic() {
    verify_tool_single_call(&ProviderTestHarness::anthropic().await).await;
}
#[tokio::test]
async fn test_tool_single_call_openai() {
    verify_tool_single_call(&ProviderTestHarness::openai().await).await;
}
#[tokio::test]
async fn test_tool_single_call_gemini() {
    verify_tool_single_call(&ProviderTestHarness::gemini().await).await;
}

// ============================================================================
// DoD 8.9.6: Multiple parallel tool calls via generate() (×3)
// ============================================================================

async fn verify_tool_parallel_calls(h: &ProviderTestHarness) {
    let parallel_body = match h.provider_name.as_str() {
        "anthropic" => anthropic_parallel_tool_call_response(),
        "openai" => openai_parallel_tool_call_response(),
        "gemini" => gemini_parallel_tool_call_response(),
        _ => panic!("Unknown provider"),
    };
    let final_body = match h.provider_name.as_str() {
        "anthropic" => anthropic_text_response("Weather: SF sunny, NYC rainy"),
        "openai" => openai_text_response("Weather: SF sunny, NYC rainy"),
        "gemini" => gemini_text_response("Weather: SF sunny, NYC rainy"),
        _ => panic!("Unknown provider"),
    };
    mount_sequenced_responses(h, vec![(parallel_body, 200), (final_body, 200)]).await;

    let tool = Tool::active(
        "get_weather",
        "Get weather",
        serde_json::json!({"type": "object", "properties": {"city": {"type": "string"}}}),
        |args| {
            Box::pin(async move {
                let city = args.get("city").and_then(|v| v.as_str()).unwrap_or("?");
                let weather = if city == "SF" { "sunny" } else { "rainy" };
                Ok(serde_json::json!({"weather": weather}))
            })
        },
    );

    let opts = GenerateOptions::new("test-model")
        .prompt("Weather in SF and NYC?")
        .tools(vec![tool])
        .max_tool_rounds(1);
    let result = api::generate(opts, &h.client).await.unwrap();

    assert_eq!(
        result.text, "Weather: SF sunny, NYC rainy",
        "{}: final text mismatch",
        h.provider_name
    );
    assert_eq!(
        result.steps[0].tool_calls.len(),
        2,
        "{}: step 0 should have 2 parallel tool calls",
        h.provider_name
    );
    assert_eq!(
        result.steps[0].tool_results.len(),
        2,
        "{}: step 0 should have 2 tool results",
        h.provider_name
    );
}

#[tokio::test]
async fn test_tool_parallel_calls_anthropic() {
    verify_tool_parallel_calls(&ProviderTestHarness::anthropic().await).await;
}
#[tokio::test]
async fn test_tool_parallel_calls_openai() {
    verify_tool_parallel_calls(&ProviderTestHarness::openai().await).await;
}
#[tokio::test]
async fn test_tool_parallel_calls_gemini() {
    verify_tool_parallel_calls(&ProviderTestHarness::gemini().await).await;
}

// ============================================================================
// DoD 8.9.7: Multi-step tool loop, 3+ rounds via generate() (×3)
// ============================================================================

async fn verify_tool_multi_step(h: &ProviderTestHarness) {
    // 3 tool-call rounds + 1 final text = 4 responses
    let tc = |_label: &str| match h.provider_name.as_str() {
        "anthropic" => anthropic_tool_call_response(),
        "openai" => openai_tool_call_response(),
        "gemini" => gemini_tool_call_response(),
        _ => panic!("Unknown provider"),
    };
    let final_body = match h.provider_name.as_str() {
        "anthropic" => anthropic_text_response("All done after 3 rounds"),
        "openai" => openai_text_response("All done after 3 rounds"),
        "gemini" => gemini_text_response("All done after 3 rounds"),
        _ => panic!("Unknown provider"),
    };
    mount_sequenced_responses(
        h,
        vec![
            (tc("r1"), 200),
            (tc("r2"), 200),
            (tc("r3"), 200),
            (final_body, 200),
        ],
    )
    .await;

    let tool = Tool::active(
        "get_weather",
        "Get weather",
        serde_json::json!({"type": "object"}),
        |_| Box::pin(async move { Ok(serde_json::json!({"result": "ok"})) }),
    );

    let opts = GenerateOptions::new("test-model")
        .prompt("Do 3 rounds")
        .tools(vec![tool])
        .max_tool_rounds(5);
    let result = api::generate(opts, &h.client).await.unwrap();

    assert_eq!(
        result.text, "All done after 3 rounds",
        "{}: final text mismatch",
        h.provider_name
    );
    assert_eq!(
        result.steps.len(),
        4,
        "{}: should have 4 steps (3 tool rounds + final)",
        h.provider_name
    );
    // First 3 steps should each have tool calls
    for i in 0..3 {
        assert!(
            !result.steps[i].tool_calls.is_empty(),
            "{}: step {} should have tool calls",
            h.provider_name,
            i
        );
    }
    // Last step should be the final text
    assert_eq!(
        result.steps[3].text, "All done after 3 rounds",
        "{}: last step text mismatch",
        h.provider_name
    );
}

#[tokio::test]
async fn test_tool_multi_step_anthropic() {
    verify_tool_multi_step(&ProviderTestHarness::anthropic().await).await;
}
#[tokio::test]
async fn test_tool_multi_step_openai() {
    verify_tool_multi_step(&ProviderTestHarness::openai().await).await;
}
#[tokio::test]
async fn test_tool_multi_step_gemini() {
    verify_tool_multi_step(&ProviderTestHarness::gemini().await).await;
}

// ============================================================================
// SSE helpers for streaming tool calls (Phase 3 T18)
// ============================================================================

fn anthropic_stream_tool_call_sse() -> String {
    build_sse_body(&[
        (
            "message_start",
            r#"{"type":"message_start","message":{"id":"msg_stream_tc","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-20250514","usage":{"input_tokens":20,"output_tokens":0}}}"#,
        ),
        (
            "content_block_start",
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_stream","name":"get_weather"}}"#,
        ),
        (
            "content_block_delta",
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"city\":\"SF\"}"}}"#,
        ),
        (
            "content_block_stop",
            r#"{"type":"content_block_stop","index":0}"#,
        ),
        (
            "message_delta",
            r#"{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":15}}"#,
        ),
        ("message_stop", r#"{"type":"message_stop"}"#),
    ])
}

fn openai_stream_tool_call_sse() -> String {
    build_sse_body(&[
        (
            "response.created",
            r#"{"response":{"id":"resp_stream_tc","object":"response","status":"in_progress","model":"gpt-4o","output":[]},"sequence_number":0}"#,
        ),
        (
            "response.output_item.added",
            r#"{"item":{"type":"function_call","call_id":"call_stream","name":"get_weather","arguments":""},"output_index":0}"#,
        ),
        (
            "response.function_call_arguments.delta",
            r#"{"delta":"{\"city\":\"SF\"}","output_index":0}"#,
        ),
        (
            "response.function_call_arguments.done",
            r#"{"arguments":"{\"city\":\"SF\"}","output_index":0}"#,
        ),
        (
            "response.output_item.done",
            r#"{"item":{"type":"function_call","call_id":"call_stream","name":"get_weather","arguments":"{\"city\":\"SF\"}"},"output_index":0}"#,
        ),
        (
            "response.completed",
            r#"{"response":{"id":"resp_stream_tc","object":"response","status":"completed","model":"gpt-4o","output":[{"type":"function_call","call_id":"call_stream","name":"get_weather","arguments":"{\"city\":\"SF\"}"}],"usage":{"input_tokens":20,"output_tokens":15,"total_tokens":35}},"sequence_number":5}"#,
        ),
    ])
}

fn gemini_stream_tool_call_sse() -> String {
    "data: {\"responseId\":\"resp_stream_tc\",\"modelVersion\":\"gemini-2.0-flash\",\"candidates\":[{\"content\":{\"parts\":[{\"functionCall\":{\"name\":\"get_weather\",\"args\":{\"city\":\"SF\"}}}],\"role\":\"model\"},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":20,\"candidatesTokenCount\":15,\"totalTokenCount\":35}}\n\n".to_string()
}

/// Mount a sequence of SSE streaming responses.
async fn mount_sequenced_stream_responses(h: &ProviderTestHarness, responses: Vec<String>) {
    for (i, sse_body) in responses.into_iter().enumerate() {
        let template = wiremock::ResponseTemplate::new(200)
            .insert_header("content-type", "text/event-stream")
            .set_body_string(sse_body);
        let priority = (i + 1) as u8;

        if h.provider_name == "gemini" {
            wiremock::Mock::given(wiremock::matchers::method("POST"))
                .and(wiremock::matchers::path_regex(
                    r"/v1beta/models/.+:streamGenerateContent",
                ))
                .respond_with(template)
                .up_to_n_times(1)
                .with_priority(priority)
                .mount(&h.server)
                .await;
        } else {
            let path = match h.provider_name.as_str() {
                "anthropic" => "/v1/messages",
                "openai" => "/v1/responses",
                _ => panic!("Unknown provider"),
            };
            wiremock::Mock::given(wiremock::matchers::method("POST"))
                .and(wiremock::matchers::path(path))
                .respond_with(template)
                .up_to_n_times(1)
                .with_priority(priority)
                .mount(&h.server)
                .await;
        }
    }
}

// ============================================================================
// DoD 8.9.8: Streaming with tool calls via stream() (×3)
// ============================================================================

async fn verify_stream_with_tools(h: &ProviderTestHarness) {
    let tc_sse = match h.provider_name.as_str() {
        "anthropic" => anthropic_stream_tool_call_sse(),
        "openai" => openai_stream_tool_call_sse(),
        "gemini" => gemini_stream_tool_call_sse(),
        _ => panic!("Unknown provider"),
    };
    let text_sse = match h.provider_name.as_str() {
        "anthropic" => anthropic_stream_text_sse("Weather result: sunny"),
        "openai" => openai_stream_text_sse("Weather result: sunny"),
        "gemini" => gemini_stream_text_sse("Weather result: sunny"),
        _ => panic!("Unknown provider"),
    };

    mount_sequenced_stream_responses(h, vec![tc_sse, text_sse]).await;

    let tool = Tool::active(
        "get_weather",
        "Get weather",
        serde_json::json!({"type": "object", "properties": {"city": {"type": "string"}}}),
        |_| Box::pin(async move { Ok(serde_json::json!({"weather": "sunny"})) }),
    );

    let opts = GenerateOptions::new("test-model")
        .prompt("What's the weather?")
        .tools(vec![tool])
        .max_tool_rounds(1);
    let mut result = api::stream(opts, &h.client).unwrap();

    let mut events = Vec::new();
    while let Some(event) = result.next().await {
        events.push(event.unwrap());
    }

    // Should have events from BOTH steps (tool call step + final text step)
    let types: Vec<_> = events.iter().map(|e| e.event_type.clone()).collect();
    assert!(
        types.contains(&StreamEventType::TextDelta),
        "{}: should have TextDelta events from final step",
        h.provider_name
    );
    assert!(
        types.contains(&StreamEventType::Finish),
        "{}: should have Finish event",
        h.provider_name
    );

    // Final accumulated text should match
    let text: String = events
        .iter()
        .filter(|e| e.event_type == StreamEventType::TextDelta)
        .filter_map(|e| e.delta.as_ref())
        .cloned()
        .collect();
    assert_eq!(
        text, "Weather result: sunny",
        "{}: accumulated text mismatch",
        h.provider_name
    );

    // YELLOW-6: Verify tool call arguments are correctly accumulated
    // (not just that TOOL_CALL events exist, but that they carry expected data)

    // ToolCallStart should exist with the correct tool name
    let tc_start = events
        .iter()
        .find(|e| e.event_type == StreamEventType::ToolCallStart);
    assert!(
        tc_start.is_some(),
        "{}: should have ToolCallStart event from tool call step",
        h.provider_name
    );
    let tc_start = tc_start.unwrap();
    assert!(
        tc_start.tool_call.is_some(),
        "{}: ToolCallStart should carry tool_call data",
        h.provider_name
    );
    assert_eq!(
        tc_start.tool_call.as_ref().unwrap().name,
        "get_weather",
        "{}: tool call name should be 'get_weather'",
        h.provider_name
    );

    // ToolCallEnd should exist
    let tc_end = events
        .iter()
        .find(|e| e.event_type == StreamEventType::ToolCallEnd);
    assert!(
        tc_end.is_some(),
        "{}: should have ToolCallEnd event",
        h.provider_name
    );

    // Verify the accumulated tool call arguments contain {"city": "SF"}.
    // Check both the final ToolCallEnd event and accumulated deltas for robustness,
    // since providers may place the final args on either.
    let tc_deltas: String = events
        .iter()
        .filter(|e| e.event_type == StreamEventType::ToolCallDelta)
        .filter_map(|e| e.delta.as_ref())
        .cloned()
        .collect();
    let has_args_in_end = tc_end
        .and_then(|e| e.tool_call.as_ref())
        .map(|tc| tc.arguments.get("city").and_then(|v| v.as_str()) == Some("SF"))
        .unwrap_or(false);
    let has_args_in_deltas = tc_deltas.contains("city") && tc_deltas.contains("SF");
    assert!(
        has_args_in_end || has_args_in_deltas,
        "{}: tool call arguments should contain {{\"city\":\"SF\"}} — \
         end has args: {}, deltas: '{}'",
        h.provider_name,
        has_args_in_end,
        tc_deltas
    );
}

#[tokio::test]
async fn test_stream_with_tools_anthropic() {
    verify_stream_with_tools(&ProviderTestHarness::anthropic().await).await;
}
#[tokio::test]
async fn test_stream_with_tools_openai() {
    verify_stream_with_tools(&ProviderTestHarness::openai().await).await;
}
#[tokio::test]
async fn test_stream_with_tools_gemini() {
    verify_stream_with_tools(&ProviderTestHarness::gemini().await).await;
}

// ============================================================================
// DoD 8.9.9: Structured output via generate_object() (×3)
// ============================================================================

async fn verify_structured_output(h: &ProviderTestHarness) {
    let json_text = r#"{"name": "Alice", "age": 30}"#;
    mount_text_response(h, json_text).await;

    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name", "age"]
    });

    let opts = GenerateOptions::new("test-model").prompt("Return a person");
    let result = api::generate_object(opts, schema, &h.client).await.unwrap();

    // output should be the parsed JSON value
    assert_eq!(
        result.output.as_ref().unwrap()["name"],
        "Alice",
        "{}: name mismatch",
        h.provider_name
    );
    assert_eq!(
        result.output.as_ref().unwrap()["age"],
        30,
        "{}: age mismatch",
        h.provider_name
    );
    // text should be the raw JSON string
    assert!(
        result.text.contains("Alice"),
        "{}: raw text should contain Alice",
        h.provider_name
    );
}

#[tokio::test]
async fn test_structured_output_anthropic() {
    verify_structured_output(&ProviderTestHarness::anthropic().await).await;
}
#[tokio::test]
async fn test_structured_output_openai() {
    verify_structured_output(&ProviderTestHarness::openai().await).await;
}
#[tokio::test]
async fn test_structured_output_gemini() {
    verify_structured_output(&ProviderTestHarness::gemini().await).await;
}

// ============================================================================
// Phase 3 T19: Caching response helpers
// ============================================================================

/// Anthropic: cache tokens reported in top-level usage
fn anthropic_cached_response(
    text: &str,
    input: u32,
    output: u32,
    cache_read: u32,
) -> serde_json::Value {
    serde_json::json!({
        "id": "msg_cache", "type": "message", "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": "claude-sonnet-4-20250514", "stop_reason": "end_turn",
        "usage": {
            "input_tokens": input,
            "output_tokens": output,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": cache_read
        }
    })
}

/// OpenAI: cache tokens reported in input_tokens_details
fn openai_cached_response(
    text: &str,
    input: u32,
    output: u32,
    cache_read: u32,
) -> serde_json::Value {
    serde_json::json!({
        "id": "resp_cache", "object": "response", "status": "completed", "model": "gpt-4o",
        "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}],
        "usage": {
            "input_tokens": input,
            "output_tokens": output,
            "total_tokens": input + output,
            "input_tokens_details": {"cached_tokens": cache_read}
        }
    })
}

/// Gemini: cache tokens reported in cachedContentTokenCount
fn gemini_cached_response(
    text: &str,
    input: u32,
    output: u32,
    cache_read: u32,
) -> serde_json::Value {
    serde_json::json!({
        "responseId": "resp_cache", "modelVersion": "gemini-2.0-flash",
        "candidates": [{"content": {"parts": [{"text": text}], "role": "model"}, "finishReason": "STOP"}],
        "usageMetadata": {
            "promptTokenCount": input,
            "candidatesTokenCount": output,
            "totalTokenCount": input + output,
            "cachedContentTokenCount": cache_read
        }
    })
}

// ============================================================================
// Cache token field parsing via generate() (×3)
// These tests validate that cache_read_tokens are correctly extracted from
// each provider's JSON response format. They are NOT real caching tests —
// the cache_read values are baked into the wiremock responses.
// Real multi-turn caching is tested in compliance_harness.rs (8.6.9).
// ============================================================================

async fn verify_multi_turn_caching(h: &ProviderTestHarness) {
    // Simulate 5 turns. Each turn gets a fresh wiremock mount.
    // Turn usage simulates growing context with increasing cache hits:
    //   Turn 1: input=100, cache_read=0   (cold start)
    //   Turn 2: input=200, cache_read=50  (25% cached)
    //   Turn 3: input=300, cache_read=150 (50% cached)
    //   Turn 4: input=400, cache_read=250 (62% cached)
    //   Turn 5: input=500, cache_read=350 (70% cached) ← must be >50%
    let turns: Vec<(u32, u32, u32)> = vec![
        (100, 10, 0),
        (200, 10, 50),
        (300, 10, 150),
        (400, 10, 250),
        (500, 10, 350),
    ];

    let mut messages = Vec::new();
    let mut last_result = None;

    for (turn_idx, (input, output, cache_read)) in turns.iter().enumerate() {
        let text = format!("Response for turn {}", turn_idx + 1);

        // Mount a fresh response for this turn
        let body = match h.provider_name.as_str() {
            "anthropic" => anthropic_cached_response(&text, *input, *output, *cache_read),
            "openai" => openai_cached_response(&text, *input, *output, *cache_read),
            "gemini" => gemini_cached_response(&text, *input, *output, *cache_read),
            _ => panic!("Unknown provider"),
        };
        mount_sequenced_responses(h, vec![(body, 200)]).await;

        // Build conversation: alternate user/assistant messages
        messages.push(Message::user(format!("User message turn {}", turn_idx + 1)));

        let opts = GenerateOptions::new("test-model").messages(messages.clone());
        let result = api::generate(opts, &h.client).await.unwrap();

        // Append assistant response to conversation for next turn
        messages.push(Message::assistant(&result.text));
        last_result = Some(result);
    }

    // Verify turn 5: cache_read_tokens > 50% of input_tokens
    let final_result = last_result.unwrap();
    let usage = &final_result.total_usage;
    let cache_read = usage.cache_read_tokens.unwrap_or(0);
    let input = usage.input_tokens;
    assert!(
        cache_read > input / 2,
        "{}: Turn 5 cache_read_tokens ({}) should be > 50% of input_tokens ({}). Got {}%",
        h.provider_name,
        cache_read,
        input,
        if input > 0 {
            cache_read * 100 / input
        } else {
            0
        }
    );
    assert!(
        cache_read > 0,
        "{}: cache_read_tokens should be non-zero on turn 5",
        h.provider_name
    );
}

#[tokio::test]
async fn test_cache_token_field_parsing_anthropic() {
    verify_multi_turn_caching(&ProviderTestHarness::anthropic().await).await;
}
#[tokio::test]
async fn test_cache_token_field_parsing_openai() {
    verify_multi_turn_caching(&ProviderTestHarness::openai().await).await;
}
#[tokio::test]
async fn test_cache_token_field_parsing_gemini() {
    verify_multi_turn_caching(&ProviderTestHarness::gemini().await).await;
}
