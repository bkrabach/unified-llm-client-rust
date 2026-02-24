//! Phase 2C Integration Tests: Full Gemini stack through Client.
//!
//! These tests verify the complete Gemini adapter works end-to-end
//! through the Client abstraction, matching the pattern from phase1_anthropic.rs
//! and phase2_openai.rs.

use futures::StreamExt;
use secrecy::SecretString;
use unified_llm::client::ClientBuilder;
use unified_llm::providers::gemini::GeminiAdapter;
use unified_llm_types::*;

#[tokio::test]
async fn test_gemini_complete_through_client() {
    let server = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .and(wiremock::matchers::path_regex(
            r"/v1beta/models/.+:generateContent",
        ))
        // Auth is via ?key= query param (spec §7.8) — wiremock path_regex
        // matcher handles the path; the key is verified in the dedicated auth test.
        .respond_with(
            wiremock::ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "responseId": "resp_int",
                "modelVersion": "gemini-2.0-flash",
                "candidates": [{
                    "content": {
                        "parts": [{"text": "Gemini works!"}],
                        "role": "model"
                    },
                    "finishReason": "STOP"
                }],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 4,
                    "totalTokenCount": 14
                }
            })),
        )
        .mount(&server)
        .await;

    let adapter =
        GeminiAdapter::new_with_base_url(SecretString::from("test-key".to_string()), server.uri());
    let client = ClientBuilder::new()
        .provider("gemini", Box::new(adapter))
        .build()
        .unwrap();
    let resp = client
        .complete(
            Request::default()
                .model("gemini-2.0-flash")
                .messages(vec![Message::user("Test")]),
        )
        .await
        .unwrap();

    assert_eq!(resp.text(), "Gemini works!");
    assert_eq!(resp.provider, "gemini");
    assert_eq!(resp.id, "resp_int");
    assert_eq!(resp.usage.input_tokens, 10);
    assert_eq!(resp.usage.output_tokens, 4);
    assert_eq!(resp.usage.total_tokens, 14);
}

#[tokio::test]
async fn test_gemini_stream_through_client() {
    let sse_body = [
        "data: {\"responseId\":\"resp_stream\",\"modelVersion\":\"gemini-2.0-flash\",\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hello\"}],\"role\":\"model\"}}]}\n\n",
        "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\" from\"}],\"role\":\"model\"}}]}\n\n",
        "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\" Gemini!\"}],\"role\":\"model\"},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":8,\"candidatesTokenCount\":3,\"totalTokenCount\":11}}\n\n",
    ].join("");

    let server = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .and(wiremock::matchers::path_regex(
            r"/v1beta/models/.+:streamGenerateContent",
        ))
        .respond_with(
            wiremock::ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse_body),
        )
        .mount(&server)
        .await;

    let adapter =
        GeminiAdapter::new_with_base_url(SecretString::from("test-key".to_string()), server.uri());
    let client = ClientBuilder::new()
        .provider("gemini", Box::new(adapter))
        .build()
        .unwrap();
    let stream = client
        .stream(
            Request::default()
                .model("gemini-2.0-flash")
                .messages(vec![Message::user("Stream test")]),
        )
        .unwrap();
    let events: Vec<StreamEvent> = stream
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .filter_map(|r| r.ok())
        .collect();

    // Verify event sequence
    let types: Vec<_> = events.iter().map(|e| &e.event_type).collect();
    assert!(types.contains(&&StreamEventType::StreamStart));
    assert!(types.contains(&&StreamEventType::TextStart));
    assert!(types.contains(&&StreamEventType::TextDelta));
    assert!(types.contains(&&StreamEventType::TextEnd));
    assert!(types.contains(&&StreamEventType::Finish));

    // Verify text content
    let text: String = events
        .iter()
        .filter(|e| e.event_type == StreamEventType::TextDelta)
        .filter_map(|e| e.delta.as_deref())
        .collect();
    assert_eq!(text, "Hello from Gemini!");

    // Verify finish has usage
    let finish = events
        .iter()
        .find(|e| e.event_type == StreamEventType::Finish)
        .unwrap();
    assert_eq!(finish.usage.as_ref().unwrap().total_tokens, 11);
    assert_eq!(finish.finish_reason.as_ref().unwrap().reason, "stop");
}

#[tokio::test]
async fn test_gemini_error_through_client() {
    let server = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .respond_with(
            wiremock::ResponseTemplate::new(429).set_body_json(serde_json::json!({
                "error": {
                    "code": 429,
                    "message": "Resource has been exhausted",
                    "status": "RESOURCE_EXHAUSTED"
                }
            })),
        )
        .mount(&server)
        .await;

    let adapter =
        GeminiAdapter::new_with_base_url(SecretString::from("test-key".to_string()), server.uri());
    let client = ClientBuilder::new()
        .provider("gemini", Box::new(adapter))
        .build()
        .unwrap();

    let err = client
        .complete(
            Request::default()
                .model("gemini-2.0-flash")
                .messages(vec![Message::user("Hello")]),
        )
        .await
        .unwrap_err();

    assert_eq!(err.kind, ErrorKind::RateLimit);
    assert!(err.retryable);
    assert!(err.message.contains("exhausted"));
    assert_eq!(err.error_code, Some("RESOURCE_EXHAUSTED".to_string()));
    assert_eq!(err.provider, Some("gemini".to_string()));
}

#[tokio::test]
async fn test_gemini_tool_call_with_synthetic_id() {
    let server = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .respond_with(
            wiremock::ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "responseId": "resp_tc",
                "modelVersion": "gemini-2.0-flash",
                "candidates": [{
                    "content": {
                        "parts": [{
                            "functionCall": {
                                "name": "get_weather",
                                "args": {"city": "NYC", "unit": "fahrenheit"}
                            }
                        }],
                        "role": "model"
                    },
                    "finishReason": "STOP"
                }],
                "usageMetadata": {
                    "promptTokenCount": 15,
                    "candidatesTokenCount": 8,
                    "totalTokenCount": 23
                }
            })),
        )
        .mount(&server)
        .await;

    let adapter =
        GeminiAdapter::new_with_base_url(SecretString::from("test-key".to_string()), server.uri());
    let client = ClientBuilder::new()
        .provider("gemini", Box::new(adapter))
        .build()
        .unwrap();

    let resp = client
        .complete(
            Request::default()
                .model("gemini-2.0-flash")
                .messages(vec![Message::user("What's the weather in NYC?")])
                .tools(vec![ToolDefinition {
                    name: "get_weather".to_string(),
                    description: "Get weather for a city".to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "unit": {"type": "string"}
                        }
                    }),
                    strict: None,
                }]),
        )
        .await
        .unwrap();

    // Finish reason should be tool_calls (inferred from functionCall presence)
    assert_eq!(resp.finish_reason.reason, "tool_calls");

    // Should have exactly one tool call
    let tool_calls = resp.tool_calls();
    assert_eq!(tool_calls.len(), 1);

    // Synthetic ID should start with "call_" (since Gemini didn't provide one)
    assert!(
        tool_calls[0].id.starts_with("call_"),
        "Expected synthetic ID starting with 'call_', got: {}",
        tool_calls[0].id
    );

    // Function name and arguments should be correct
    assert_eq!(tool_calls[0].name, "get_weather");
    // ToolCall.arguments is already a parsed Map<String, Value>
    assert_eq!(
        tool_calls[0].arguments.get("city").unwrap(),
        &serde_json::json!("NYC")
    );
    assert_eq!(
        tool_calls[0].arguments.get("unit").unwrap(),
        &serde_json::json!("fahrenheit")
    );
}

#[tokio::test]
async fn test_gemini_request_body_structure() {
    let server = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .respond_with(wiremock::ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "candidates": [{
                "content": {"parts": [{"text": "ok"}], "role": "model"},
                "finishReason": "STOP"
            }],
            "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2}
        })))
        .mount(&server)
        .await;

    let adapter = GeminiAdapter::new_with_base_url(
        SecretString::from("test-gemini-key".to_string()),
        server.uri(),
    );
    let client = ClientBuilder::new()
        .provider("gemini", Box::new(adapter))
        .build()
        .unwrap();

    client
        .complete(
            Request::default()
                .model("gemini-2.0-flash")
                .messages(vec![Message::system("Be helpful"), Message::user("Hi")])
                .temperature(0.5),
        )
        .await
        .unwrap();

    // Verify the request
    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);

    // Model in URL path, NOT in body
    assert!(requests[0]
        .url
        .path()
        .contains("gemini-2.0-flash:generateContent"));

    // Auth via query param (spec §7.8)
    let url_str = requests[0].url.to_string();
    assert!(
        url_str.contains("key=test-gemini-key"),
        "API key should be in URL query param, got: {url_str}"
    );
    assert!(!url_str.is_empty(), "API key should NOT appear in URL");

    // Verify body structure
    let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert!(
        body.get("systemInstruction").is_some(),
        "should have systemInstruction"
    );
    assert!(body.get("contents").is_some(), "should have contents");
    assert!(
        body.get("generationConfig").is_some(),
        "should have generationConfig"
    );
    assert!(body.get("model").is_none(), "model should NOT be in body");
}

#[tokio::test]
async fn test_gemini_stream_with_tool_calls() {
    // Gemini sends function calls complete in one chunk
    let sse_body = "data: {\"responseId\":\"resp_tc_stream\",\"modelVersion\":\"gemini-2.0-flash\",\"candidates\":[{\"content\":{\"parts\":[{\"functionCall\":{\"name\":\"get_weather\",\"args\":{\"city\":\"London\"}}}],\"role\":\"model\"},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":12,\"candidatesTokenCount\":6,\"totalTokenCount\":18}}\n\n";

    let server = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .respond_with(
            wiremock::ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse_body),
        )
        .mount(&server)
        .await;

    let adapter =
        GeminiAdapter::new_with_base_url(SecretString::from("test-key".to_string()), server.uri());
    let client = ClientBuilder::new()
        .provider("gemini", Box::new(adapter))
        .build()
        .unwrap();

    let stream = client
        .stream(
            Request::default()
                .model("gemini-2.0-flash")
                .messages(vec![Message::user("Weather in London?")]),
        )
        .unwrap();

    let events: Vec<StreamEvent> = stream
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .filter_map(|r| r.ok())
        .collect();

    let types: Vec<_> = events.iter().map(|e| &e.event_type).collect();

    // Function calls arrive complete — both Start and End in same batch
    assert!(types.contains(&&StreamEventType::ToolCallStart));
    assert!(types.contains(&&StreamEventType::ToolCallEnd));
    assert!(types.contains(&&StreamEventType::Finish));

    // Verify tool call details
    let tc_start = events
        .iter()
        .find(|e| e.event_type == StreamEventType::ToolCallStart)
        .unwrap();
    let tc = tc_start.tool_call.as_ref().unwrap();
    assert_eq!(tc.name, "get_weather");
    assert!(tc.id.starts_with("call_"));

    // Finish reason should be tool_calls
    let finish = events
        .iter()
        .find(|e| e.event_type == StreamEventType::Finish)
        .unwrap();
    assert_eq!(finish.finish_reason.as_ref().unwrap().reason, "tool_calls");
}

#[tokio::test]
async fn test_gemini_stream_error_propagation() {
    let server = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .respond_with(
            wiremock::ResponseTemplate::new(401).set_body_json(serde_json::json!({
                "error": {
                    "code": 401,
                    "message": "API key not valid. Please pass a valid API key.",
                    "status": "UNAUTHENTICATED"
                }
            })),
        )
        .mount(&server)
        .await;

    let adapter =
        GeminiAdapter::new_with_base_url(SecretString::from("bad-key".to_string()), server.uri());
    let client = ClientBuilder::new()
        .provider("gemini", Box::new(adapter))
        .build()
        .unwrap();

    let stream = client
        .stream(
            Request::default()
                .model("gemini-2.0-flash")
                .messages(vec![Message::user("Hello")]),
        )
        .unwrap();

    let events: Vec<Result<StreamEvent, Error>> = stream.collect::<Vec<_>>().await;
    assert_eq!(events.len(), 1);
    let err = events[0].as_ref().unwrap_err();
    assert_eq!(err.kind, ErrorKind::Authentication);
    assert_eq!(err.error_code, Some("UNAUTHENTICATED".to_string()));
}
