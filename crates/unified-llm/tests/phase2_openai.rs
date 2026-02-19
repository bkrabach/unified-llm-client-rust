//! Phase 2B Integration Tests: Full OpenAI stack through Client.
//!
//! These tests verify the complete OpenAI adapter works end-to-end
//! through the Client abstraction, matching the pattern from phase1_anthropic.rs.

use futures::StreamExt;
use secrecy::SecretString;
use unified_llm::client::ClientBuilder;
use unified_llm::providers::openai::OpenAiAdapter;
use unified_llm_types::*;

fn build_sse_body(events: &[(&str, &str)]) -> String {
    events
        .iter()
        .map(|(t, d)| format!("event: {t}\ndata: {d}\n\n"))
        .collect()
}

#[tokio::test]
async fn test_openai_complete_through_client() {
    let server = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .and(wiremock::matchers::path("/v1/responses"))
        .and(wiremock::matchers::header_exists("authorization"))
        .respond_with(wiremock::ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "resp_int", "object": "response", "status": "completed", "model": "gpt-4o",
            "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "OpenAI works!"}]}],
            "usage": {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14}
        })))
        .mount(&server)
        .await;

    let adapter =
        OpenAiAdapter::new_with_base_url(SecretString::from("sk-test".to_string()), server.uri());
    let client = ClientBuilder::new()
        .provider("openai", Box::new(adapter))
        .build()
        .unwrap();
    let resp = client
        .complete(
            Request::default()
                .model("gpt-4o")
                .messages(vec![Message::user("Test")]),
        )
        .await
        .unwrap();

    assert_eq!(resp.text(), "OpenAI works!");
    assert_eq!(resp.provider, "openai");
    assert_eq!(resp.usage.total_tokens, 14);
}

#[tokio::test]
async fn test_openai_stream_through_client() {
    let sse_body = build_sse_body(&[
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
            r#"{"delta":"Stream","output_index":0,"content_index":0}"#,
        ),
        (
            "response.output_text.delta",
            r#"{"delta":"ing!","output_index":0,"content_index":0}"#,
        ),
        (
            "response.output_text.done",
            r#"{"text":"Streaming!","output_index":0,"content_index":0}"#,
        ),
        (
            "response.output_item.done",
            r#"{"item":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Streaming!"}]},"output_index":0}"#,
        ),
        (
            "response.completed",
            r#"{"response":{"id":"resp_stream","object":"response","status":"completed","model":"gpt-4o","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Streaming!"}]}],"usage":{"input_tokens":8,"output_tokens":3,"total_tokens":11}},"sequence_number":5}"#,
        ),
    ]);

    let server = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .and(wiremock::matchers::path("/v1/responses"))
        .respond_with(
            wiremock::ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse_body),
        )
        .mount(&server)
        .await;

    let adapter =
        OpenAiAdapter::new_with_base_url(SecretString::from("sk-test".to_string()), server.uri());
    let client = ClientBuilder::new()
        .provider("openai", Box::new(adapter))
        .build()
        .unwrap();
    let stream = client
        .stream(
            Request::default()
                .model("gpt-4o")
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
    let types: Vec<_> = events.iter().map(|e| e.event_type.clone()).collect();
    assert!(types.contains(&StreamEventType::StreamStart));
    assert!(types.contains(&StreamEventType::TextStart));
    assert!(types.contains(&StreamEventType::TextDelta));
    assert!(types.contains(&StreamEventType::TextEnd));
    assert!(types.contains(&StreamEventType::Finish));

    // Verify accumulated text
    let text: String = events
        .iter()
        .filter(|e| e.event_type == StreamEventType::TextDelta)
        .filter_map(|e| e.delta.as_ref())
        .cloned()
        .collect();
    assert_eq!(text, "Streaming!");

    // Verify finish has usage
    let finish = events
        .iter()
        .find(|e| e.event_type == StreamEventType::Finish)
        .unwrap();
    let usage = finish.usage.as_ref().unwrap();
    assert_eq!(usage.input_tokens, 8);
    assert_eq!(usage.output_tokens, 3);
    assert_eq!(usage.total_tokens, 11);
}

#[tokio::test]
async fn test_openai_error_through_client() {
    let server = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .respond_with(wiremock::ResponseTemplate::new(401).set_body_json(serde_json::json!({
            "error": {"message": "Invalid API key", "type": "invalid_request_error", "code": "invalid_api_key"}
        })))
        .mount(&server)
        .await;

    let adapter =
        OpenAiAdapter::new_with_base_url(SecretString::from("bad-key".to_string()), server.uri());
    let client = ClientBuilder::new()
        .provider("openai", Box::new(adapter))
        .build()
        .unwrap();
    let err = client
        .complete(
            Request::default()
                .model("gpt-4o")
                .messages(vec![Message::user("hi")]),
        )
        .await
        .unwrap_err();
    assert_eq!(err.kind, ErrorKind::Authentication);
    assert_eq!(err.provider, Some("openai".into()));
}

#[tokio::test]
async fn test_openai_rate_limit_with_retry_after() {
    let server = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .respond_with(
            wiremock::ResponseTemplate::new(429)
                .insert_header("retry-after", "30")
                .set_body_json(serde_json::json!({
                    "error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}
                })),
        )
        .mount(&server)
        .await;

    let adapter =
        OpenAiAdapter::new_with_base_url(SecretString::from("sk-test".to_string()), server.uri());
    let client = ClientBuilder::new()
        .provider("openai", Box::new(adapter))
        .build()
        .unwrap();
    let err = client
        .complete(
            Request::default()
                .model("gpt-4o")
                .messages(vec![Message::user("hi")]),
        )
        .await
        .unwrap_err();
    assert_eq!(err.kind, ErrorKind::RateLimit);
    assert!(err.retryable);
    assert_eq!(err.retry_after, Some(std::time::Duration::from_secs(30)));
}

#[tokio::test]
async fn test_openai_stream_with_tool_calls() {
    let sse_body = build_sse_body(&[
        (
            "response.created",
            r#"{"response":{"id":"resp_tc","status":"in_progress","model":"gpt-4o","output":[]},"sequence_number":0}"#,
        ),
        (
            "response.output_item.added",
            r#"{"item":{"type":"function_call","call_id":"call_int1","name":"get_weather","arguments":""},"output_index":0}"#,
        ),
        (
            "response.function_call_arguments.delta",
            r#"{"delta":"{\"city\":\"NYC\"}","output_index":0}"#,
        ),
        (
            "response.function_call_arguments.done",
            r#"{"arguments":"{\"city\":\"NYC\"}","output_index":0}"#,
        ),
        (
            "response.output_item.done",
            r#"{"item":{"type":"function_call","call_id":"call_int1","name":"get_weather","arguments":"{\"city\":\"NYC\"}"},"output_index":0}"#,
        ),
        (
            "response.completed",
            r#"{"response":{"id":"resp_tc","status":"completed","model":"gpt-4o","output":[{"type":"function_call","call_id":"call_int1","name":"get_weather","arguments":"{\"city\":\"NYC\"}"}],"usage":{"input_tokens":15,"output_tokens":8,"total_tokens":23}},"sequence_number":5}"#,
        ),
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

    let adapter =
        OpenAiAdapter::new_with_base_url(SecretString::from("sk-test".to_string()), server.uri());
    let client = ClientBuilder::new()
        .provider("openai", Box::new(adapter))
        .build()
        .unwrap();
    let stream = client
        .stream(
            Request::default()
                .model("gpt-4o")
                .messages(vec![Message::user("weather?")]),
        )
        .unwrap();
    let events: Vec<StreamEvent> = stream
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .filter_map(|r| r.ok())
        .collect();

    let types: Vec<_> = events.iter().map(|e| e.event_type.clone()).collect();
    assert!(types.contains(&StreamEventType::ToolCallStart));
    assert!(types.contains(&StreamEventType::ToolCallDelta));
    assert!(types.contains(&StreamEventType::ToolCallEnd));
    assert!(types.contains(&StreamEventType::Finish));

    // Verify tool call end has correct data
    let tc_end = events
        .iter()
        .find(|e| e.event_type == StreamEventType::ToolCallEnd)
        .unwrap();
    let tc = tc_end.tool_call.as_ref().unwrap();
    assert_eq!(tc.name, "get_weather");
    assert_eq!(tc.arguments["city"], "NYC");
}

#[tokio::test]
async fn test_openai_request_body_structure() {
    // Verify the request body sent to OpenAI has the correct structure
    let server = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .and(wiremock::matchers::path("/v1/responses"))
        .respond_with(wiremock::ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "resp_body", "object": "response", "status": "completed", "model": "gpt-4o",
            "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]}],
            "usage": {"input_tokens": 5, "output_tokens": 1, "total_tokens": 6}
        })))
        .mount(&server)
        .await;

    let adapter =
        OpenAiAdapter::new_with_base_url(SecretString::from("sk-test".to_string()), server.uri());
    let client = ClientBuilder::new()
        .provider("openai", Box::new(adapter))
        .build()
        .unwrap();

    // Build a request with system message, tools, and provider options
    let _resp = client
        .complete(
            Request::default()
                .model("gpt-4o")
                .messages(vec![
                    Message::system("You are helpful."),
                    Message::user("Hello"),
                ])
                .tools(vec![ToolDefinition {
                    name: "test_fn".into(),
                    description: "A test function".into(),
                    parameters: serde_json::json!({"type": "object"}),
                    strict: None,
                }])
                .max_tokens(100),
        )
        .await
        .unwrap();

    // Verify request was received (mock matched means all matchers passed)
    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);

    // Parse the request body to verify structure
    let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(body["model"], "gpt-4o");
    assert_eq!(body["instructions"], "You are helpful.");
    assert!(body["input"].is_array());
    assert_eq!(body["max_output_tokens"], 100);
    assert!(body.get("max_tokens").is_none()); // NOT max_tokens
    assert!(body["tools"].is_array());
    assert_eq!(body["tools"][0]["type"], "function");
    assert_eq!(body["tools"][0]["strict"], true);
}

#[tokio::test]
async fn test_openai_stream_error_propagation() {
    let server = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .respond_with(
            wiremock::ResponseTemplate::new(500).set_body_json(serde_json::json!({
                "error": {"message": "Internal server error"}
            })),
        )
        .mount(&server)
        .await;

    let adapter =
        OpenAiAdapter::new_with_base_url(SecretString::from("sk-test".to_string()), server.uri());
    let client = ClientBuilder::new()
        .provider("openai", Box::new(adapter))
        .build()
        .unwrap();
    let stream = client
        .stream(
            Request::default()
                .model("gpt-4o")
                .messages(vec![Message::user("hi")]),
        )
        .unwrap();
    let results: Vec<Result<StreamEvent, Error>> = stream.collect::<Vec<_>>().await;
    assert_eq!(results.len(), 1);
    assert!(results[0].is_err());
    assert_eq!(results[0].as_ref().unwrap_err().kind, ErrorKind::Server);
}
