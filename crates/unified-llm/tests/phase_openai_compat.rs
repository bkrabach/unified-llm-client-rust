//! Integration Tests: OpenAI-Compatible adapter through Client.
//!
//! These tests verify the complete Chat Completions adapter works end-to-end
//! through the Client abstraction using wiremock.

#![cfg(feature = "openai-compat")]

use futures::StreamExt;
use secrecy::SecretString;
use unified_llm::client::ClientBuilder;
use unified_llm::providers::openai_compat::OpenAICompatibleAdapter;
use unified_llm_types::*;

fn build_chat_sse_body(data_lines: &[&str]) -> String {
    data_lines
        .iter()
        .map(|d| format!("data: {d}\n\n"))
        .collect()
}

#[tokio::test]
async fn test_compat_complete_through_client() {
    let server = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .and(wiremock::matchers::path("/v1/chat/completions"))
        .and(wiremock::matchers::header_exists("authorization"))
        .respond_with(
            wiremock::ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "chatcmpl-int",
                "model": "llama-3.1-70b",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "Compat works!"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}
            })),
        )
        .mount(&server)
        .await;

    let adapter =
        OpenAICompatibleAdapter::new(SecretString::from("sk-test".to_string()), server.uri());
    let client = ClientBuilder::new()
        .provider("vllm", Box::new(adapter))
        .build()
        .unwrap();
    let resp = client
        .complete(
            Request::default()
                .model("llama-3.1-70b")
                .messages(vec![Message::user("Test")]),
        )
        .await
        .unwrap();

    assert_eq!(resp.text(), "Compat works!");
    assert_eq!(resp.provider, "openai-compatible");
    assert_eq!(resp.usage.total_tokens, 14);
}

#[tokio::test]
async fn test_compat_stream_through_client() {
    let sse_body = build_chat_sse_body(&[
        r#"{"id":"chatcmpl-s","model":"llama-3.1","choices":[{"index":0,"delta":{"role":"assistant","content":"Stream"},"finish_reason":null}]}"#,
        r#"{"id":"chatcmpl-s","choices":[{"index":0,"delta":{"content":"ing!"},"finish_reason":null}]}"#,
        r#"{"id":"chatcmpl-s","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":8,"completion_tokens":3,"total_tokens":11}}"#,
        "[DONE]",
    ]);

    let server = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .and(wiremock::matchers::path("/v1/chat/completions"))
        .respond_with(
            wiremock::ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse_body),
        )
        .mount(&server)
        .await;

    let adapter =
        OpenAICompatibleAdapter::new(SecretString::from("sk-test".to_string()), server.uri());
    let client = ClientBuilder::new()
        .provider("vllm", Box::new(adapter))
        .build()
        .unwrap();
    let stream = client
        .stream(
            Request::default()
                .model("llama-3.1")
                .messages(vec![Message::user("Stream test")]),
        )
        .unwrap();
    let events: Vec<StreamEvent> = stream
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .filter_map(|r| r.ok())
        .collect();

    let deltas: Vec<&str> = events
        .iter()
        .filter(|e| e.event_type == StreamEventType::TextDelta)
        .filter_map(|e| e.delta.as_deref())
        .collect();
    assert_eq!(deltas, vec!["Stream", "ing!"]);
    assert!(events
        .iter()
        .any(|e| e.event_type == StreamEventType::Finish));
}

#[tokio::test]
async fn test_compat_error_through_client() {
    let server = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .and(wiremock::matchers::path("/v1/chat/completions"))
        .respond_with(
            wiremock::ResponseTemplate::new(401).set_body_json(serde_json::json!({
                "error": {"message": "Invalid API key", "code": "invalid_api_key"}
            })),
        )
        .mount(&server)
        .await;

    let adapter =
        OpenAICompatibleAdapter::new(SecretString::from("bad-key".to_string()), server.uri());
    let client = ClientBuilder::new()
        .provider("vllm", Box::new(adapter))
        .build()
        .unwrap();
    let err = client
        .complete(
            Request::default()
                .model("gpt-4o")
                .messages(vec![Message::user("Hi")]),
        )
        .await
        .unwrap_err();

    assert!(matches!(err.kind, ErrorKind::Authentication));
    assert!(err.message.contains("Invalid API key"));
}

#[tokio::test]
async fn test_compat_configurable_base_url() {
    // Two different wiremock servers simulating vLLM and Together AI
    let server1 = wiremock::MockServer::start().await;
    let server2 = wiremock::MockServer::start().await;

    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .and(wiremock::matchers::path("/v1/chat/completions"))
        .respond_with(wiremock::ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "chatcmpl-1", "model": "llama-3.1",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "From vLLM server"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        })))
        .mount(&server1)
        .await;

    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .and(wiremock::matchers::path("/v1/chat/completions"))
        .respond_with(wiremock::ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "chatcmpl-2", "model": "mixtral",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "From Together server"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        })))
        .mount(&server2)
        .await;

    let adapter1 =
        OpenAICompatibleAdapter::new(SecretString::from("key1".to_string()), server1.uri());
    let adapter2 =
        OpenAICompatibleAdapter::new(SecretString::from("key2".to_string()), server2.uri());

    let client = ClientBuilder::new()
        .provider("vllm", Box::new(adapter1))
        .provider("together", Box::new(adapter2))
        .build()
        .unwrap();

    // Request to vLLM should hit server1
    let resp1 = client
        .complete(
            Request::default()
                .provider(Some("vllm".to_string()))
                .model("llama-3.1")
                .messages(vec![Message::user("Test")]),
        )
        .await
        .unwrap();
    assert_eq!(resp1.text(), "From vLLM server");

    // Request to Together should hit server2
    let resp2 = client
        .complete(
            Request::default()
                .provider(Some("together".to_string()))
                .model("mixtral")
                .messages(vec![Message::user("Test")]),
        )
        .await
        .unwrap();
    assert_eq!(resp2.text(), "From Together server");
}
