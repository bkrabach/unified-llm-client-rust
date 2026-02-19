//! Integration smoke tests with real API keys.
//!
//! Run with: cargo test -p unified-llm --test smoke_test -- --ignored
//!
//! Requires: ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY (or GOOGLE_API_KEY)
//!
//! Spec reference: S8.10 Integration Smoke Test

use unified_llm::api::types::GenerateOptions;
use unified_llm::catalog_data::get_latest_model;
use unified_llm::client::Client;

fn require_api_keys() -> Client {
    Client::from_env().expect(
        "Integration tests require API keys: \
         ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY",
    )
}

/// DoD 8.10.1: Basic generation across all 3 providers.
#[tokio::test]
async fn test_smoke_basic_generation_all_providers() {
    let client = require_api_keys();

    for provider_name in ["anthropic", "openai", "gemini"] {
        let model_info = get_latest_model(provider_name, None)
            .unwrap_or_else(|| panic!("No model found for {provider_name}"));

        let options = GenerateOptions::new(&model_info.id)
            .prompt("Say hello in one sentence.")
            .max_tokens(500)
            .provider(provider_name);

        let result = unified_llm::generate(options, &client)
            .await
            .unwrap_or_else(|e| panic!("{provider_name}: generate() failed: {e}"));

        assert!(
            !result.text.is_empty(),
            "{provider_name}: text should not be empty"
        );
        assert!(
            result.usage.input_tokens > 0,
            "{provider_name}: input_tokens should be > 0"
        );
        assert!(
            result.usage.output_tokens > 0,
            "{provider_name}: output_tokens should be > 0"
        );
        assert_eq!(
            result.finish_reason.reason, "stop",
            "{provider_name}: finish_reason should be 'stop', got '{}'",
            result.finish_reason.reason
        );
    }
}

/// DoD 8.10.2: Streaming.
#[tokio::test]
async fn test_smoke_streaming() {
    use futures::StreamExt;
    use unified_llm::api::stream::stream;
    use unified_llm_types::StreamEventType;

    let client = require_api_keys();

    let model = get_latest_model("anthropic", None).expect("No Anthropic model in catalog");

    let options = GenerateOptions::new(&model.id)
        .prompt("Write a haiku about the ocean.")
        .max_tokens(100)
        .provider("anthropic");

    let mut stream_result = stream(options, &client).expect("stream() should not fail upfront");

    let mut text_chunks: Vec<String> = Vec::new();
    while let Some(event_result) = stream_result.next().await {
        let event = event_result.expect("Stream event should not error");
        if event.event_type == StreamEventType::TextDelta {
            if let Some(delta) = &event.delta {
                text_chunks.push(delta.clone());
            }
        }
    }

    let concatenated = text_chunks.join("");
    assert!(!concatenated.is_empty(), "Should have received text deltas");

    if let Some(response) = stream_result.response() {
        let resp_text = response.text();
        assert_eq!(
            concatenated, resp_text,
            "Concatenated deltas should match accumulated response text"
        );
    }
}

/// DoD 8.10.3: Tool calling with parallel execution.
#[tokio::test]
async fn test_smoke_tool_calling() {
    use unified_llm::api::types::Tool;

    let client = require_api_keys();
    let model =
        get_latest_model("anthropic", Some("tools")).expect("No tool-capable Anthropic model");

    let weather_tool = Tool::active(
        "get_weather",
        "Get the current weather for a location",
        serde_json::json!({
            "type": "object",
            "properties": {
                "location": { "type": "string", "description": "City name" }
            },
            "required": ["location"]
        }),
        |args| {
            Box::pin(async move {
                let location = args["location"].as_str().unwrap_or("Unknown");
                Ok(serde_json::json!({
                    "temperature": "72F",
                    "conditions": "sunny",
                    "location": location
                }))
            })
        },
    );

    let options = GenerateOptions::new(&model.id)
        .prompt("What is the weather in San Francisco and New York?")
        .tools(vec![weather_tool])
        .max_tool_rounds(3)
        .max_tokens(500)
        .provider("anthropic");

    let result = unified_llm::generate(options, &client)
        .await
        .expect("generate() with tools should succeed");

    assert!(
        result.steps.len() >= 2,
        "Should have at least 2 steps (initial + after tool results), got {}",
        result.steps.len()
    );
    let text_lower = result.text.to_lowercase();
    assert!(
        text_lower.contains("san francisco"),
        "Should mention San Francisco"
    );
    assert!(text_lower.contains("new york"), "Should mention New York");
}

/// DoD 8.10.4: Image input.
#[tokio::test]
async fn test_smoke_image_input() {
    use unified_llm_types::{ContentPart, Message, Role};

    let client = require_api_keys();
    let model =
        get_latest_model("anthropic", Some("vision")).expect("No vision-capable Anthropic model");

    // Minimal 1x1 red PNG
    let png_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==";
    let png_bytes = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, png_base64)
        .expect("Valid base64");

    let messages = vec![Message {
        role: Role::User,
        content: vec![
            ContentPart::text("What color is this image? Answer in one word."),
            ContentPart::image_bytes(png_bytes, "image/png"),
        ],
        name: None,
        tool_call_id: None,
    }];

    let options = GenerateOptions::new(&model.id)
        .messages(messages)
        .max_tokens(50)
        .provider("anthropic");

    let result = unified_llm::generate(options, &client)
        .await
        .expect("generate() with image should succeed");

    assert!(
        !result.text.is_empty(),
        "Response text should not be empty for image input"
    );
}

/// DoD 8.10.5: Structured output (generate_object).
#[tokio::test]
async fn test_smoke_structured_output() {
    use unified_llm::generate_object;

    let client = require_api_keys();
    let model =
        get_latest_model("openai", Some("tools")).expect("No OpenAI model for structured output");

    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "name": { "type": "string" },
            "age": { "type": "integer" }
        },
        "required": ["name", "age"],
        "additionalProperties": false
    });

    let options = GenerateOptions::new(&model.id)
        .prompt("Extract the person's information: Alice is 30 years old.")
        .max_tokens(100)
        .provider("openai");

    let result = generate_object(options, schema, &client)
        .await
        .expect("generate_object() should succeed");

    let output = result.output.expect("output should be present");
    assert_eq!(output["name"], "Alice");
    assert_eq!(output["age"], 30);
}

/// DoD 8.10.6: Error handling.
#[tokio::test]
async fn test_smoke_error_handling() {
    use unified_llm_types::ErrorKind;

    let client = require_api_keys();

    let options = GenerateOptions::new("nonexistent-model-xyz")
        .prompt("test")
        .provider("openai")
        .max_retries(0);

    let result = unified_llm::generate(options, &client).await;

    assert!(
        result.is_err(),
        "Should have raised an error for nonexistent model"
    );
    let error = result.unwrap_err();
    assert_eq!(
        error.kind,
        ErrorKind::NotFound,
        "Error should be NotFoundError, got {:?}: {}",
        error.kind,
        error.message
    );
}
