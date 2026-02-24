//! Integration smoke tests with real API keys.
//!
//! Run with: cargo test -p unified-llm --test smoke_test -- --ignored --test-threads=1
//!
//! Requires: ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY (or GOOGLE_API_KEY)
//!
//! Spec reference: S8.10 Integration Smoke Test

use unified_llm::api::types::GenerateOptions;
use unified_llm::client::Client;

/// Cost-conscious models for smoke testing.
const ANTHROPIC_MODEL: &str = "claude-sonnet-4-20250514";
const OPENAI_MODEL: &str = "gpt-4o-mini";
const GEMINI_MODEL: &str = "gemini-2.0-flash";

fn require_api_keys() -> Client {
    Client::from_env().expect(
        "Integration tests require API keys: \
         ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY",
    )
}

/// Return (provider_name, model_id) tuples for all 3 providers.
fn all_providers() -> Vec<(&'static str, &'static str)> {
    vec![
        ("anthropic", ANTHROPIC_MODEL),
        ("openai", OPENAI_MODEL),
        ("gemini", GEMINI_MODEL),
    ]
}

/// DoD 8.10.1: Basic generation across all 3 providers.
#[tokio::test]
async fn test_smoke_basic_generation_all_providers() {
    let client = require_api_keys();

    for (provider_name, model) in all_providers() {
        let options = GenerateOptions::new(model)
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

/// DoD 8.10.2: Streaming across all 3 providers.
#[tokio::test]
async fn test_smoke_streaming() {
    use futures::StreamExt;
    use unified_llm::api::stream::stream;
    use unified_llm_types::StreamEventType;

    let client = require_api_keys();

    for (provider_name, model) in all_providers() {
        let options = GenerateOptions::new(model)
            .prompt("Write a haiku about the ocean.")
            .max_tokens(100)
            .provider(provider_name);

        let mut stream_result = stream(options, &client)
            .unwrap_or_else(|e| panic!("{provider_name}: stream() failed upfront: {e}"));

        let mut text_chunks: Vec<String> = Vec::new();
        while let Some(event_result) = stream_result.next().await {
            let event =
                event_result.unwrap_or_else(|e| panic!("{provider_name}: stream event error: {e}"));
            if event.event_type == StreamEventType::TextDelta {
                if let Some(delta) = &event.delta {
                    text_chunks.push(delta.clone());
                }
            }
        }

        let concatenated = text_chunks.join("");
        assert!(
            !concatenated.is_empty(),
            "{provider_name}: should have received text deltas"
        );

        if let Some(response) = stream_result.response() {
            let resp_text = response.text();
            assert_eq!(
                concatenated, resp_text,
                "{provider_name}: concatenated deltas should match accumulated response text"
            );
        }
    }
}

/// DoD 8.10.3: Tool calling with parallel execution across all 3 providers.
#[tokio::test]
async fn test_smoke_tool_calling() {
    use unified_llm::api::types::Tool;

    let client = require_api_keys();

    for (provider_name, model) in all_providers() {
        // Gemini rejects additionalProperties in tool schemas
        let schema = if provider_name == "gemini" {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "location": { "type": "string", "description": "City name" }
                },
                "required": ["location"]
            })
        } else {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "location": { "type": "string", "description": "City name" }
                },
                "required": ["location"],
                "additionalProperties": false
            })
        };

        let weather_tool = Tool::active(
            "get_weather",
            "Get the current weather for a location",
            schema,
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

        let options = GenerateOptions::new(model)
            .prompt("What is the weather in San Francisco and New York?")
            .tools(vec![weather_tool])
            .max_tool_rounds(3)
            .max_tokens(500)
            .provider(provider_name);

        let result = unified_llm::generate(options, &client)
            .await
            .unwrap_or_else(|e| panic!("{provider_name}: generate() with tools failed: {e}"));

        assert!(
            result.steps.len() >= 2,
            "{provider_name}: should have at least 2 steps (initial + after tool results), got {}",
            result.steps.len()
        );
        let text_lower = result.text.to_lowercase();
        assert!(
            text_lower.contains("san francisco")
                || text_lower.contains("new york")
                || text_lower.contains("72"),
            "{provider_name}: final text should mention weather/cities, got: {}",
            result.text
        );
    }
}

/// DoD 8.10.4: Image input across all 3 providers.
#[tokio::test]
async fn test_smoke_image_input() {
    use unified_llm_types::{ContentPart, Message, Role};

    let client = require_api_keys();

    // Minimal 1x1 red PNG
    let png_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==";
    let png_bytes = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, png_base64)
        .expect("Valid base64");

    for (provider_name, model) in all_providers() {
        let messages = vec![Message {
            role: Role::User,
            content: vec![
                ContentPart::text("What color is this image? Answer in one word."),
                ContentPart::image_bytes(png_bytes.clone(), "image/png"),
            ],
            name: None,
            tool_call_id: None,
        }];

        let options = GenerateOptions::new(model)
            .messages(messages)
            .max_tokens(50)
            .provider(provider_name);

        let result = unified_llm::generate(options, &client)
            .await
            .unwrap_or_else(|e| panic!("{provider_name}: generate() with image failed: {e}"));

        assert!(
            !result.text.is_empty(),
            "{provider_name}: response text should not be empty for image input"
        );
    }
}

/// DoD 8.10.5: Structured output (generate_object) across all 3 providers.
///
/// Provider-specific quirks:
/// - OpenAI: supports additionalProperties in schema
/// - Anthropic: uses tool-based structured output extraction (synthetic tool with forced tool_choice)
/// - Gemini: rejects additionalProperties in schemas
#[tokio::test]
async fn test_smoke_structured_output() {
    use unified_llm::generate_object;

    let client = require_api_keys();

    for (provider_name, model) in all_providers() {
        // Gemini and Anthropic do not support additionalProperties
        let schema = if provider_name == "openai" {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "name": { "type": "string" },
                    "age": { "type": "integer" }
                },
                "required": ["name", "age"],
                "additionalProperties": false
            })
        } else {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "name": { "type": "string" },
                    "age": { "type": "integer" }
                },
                "required": ["name", "age"]
            })
        };

        // Anthropic tool-based extraction may need more tokens
        let max_tokens = if provider_name == "anthropic" {
            200
        } else {
            100
        };

        let options = GenerateOptions::new(model)
            .prompt("Extract the person's information: Alice is 30 years old.")
            .max_tokens(max_tokens)
            .provider(provider_name);

        let result = generate_object(options, schema, &client)
            .await
            .unwrap_or_else(|e| panic!("{provider_name}: generate_object() failed: {e}"));

        let output = result
            .output
            .unwrap_or_else(|| panic!("{provider_name}: output should be present"));
        assert_eq!(
            output["name"], "Alice",
            "{provider_name}: name should be Alice"
        );
        assert_eq!(output["age"], 30, "{provider_name}: age should be 30");
    }
}

/// DoD 8.10.6: Error handling across all 3 providers.
#[tokio::test]
async fn test_smoke_error_handling() {
    let client = require_api_keys();

    for (provider_name, _model) in all_providers() {
        let options = GenerateOptions::new("nonexistent-model-xyz")
            .prompt("test")
            .provider(provider_name)
            .max_retries(0);

        let result = unified_llm::generate(options, &client).await;

        assert!(
            result.is_err(),
            "{provider_name}: should have raised an error for nonexistent model"
        );
        let error = result.unwrap_err();
        // All providers should return a structured error (not a panic)
        assert!(
            !error.message.is_empty(),
            "{provider_name}: error message should be non-empty, got {:?}",
            error.kind
        );
    }
}
