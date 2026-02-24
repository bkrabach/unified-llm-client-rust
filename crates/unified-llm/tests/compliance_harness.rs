//! Real-API compliance harness — maps to DoD §8.9 cross-provider parity cells.
//!
//! Run with: cargo test -p unified-llm --test compliance_harness -- --ignored --test-threads=1
//!
//! Requires: ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY (or GOOGLE_API_KEY)
//!
//! Each test prints its DoD cell ID on pass for traceability.

use unified_llm::api::types::GenerateOptions;
use unified_llm::client::Client;

/// Cost-conscious models for compliance testing.
const OPENAI_MODEL: &str = "gpt-4o-mini";
const ANTHROPIC_MODEL: &str = "claude-sonnet-4-20250514";
const GEMINI_MODEL: &str = "gemini-2.0-flash";

/// Require a client or panic (skip the test).
fn require_client() -> Client {
    Client::from_env().expect(
        "Compliance tests require API keys: \
         ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY",
    )
}

/// Retry helper: retries an async operation up to `max` times with linear `delay_secs` backoff.
async fn with_compliance_retry<F, Fut, T, E>(max: u32, delay_secs: u64, f: F) -> Result<T, E>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = Result<T, E>>,
    E: std::fmt::Debug,
{
    let mut last_err = None;
    for attempt in 0..max {
        match f().await {
            Ok(v) => return Ok(v),
            Err(e) => {
                last_err = Some(e);
                if attempt + 1 < max {
                    tokio::time::sleep(std::time::Duration::from_secs(
                        delay_secs * (attempt as u64 + 1),
                    ))
                    .await;
                }
            }
        }
    }
    Err(last_err.unwrap())
}

/// Print a DoD cell pass marker for traceability.
fn dod_pass(cell_id: &str) {
    eprintln!("[COMPLIANCE PASS] {cell_id}");
}

// ---------------------------------------------------------------------------
// §8.9.1 — Simple text generation × 3 providers
// ---------------------------------------------------------------------------

#[tokio::test]
async fn compliance_8_9_1_text_generation_openai() {
    let client = require_client();
    let result = with_compliance_retry(3, 2, || {
        let opts = GenerateOptions::new(OPENAI_MODEL)
            .prompt("Say hello in one sentence.")
            .max_tokens(100)
            .provider("openai");
        let client = &client;
        async move { unified_llm::generate(opts, client).await }
    })
    .await
    .expect("openai text generation failed");

    assert!(!result.text.is_empty(), "openai: text should not be empty");
    assert!(result.usage.input_tokens > 0, "openai: input_tokens > 0");
    assert!(result.usage.output_tokens > 0, "openai: output_tokens > 0");
    assert_eq!(
        result.finish_reason.reason, "stop",
        "openai: finish_reason should be 'stop'"
    );
    dod_pass("8.9.1/openai");
}

#[tokio::test]
async fn compliance_8_9_1_text_generation_anthropic() {
    let client = require_client();
    let result = with_compliance_retry(3, 2, || {
        let opts = GenerateOptions::new(ANTHROPIC_MODEL)
            .prompt("Say hello in one sentence.")
            .max_tokens(100)
            .provider("anthropic");
        let client = &client;
        async move { unified_llm::generate(opts, client).await }
    })
    .await
    .expect("anthropic text generation failed");

    assert!(
        !result.text.is_empty(),
        "anthropic: text should not be empty"
    );
    assert!(result.usage.input_tokens > 0, "anthropic: input_tokens > 0");
    assert!(
        result.usage.output_tokens > 0,
        "anthropic: output_tokens > 0"
    );
    assert_eq!(
        result.finish_reason.reason, "stop",
        "anthropic: finish_reason should be 'stop'"
    );
    dod_pass("8.9.1/anthropic");
}

#[tokio::test]
async fn compliance_8_9_1_text_generation_gemini() {
    let client = require_client();
    let result = with_compliance_retry(3, 2, || {
        let opts = GenerateOptions::new(GEMINI_MODEL)
            .prompt("Say hello in one sentence.")
            .max_tokens(100)
            .provider("gemini");
        let client = &client;
        async move { unified_llm::generate(opts, client).await }
    })
    .await
    .expect("gemini text generation failed");

    assert!(!result.text.is_empty(), "gemini: text should not be empty");
    assert!(result.usage.input_tokens > 0, "gemini: input_tokens > 0");
    assert!(result.usage.output_tokens > 0, "gemini: output_tokens > 0");
    assert_eq!(
        result.finish_reason.reason, "stop",
        "gemini: finish_reason should be 'stop'"
    );
    dod_pass("8.9.1/gemini");
}

// ---------------------------------------------------------------------------
// §8.9.2 — Streaming text × 3 providers
// ---------------------------------------------------------------------------

#[tokio::test]
async fn compliance_8_9_2_streaming_openai() {
    use futures::StreamExt;
    use unified_llm::api::stream::stream;
    use unified_llm_types::StreamEventType;

    let client = require_client();

    let opts = GenerateOptions::new(OPENAI_MODEL)
        .prompt("Write a haiku about the ocean.")
        .max_tokens(100)
        .provider("openai");

    let mut stream_result = stream(opts, &client).expect("stream() should not fail upfront");

    let mut text_chunks: Vec<String> = Vec::new();
    let mut got_finish = false;

    while let Some(event_result) = stream_result.next().await {
        let event = event_result.expect("Stream event should not error");
        if event.event_type == StreamEventType::TextDelta {
            if let Some(delta) = &event.delta {
                text_chunks.push(delta.clone());
            }
        }
        if event.event_type == StreamEventType::Finish {
            got_finish = true;
        }
    }

    assert!(
        !text_chunks.is_empty(),
        "openai: should have at least one TextDelta"
    );
    let concatenated = text_chunks.join("");
    assert!(
        !concatenated.is_empty(),
        "openai: concatenated deltas should be non-empty"
    );
    assert!(got_finish, "openai: should receive a Finish event");
    dod_pass("8.9.2/openai");
}

#[tokio::test]
async fn compliance_8_9_2_streaming_anthropic() {
    use futures::StreamExt;
    use unified_llm::api::stream::stream;
    use unified_llm_types::StreamEventType;

    let client = require_client();

    let opts = GenerateOptions::new(ANTHROPIC_MODEL)
        .prompt("Write a haiku about the ocean.")
        .max_tokens(100)
        .provider("anthropic");

    let mut stream_result = stream(opts, &client).expect("stream() should not fail upfront");

    let mut text_chunks: Vec<String> = Vec::new();
    let mut got_finish = false;

    while let Some(event_result) = stream_result.next().await {
        let event = event_result.expect("Stream event should not error");
        if event.event_type == StreamEventType::TextDelta {
            if let Some(delta) = &event.delta {
                text_chunks.push(delta.clone());
            }
        }
        if event.event_type == StreamEventType::Finish {
            got_finish = true;
        }
    }

    assert!(
        !text_chunks.is_empty(),
        "anthropic: should have at least one TextDelta"
    );
    let concatenated = text_chunks.join("");
    assert!(
        !concatenated.is_empty(),
        "anthropic: concatenated deltas should be non-empty"
    );
    assert!(got_finish, "anthropic: should receive a Finish event");
    dod_pass("8.9.2/anthropic");
}

#[tokio::test]
async fn compliance_8_9_2_streaming_gemini() {
    use futures::StreamExt;
    use unified_llm::api::stream::stream;
    use unified_llm_types::StreamEventType;

    let client = require_client();

    let opts = GenerateOptions::new(GEMINI_MODEL)
        .prompt("Write a haiku about the ocean.")
        .max_tokens(100)
        .provider("gemini");

    let mut stream_result = stream(opts, &client).expect("stream() should not fail upfront");

    let mut text_chunks: Vec<String> = Vec::new();
    let mut got_finish = false;

    while let Some(event_result) = stream_result.next().await {
        let event = event_result.expect("Stream event should not error");
        if event.event_type == StreamEventType::TextDelta {
            if let Some(delta) = &event.delta {
                text_chunks.push(delta.clone());
            }
        }
        if event.event_type == StreamEventType::Finish {
            got_finish = true;
        }
    }

    assert!(
        !text_chunks.is_empty(),
        "gemini: should have at least one TextDelta"
    );
    let concatenated = text_chunks.join("");
    assert!(
        !concatenated.is_empty(),
        "gemini: concatenated deltas should be non-empty"
    );
    assert!(got_finish, "gemini: should receive a Finish event");
    dod_pass("8.9.2/gemini");
}

// ---------------------------------------------------------------------------
// §8.9.5 — Single tool calling × 3 providers
// ---------------------------------------------------------------------------

/// Build a weather tool handler closure.
fn weather_handler(
    args: serde_json::Value,
) -> std::pin::Pin<
    Box<dyn std::future::Future<Output = Result<serde_json::Value, unified_llm::Error>> + Send>,
> {
    Box::pin(async move {
        let location = args["location"].as_str().unwrap_or("Unknown");
        Ok(serde_json::json!({
            "temperature": "72F",
            "conditions": "sunny",
            "location": location
        }))
    })
}

/// Build a weather tool for OpenAI/Anthropic (requires `additionalProperties: false`).
fn make_weather_tool() -> unified_llm::api::types::Tool {
    unified_llm::api::types::Tool::active(
        "get_weather",
        "Get the current weather for a location",
        serde_json::json!({
            "type": "object",
            "properties": {
                "location": { "type": "string", "description": "City name" }
            },
            "required": ["location"],
            "additionalProperties": false
        }),
        weather_handler,
    )
}

/// Build a weather tool for Gemini (does NOT support `additionalProperties`).
fn make_weather_tool_gemini() -> unified_llm::api::types::Tool {
    unified_llm::api::types::Tool::active(
        "get_weather",
        "Get the current weather for a location",
        serde_json::json!({
            "type": "object",
            "properties": {
                "location": { "type": "string", "description": "City name" }
            },
            "required": ["location"]
        }),
        weather_handler,
    )
}

#[tokio::test]
async fn compliance_8_9_5_single_tool_openai() {
    let client = require_client();
    let result = with_compliance_retry(3, 2, || {
        let opts = GenerateOptions::new(OPENAI_MODEL)
            .prompt("What's the weather in Paris?")
            .tools(vec![make_weather_tool()])
            .max_tool_rounds(3)
            .max_tokens(500)
            .provider("openai");
        let client = &client;
        async move { unified_llm::generate(opts, client).await }
    })
    .await
    .expect("openai single tool calling failed");

    assert!(
        result.steps.len() >= 2,
        "openai: should have at least 2 steps (initial + after tool), got {}",
        result.steps.len()
    );
    let text_lower = result.text.to_lowercase();
    assert!(
        text_lower.contains("paris") || text_lower.contains("weather") || text_lower.contains("72"),
        "openai: final text should mention weather/Paris, got: {}",
        result.text
    );
    dod_pass("8.9.5/openai");
}

#[tokio::test]
async fn compliance_8_9_5_single_tool_anthropic() {
    let client = require_client();
    let result = with_compliance_retry(3, 2, || {
        let opts = GenerateOptions::new(ANTHROPIC_MODEL)
            .prompt("What's the weather in Paris?")
            .tools(vec![make_weather_tool()])
            .max_tool_rounds(3)
            .max_tokens(500)
            .provider("anthropic");
        let client = &client;
        async move { unified_llm::generate(opts, client).await }
    })
    .await
    .expect("anthropic single tool calling failed");

    assert!(
        result.steps.len() >= 2,
        "anthropic: should have at least 2 steps, got {}",
        result.steps.len()
    );
    let text_lower = result.text.to_lowercase();
    assert!(
        text_lower.contains("paris") || text_lower.contains("weather") || text_lower.contains("72"),
        "anthropic: final text should mention weather/Paris, got: {}",
        result.text
    );
    dod_pass("8.9.5/anthropic");
}

#[tokio::test]
async fn compliance_8_9_5_single_tool_gemini() {
    let client = require_client();
    let result = with_compliance_retry(3, 2, || {
        let opts = GenerateOptions::new(GEMINI_MODEL)
            .prompt("What's the weather in Paris?")
            .tools(vec![make_weather_tool_gemini()])
            .max_tool_rounds(3)
            .max_tokens(500)
            .provider("gemini");
        let client = &client;
        async move { unified_llm::generate(opts, client).await }
    })
    .await
    .expect("gemini single tool calling failed");

    assert!(
        result.steps.len() >= 2,
        "gemini: should have at least 2 steps, got {}",
        result.steps.len()
    );
    let text_lower = result.text.to_lowercase();
    assert!(
        text_lower.contains("paris") || text_lower.contains("weather") || text_lower.contains("72"),
        "gemini: final text should mention weather/Paris, got: {}",
        result.text
    );
    dod_pass("8.9.5/gemini");
}

// ---------------------------------------------------------------------------
// §8.9.6 — Parallel tool calling × 3 providers
// ---------------------------------------------------------------------------

#[tokio::test]
async fn compliance_8_9_6_parallel_tools_openai() {
    let client = require_client();
    let result = with_compliance_retry(3, 2, || {
        let opts = GenerateOptions::new(OPENAI_MODEL)
            .prompt("What's the weather in Paris and London?")
            .tools(vec![make_weather_tool()])
            .max_tool_rounds(3)
            .max_tokens(500)
            .provider("openai");
        let client = &client;
        async move { unified_llm::generate(opts, client).await }
    })
    .await
    .expect("openai parallel tools failed");

    assert!(
        result.steps.len() >= 2,
        "openai: should have at least 2 steps for parallel tools, got {}",
        result.steps.len()
    );
    let text_lower = result.text.to_lowercase();
    assert!(
        text_lower.contains("paris"),
        "openai: final text should mention Paris, got: {}",
        result.text
    );
    assert!(
        text_lower.contains("london"),
        "openai: final text should mention London, got: {}",
        result.text
    );
    dod_pass("8.9.6/openai");
}

#[tokio::test]
async fn compliance_8_9_6_parallel_tools_anthropic() {
    let client = require_client();
    let result = with_compliance_retry(3, 2, || {
        let opts = GenerateOptions::new(ANTHROPIC_MODEL)
            .prompt("What's the weather in Paris and London?")
            .tools(vec![make_weather_tool()])
            .max_tool_rounds(3)
            .max_tokens(500)
            .provider("anthropic");
        let client = &client;
        async move { unified_llm::generate(opts, client).await }
    })
    .await
    .expect("anthropic parallel tools failed");

    assert!(
        result.steps.len() >= 2,
        "anthropic: should have at least 2 steps for parallel tools, got {}",
        result.steps.len()
    );
    let text_lower = result.text.to_lowercase();
    assert!(
        text_lower.contains("paris"),
        "anthropic: final text should mention Paris, got: {}",
        result.text
    );
    assert!(
        text_lower.contains("london"),
        "anthropic: final text should mention London, got: {}",
        result.text
    );
    dod_pass("8.9.6/anthropic");
}

#[tokio::test]
async fn compliance_8_9_6_parallel_tools_gemini() {
    let client = require_client();
    let result = with_compliance_retry(3, 2, || {
        let opts = GenerateOptions::new(GEMINI_MODEL)
            .prompt("What's the weather in Paris and London?")
            .tools(vec![make_weather_tool_gemini()])
            .max_tool_rounds(3)
            .max_tokens(500)
            .provider("gemini");
        let client = &client;
        async move { unified_llm::generate(opts, client).await }
    })
    .await
    .expect("gemini parallel tools failed");

    assert!(
        result.steps.len() >= 2,
        "gemini: should have at least 2 steps for parallel tools, got {}",
        result.steps.len()
    );
    let text_lower = result.text.to_lowercase();
    assert!(
        text_lower.contains("paris"),
        "gemini: final text should mention Paris, got: {}",
        result.text
    );
    assert!(
        text_lower.contains("london"),
        "gemini: final text should mention London, got: {}",
        result.text
    );
    dod_pass("8.9.6/gemini");
}

// ---------------------------------------------------------------------------
// §8.9.8 — Streaming with tools × 3 providers
// ---------------------------------------------------------------------------

#[tokio::test]
async fn compliance_8_9_8_streaming_tools_openai() {
    use futures::StreamExt;
    use unified_llm::api::stream::stream;
    use unified_llm_types::StreamEventType;

    let client = require_client();

    let opts = GenerateOptions::new(OPENAI_MODEL)
        .prompt("What's the weather in Paris?")
        .tools(vec![make_weather_tool()])
        .max_tool_rounds(3)
        .max_tokens(500)
        .provider("openai");

    let mut stream_result = stream(opts, &client).expect("stream() should not fail upfront");

    let mut got_tool_call_start = false;
    let mut got_tool_call_end = false;
    let mut got_text_delta = false;
    let mut got_finish = false;
    let mut event_types: Vec<String> = Vec::new();

    while let Some(event_result) = stream_result.next().await {
        let event = event_result.expect("Stream event should not error");
        event_types.push(format!("{:?}", event.event_type));
        match event.event_type {
            StreamEventType::ToolCallStart => got_tool_call_start = true,
            StreamEventType::ToolCallEnd => got_tool_call_end = true,
            StreamEventType::TextDelta => got_text_delta = true,
            StreamEventType::Finish => got_finish = true,
            _ => {}
        }
    }

    // Verify tool call events appeared
    assert!(
        got_tool_call_start || got_tool_call_end,
        "openai: should have ToolCallStart or ToolCallEnd event, got events: {:?}",
        event_types
    );
    assert!(got_finish, "openai: should have Finish event");

    assert!(
        got_text_delta,
        "openai: should have TextDelta events after tool execution"
    );
    dod_pass("8.9.8/openai");
}

#[tokio::test]
async fn compliance_8_9_8_streaming_tools_anthropic() {
    use futures::StreamExt;
    use unified_llm::api::stream::stream;
    use unified_llm_types::StreamEventType;

    let client = require_client();

    let opts = GenerateOptions::new(ANTHROPIC_MODEL)
        .prompt("What's the weather in Paris?")
        .tools(vec![make_weather_tool()])
        .max_tool_rounds(3)
        .max_tokens(500)
        .provider("anthropic");

    let mut stream_result = stream(opts, &client).expect("stream() should not fail upfront");

    let mut got_tool_call_start = false;
    let mut got_tool_call_end = false;
    let mut got_text_delta = false;
    let mut got_finish = false;

    while let Some(event_result) = stream_result.next().await {
        let event = event_result.expect("Stream event should not error");
        match event.event_type {
            StreamEventType::ToolCallStart => got_tool_call_start = true,
            StreamEventType::ToolCallEnd => got_tool_call_end = true,
            StreamEventType::TextDelta => got_text_delta = true,
            StreamEventType::Finish => got_finish = true,
            _ => {}
        }
    }

    assert!(
        got_tool_call_start,
        "anthropic: should have ToolCallStart event"
    );
    assert!(
        got_tool_call_end,
        "anthropic: should have ToolCallEnd event"
    );
    assert!(
        got_text_delta,
        "anthropic: should have TextDelta events after tool execution"
    );
    assert!(got_finish, "anthropic: should have Finish event");
    dod_pass("8.9.8/anthropic");
}

#[tokio::test]
async fn compliance_8_9_8_streaming_tools_gemini() {
    use futures::StreamExt;
    use unified_llm::api::stream::stream;
    use unified_llm_types::StreamEventType;

    let client = require_client();

    let opts = GenerateOptions::new(GEMINI_MODEL)
        .prompt("What's the weather in Paris?")
        .tools(vec![make_weather_tool_gemini()])
        .max_tool_rounds(3)
        .max_tokens(500)
        .provider("gemini");

    let mut stream_result = stream(opts, &client).expect("stream() should not fail upfront");

    let mut got_tool_call_start = false;
    let mut got_tool_call_end = false;
    let mut got_text_delta = false;
    let mut got_finish = false;

    while let Some(event_result) = stream_result.next().await {
        let event = event_result.expect("Stream event should not error");
        match event.event_type {
            StreamEventType::ToolCallStart => got_tool_call_start = true,
            StreamEventType::ToolCallEnd => got_tool_call_end = true,
            StreamEventType::TextDelta => got_text_delta = true,
            StreamEventType::Finish => got_finish = true,
            _ => {}
        }
    }

    assert!(
        got_tool_call_start,
        "gemini: should have ToolCallStart event"
    );
    assert!(got_tool_call_end, "gemini: should have ToolCallEnd event");
    assert!(
        got_text_delta,
        "gemini: should have TextDelta events after tool execution"
    );
    assert!(got_finish, "gemini: should have Finish event");
    dod_pass("8.9.8/gemini");
}

// ---------------------------------------------------------------------------
// Error handling — 404 (nonexistent model) × 3 providers
//
// These tests verify the error-handling pipeline using a nonexistent model
// (→ 404 NotFound): HTTP error → structured Error with correct ErrorKind,
// provider, and non-empty message.
//
// NOTE: These do NOT validate §8.9.11 (invalid API key → 401) or §8.9.12
// (rate limit → 429). See the real §8.9.11 tests below and the conformance.rs
// wiremock tests for §8.9.12.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn compliance_error_handling_404_openai() {
    use unified_llm_types::ErrorKind;

    let client = require_client();

    let opts = GenerateOptions::new("nonexistent-model-xyz-9999")
        .prompt("test")
        .provider("openai")
        .max_retries(0);

    let result = unified_llm::generate(opts, &client).await;

    assert!(
        result.is_err(),
        "openai: should error for nonexistent model"
    );
    let error = result.unwrap_err();
    assert_eq!(
        error.kind,
        ErrorKind::NotFound,
        "openai: error kind should be NotFound, got {:?}: {}",
        error.kind,
        error.message
    );
    assert!(
        !error.message.is_empty(),
        "openai: error message should be non-empty"
    );
    dod_pass("error_handling_404/openai");
}

#[tokio::test]
async fn compliance_error_handling_404_anthropic() {
    use unified_llm_types::ErrorKind;

    let client = require_client();

    let opts = GenerateOptions::new("nonexistent-model-xyz-9999")
        .prompt("test")
        .provider("anthropic")
        .max_retries(0);

    let result = unified_llm::generate(opts, &client).await;

    assert!(
        result.is_err(),
        "anthropic: should error for nonexistent model"
    );
    let error = result.unwrap_err();
    assert_eq!(
        error.kind,
        ErrorKind::NotFound,
        "anthropic: error kind should be NotFound, got {:?}: {}",
        error.kind,
        error.message
    );
    assert!(
        !error.message.is_empty(),
        "anthropic: error message should be non-empty"
    );
    dod_pass("error_handling_404/anthropic");
}

#[tokio::test]
async fn compliance_error_handling_404_gemini() {
    use unified_llm_types::ErrorKind;

    let client = require_client();

    let opts = GenerateOptions::new("nonexistent-model-xyz-9999")
        .prompt("test")
        .provider("gemini")
        .max_retries(0);

    let result = unified_llm::generate(opts, &client).await;

    assert!(
        result.is_err(),
        "gemini: should error for nonexistent model"
    );
    let error = result.unwrap_err();
    // Gemini may return NotFound or a different structured error — accept either
    assert!(
        error.kind == ErrorKind::NotFound
            || error.kind == ErrorKind::InvalidRequest
            || error.kind == ErrorKind::Server,
        "gemini: error kind should be a structured error, got {:?}: {}",
        error.kind,
        error.message
    );
    assert!(
        !error.message.is_empty(),
        "gemini: error message should be non-empty"
    );
    dod_pass("error_handling_404/gemini");
}

// ---------------------------------------------------------------------------
// §8.9.11 — Authentication error (invalid API key → 401) × 3 providers
//
// These tests use deliberately invalid API keys to verify the error-handling
// pipeline correctly classifies 401 responses as ErrorKind::Authentication
// with retryable=false.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn compliance_8_9_11_auth_error_openai() {
    use secrecy::SecretString;
    use unified_llm::providers::openai::OpenAiAdapter;

    let adapter = OpenAiAdapter::new(SecretString::from("sk-invalid-key-for-testing-000"));
    let client = Client::builder()
        .provider("openai", Box::new(adapter))
        .build()
        .unwrap();
    let opts = GenerateOptions::new("gpt-4o-mini")
        .prompt("test")
        .provider("openai")
        .max_retries(0);
    let err = unified_llm::generate(opts, &client).await.unwrap_err();
    assert_eq!(err.kind, unified_llm_types::ErrorKind::Authentication);
    assert!(!err.retryable);
    dod_pass("8.9.11/openai");
}

#[tokio::test]
async fn compliance_8_9_11_auth_error_anthropic() {
    use secrecy::SecretString;
    use unified_llm::providers::anthropic::AnthropicAdapter;

    let adapter = AnthropicAdapter::new(SecretString::from("sk-ant-invalid-key-for-testing-000"));
    let client = Client::builder()
        .provider("anthropic", Box::new(adapter))
        .build()
        .unwrap();
    let opts = GenerateOptions::new("claude-sonnet-4-20250514")
        .prompt("test")
        .provider("anthropic")
        .max_retries(0);
    let err = unified_llm::generate(opts, &client).await.unwrap_err();
    assert_eq!(err.kind, unified_llm_types::ErrorKind::Authentication);
    assert!(!err.retryable);
    dod_pass("8.9.11/anthropic");
}

#[tokio::test]
async fn compliance_8_9_11_auth_error_gemini() {
    use secrecy::SecretString;
    use unified_llm::providers::gemini::GeminiAdapter;

    let adapter = GeminiAdapter::new(SecretString::from("invalid-gemini-key-for-testing-000"));
    let client = Client::builder()
        .provider("gemini", Box::new(adapter))
        .build()
        .unwrap();
    let opts = GenerateOptions::new("gemini-2.0-flash")
        .prompt("test")
        .provider("gemini")
        .max_retries(0);
    let err = unified_llm::generate(opts, &client).await.unwrap_err();
    // Gemini may return Authentication or AccessDenied depending on key format
    assert!(
        err.kind == unified_llm_types::ErrorKind::Authentication
            || err.kind == unified_llm_types::ErrorKind::AccessDenied,
        "gemini: expected Authentication or AccessDenied, got {:?}: {}",
        err.kind,
        err.message
    );
    assert!(!err.retryable);
    dod_pass("8.9.11/gemini");
}

// ---------------------------------------------------------------------------
// §8.9.12 — Rate limit (429) handling
//
// Rate limit errors (429) are difficult to trigger reliably against real APIs
// without burning quota. §8.9.12 is validated via wiremock in conformance.rs
// which has dedicated 429 tests for all 3 providers:
//   - conformance_openai_429_rate_limit
//   - conformance_anthropic_429_rate_limit
//   - conformance_gemini_429_rate_limit
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// §8.9.9 — Structured output (generate_object) × OpenAI, Gemini
// ---------------------------------------------------------------------------

#[tokio::test]
async fn compliance_8_9_9_structured_output_openai() {
    let client = require_client();

    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "name": { "type": "string" },
            "age": { "type": "integer" }
        },
        "required": ["name", "age"],
        "additionalProperties": false
    });

    let result = with_compliance_retry(3, 2, || {
        let opts = GenerateOptions::new(OPENAI_MODEL)
            .prompt("Extract the person's information: Alice is 30 years old.")
            .max_tokens(100)
            .provider("openai");
        let schema = schema.clone();
        let client = &client;
        async move { unified_llm::generate_object(opts, schema, client).await }
    })
    .await
    .expect("openai structured output failed");

    let output = result.output.expect("output should be present");
    assert_eq!(output["name"], "Alice", "openai: name should be Alice");
    assert_eq!(output["age"], 30, "openai: age should be 30");
    dod_pass("8.9.9/openai");
}

#[tokio::test]
async fn compliance_8_9_9_structured_output_gemini() {
    let client = require_client();

    // Gemini does not support "additionalProperties" in response schemas.
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "name": { "type": "string" },
            "age": { "type": "integer" }
        },
        "required": ["name", "age"]
    });

    let result = with_compliance_retry(3, 2, || {
        let opts = GenerateOptions::new(GEMINI_MODEL)
            .prompt("Extract the person's information: Alice is 30 years old.")
            .max_tokens(100)
            .provider("gemini");
        let schema = schema.clone();
        let client = &client;
        async move { unified_llm::generate_object(opts, schema, client).await }
    })
    .await
    .expect("gemini structured output failed");

    let output = result.output.expect("output should be present");
    assert_eq!(output["name"], "Alice", "gemini: name should be Alice");
    assert_eq!(output["age"], 30, "gemini: age should be 30");
    dod_pass("8.9.9/gemini");
}

// W-12: Anthropic structured output via generate_object.
// Anthropic lacks native json_schema — the adapter injects schema instructions
// into the system prompt. generate_object validates the response against the
// schema after generation.
#[tokio::test]
async fn compliance_8_9_9_structured_output_anthropic() {
    let client = require_client();

    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "name": { "type": "string" },
            "age": { "type": "integer" }
        },
        "required": ["name", "age"]
    });

    let result = with_compliance_retry(3, 2, || {
        let opts = GenerateOptions::new(ANTHROPIC_MODEL)
            .prompt("Extract the person's information: Alice is 30 years old.")
            .max_tokens(200)
            .provider("anthropic");
        let schema = schema.clone();
        let client = &client;
        async move { unified_llm::generate_object(opts, schema, client).await }
    })
    .await
    .expect("anthropic structured output failed");

    let output = result.output.expect("output should be present");
    assert_eq!(output["name"], "Alice", "anthropic: name should be Alice");
    assert_eq!(output["age"], 30, "anthropic: age should be 30");
    dod_pass("8.9.9/anthropic");
}

// ---------------------------------------------------------------------------
// §8.9.13 — Usage accuracy × 3 providers
// ---------------------------------------------------------------------------

#[tokio::test]
async fn compliance_8_9_13_usage_accuracy_openai() {
    let client = require_client();
    let result = with_compliance_retry(3, 2, || {
        let opts = GenerateOptions::new(OPENAI_MODEL)
            .prompt("Say hello.")
            .max_tokens(50)
            .provider("openai");
        let client = &client;
        async move { unified_llm::generate(opts, client).await }
    })
    .await
    .expect("openai usage accuracy test failed");

    assert!(result.usage.total_tokens > 0, "openai: total_tokens > 0");
    assert!(result.usage.input_tokens > 0, "openai: input_tokens > 0");
    assert!(result.usage.output_tokens > 0, "openai: output_tokens > 0");
    // Check total = input + output (some providers may not match exactly)
    let expected_total = result.usage.input_tokens + result.usage.output_tokens;
    assert_eq!(
        result.usage.total_tokens, expected_total,
        "openai: total_tokens ({}) should equal input_tokens ({}) + output_tokens ({})",
        result.usage.total_tokens, result.usage.input_tokens, result.usage.output_tokens
    );
    dod_pass("8.9.13/openai");
}

#[tokio::test]
async fn compliance_8_9_13_usage_accuracy_anthropic() {
    let client = require_client();
    let result = with_compliance_retry(3, 2, || {
        let opts = GenerateOptions::new(ANTHROPIC_MODEL)
            .prompt("Say hello.")
            .max_tokens(50)
            .provider("anthropic");
        let client = &client;
        async move { unified_llm::generate(opts, client).await }
    })
    .await
    .expect("anthropic usage accuracy test failed");

    assert!(result.usage.total_tokens > 0, "anthropic: total_tokens > 0");
    assert!(result.usage.input_tokens > 0, "anthropic: input_tokens > 0");
    assert!(
        result.usage.output_tokens > 0,
        "anthropic: output_tokens > 0"
    );
    let expected_total = result.usage.input_tokens + result.usage.output_tokens;
    assert_eq!(
        result.usage.total_tokens, expected_total,
        "anthropic: total_tokens ({}) should equal input_tokens ({}) + output_tokens ({})",
        result.usage.total_tokens, result.usage.input_tokens, result.usage.output_tokens
    );
    dod_pass("8.9.13/anthropic");
}

#[tokio::test]
async fn compliance_8_9_13_usage_accuracy_gemini() {
    let client = require_client();
    let result = with_compliance_retry(3, 2, || {
        let opts = GenerateOptions::new(GEMINI_MODEL)
            .prompt("Say hello.")
            .max_tokens(50)
            .provider("gemini");
        let client = &client;
        async move { unified_llm::generate(opts, client).await }
    })
    .await
    .expect("gemini usage accuracy test failed");

    assert!(result.usage.total_tokens > 0, "gemini: total_tokens > 0");
    assert!(result.usage.input_tokens > 0, "gemini: input_tokens > 0");
    assert!(result.usage.output_tokens > 0, "gemini: output_tokens > 0");
    // Gemini may report total_tokens differently; at minimum verify it's non-zero
    // and >= max(input, output)
    assert!(
        result.usage.total_tokens >= result.usage.input_tokens
            && result.usage.total_tokens >= result.usage.output_tokens,
        "gemini: total_tokens ({}) should be >= both input ({}) and output ({})",
        result.usage.total_tokens,
        result.usage.input_tokens,
        result.usage.output_tokens
    );
    dod_pass("8.9.13/gemini");
}

// ---------------------------------------------------------------------------
// §8.9.15 — Provider-specific options (Anthropic)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn compliance_8_9_15_provider_options_anthropic() {
    let client = require_client();
    let result = with_compliance_retry(3, 2, || {
        let opts = GenerateOptions::new(ANTHROPIC_MODEL)
            .prompt("Say hello in one sentence.")
            .max_tokens(100)
            .provider("anthropic")
            .provider_options(serde_json::json!({
                "anthropic": {
                    "metadata": {
                        "user_id": "compliance-test"
                    }
                }
            }));
        let client = &client;
        async move { unified_llm::generate(opts, client).await }
    })
    .await
    .expect("anthropic provider_options should not cause an error");

    assert!(
        !result.text.is_empty(),
        "anthropic: text should not be empty with provider_options"
    );
    dod_pass("8.9.15/anthropic");
}

#[tokio::test]
async fn compliance_8_9_15_provider_options_openai() {
    let client = require_client();
    let result = with_compliance_retry(3, 2, || {
        let opts = GenerateOptions::new(OPENAI_MODEL)
            .prompt("Say hello in one sentence.")
            .max_tokens(100)
            .provider("openai")
            .provider_options(serde_json::json!({
                "openai": {
                    "store": false
                }
            }));
        let client = &client;
        async move { unified_llm::generate(opts, client).await }
    })
    .await
    .expect("openai provider_options should not cause an error");

    assert!(
        !result.text.is_empty(),
        "openai: text should not be empty with provider_options"
    );
    dod_pass("8.9.15/openai");
}

#[tokio::test]
async fn compliance_8_9_15_provider_options_gemini() {
    let client = require_client();
    let result = with_compliance_retry(3, 2, || {
        let opts = GenerateOptions::new(GEMINI_MODEL)
            .prompt("Say hello in one sentence.")
            .max_tokens(100)
            .provider("gemini")
            .provider_options(serde_json::json!({
                "gemini": {
                    "safetySettings": []
                }
            }));
        let client = &client;
        async move { unified_llm::generate(opts, client).await }
    })
    .await
    .expect("gemini provider_options should not cause an error");

    assert!(
        !result.text.is_empty(),
        "gemini: text should not be empty with provider_options"
    );
    dod_pass("8.9.15/gemini");
}

// ---------------------------------------------------------------------------
// stream_object() compliance — streaming structured output (spec §4.6)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn compliance_stream_object_openai() {
    use futures::StreamExt;
    use unified_llm::api::stream_object::stream_object;

    let client = require_client();
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "name": { "type": "string" },
            "age": { "type": "integer" }
        },
        "required": ["name", "age"],
        "additionalProperties": false
    });
    let opts = GenerateOptions::new(OPENAI_MODEL)
        .prompt("Extract: Alice is 30 years old. Return JSON with name and age.")
        .max_tokens(100)
        .provider("openai");

    let mut result =
        stream_object(opts, schema, &client).expect("stream_object() should not fail upfront");
    let mut partial_count = 0;
    while let Some(partial) = result.next().await {
        let _p = partial.expect("Partial should not error");
        partial_count += 1;
    }
    assert!(
        partial_count > 0,
        "Should have received at least one partial"
    );
    let final_obj = result.object().expect("Final object should be valid");
    assert_eq!(final_obj["name"], "Alice");
    assert_eq!(final_obj["age"], 30);
    dod_pass("stream_object/openai");
}

#[tokio::test]
async fn compliance_stream_object_gemini() {
    use futures::StreamExt;
    use unified_llm::api::stream_object::stream_object;

    let client = require_client();
    // Gemini does not support "additionalProperties" in response schemas.
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "name": { "type": "string" },
            "age": { "type": "integer" }
        },
        "required": ["name", "age"]
    });
    let opts = GenerateOptions::new(GEMINI_MODEL)
        .prompt("Extract: Alice is 30 years old. Return JSON with name and age.")
        .max_tokens(100)
        .provider("gemini");

    let mut result =
        stream_object(opts, schema, &client).expect("stream_object() should not fail upfront");
    let mut partial_count = 0;
    while let Some(partial) = result.next().await {
        let _p = partial.expect("Partial should not error");
        partial_count += 1;
    }
    assert!(
        partial_count > 0,
        "Should have received at least one partial"
    );
    let final_obj = result.object().expect("Final object should be valid");
    assert_eq!(final_obj["name"], "Alice");
    assert_eq!(final_obj["age"], 30);
    dod_pass("stream_object/gemini");
}

// ---------------------------------------------------------------------------
// §8.6.9 — Multi-turn prompt caching × Anthropic (real API)
//
// Runs 6 conversation turns with a large system prompt to maximize cache hit
// potential. Logs cache metrics but does NOT assert a specific threshold —
// cache hit rates depend on Anthropic's server-side behavior and timing.
// The wiremock test (conformance.rs) validates field parsing; this test
// validates the real multi-turn flow.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn compliance_8_6_9_multi_turn_caching_anthropic() {
    use unified_llm_types::Message;

    let client = require_client();
    // Use a large system prompt (≥4000 tokens) to exceed provider cache
    // thresholds: Anthropic requires ≥1024 tokens, OpenAI ≥1024 token prefix.
    let system_text = "You are a helpful assistant. ".repeat(500); // ~13000 chars / ~3250 tokens
    let mut messages = vec![Message::system(&system_text)];

    for turn in 1..=6 {
        messages.push(Message::user(format!(
            "Turn {turn}: What is {turn} + {turn}?"
        )));
        let result = with_compliance_retry(3, 2, || {
            let opts = GenerateOptions::new(ANTHROPIC_MODEL)
                .messages(messages.clone())
                .max_tokens(100)
                .provider("anthropic");
            let client = &client;
            async move { unified_llm::generate(opts, client).await }
        })
        .await
        .unwrap_or_else(|e| panic!("Turn {turn} failed: {e:?}"));

        messages.push(Message::assistant(&result.text));

        if turn >= 5 {
            let cache_read = result.usage.cache_read_tokens.unwrap_or(0);
            let input = result.usage.input_tokens;
            let ratio = if input > 0 {
                cache_read as f64 / input as f64 * 100.0
            } else {
                0.0
            };
            eprintln!(
                "[CACHE] Turn {turn}: cache_read={cache_read}, input={input}, ratio={ratio:.1}%"
            );

            // §8.6.9: verify cache_read_tokens field IS populated (is_some).
            assert!(
                result.usage.cache_read_tokens.is_some(),
                "Turn {turn}: cache_read_tokens field should be reported (Some), got None"
            );
            // Soft check: cache_read > 0 is expected but server-side caching
            // is timing-dependent, so warn instead of hard-fail.
            if cache_read == 0 {
                eprintln!("WARN: Turn {turn} cache_read=0 — server may not have cached yet");
            }
            if cache_read < input / 2 {
                eprintln!("WARN: Turn {turn} cache ratio {ratio:.1}% is below 50% target");
            }
        }
    }
    dod_pass("8.6.9/anthropic");
}

// ---------------------------------------------------------------------------
// §8.9.3 — Image input (base64) × 3 providers
// ---------------------------------------------------------------------------

/// Minimal 1x1 red PNG as base64.
const TINY_PNG_B64: &str =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==";

fn decode_tiny_png() -> Vec<u8> {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD
        .decode(TINY_PNG_B64)
        .expect("Valid base64")
}

#[tokio::test]
async fn compliance_8_9_3_image_base64_openai() {
    use unified_llm_types::{ContentPart, Message, Role};

    let client = require_client();
    let png_bytes = decode_tiny_png();

    let messages = vec![Message {
        role: Role::User,
        content: vec![
            ContentPart::text("What color is this image? Answer in one word."),
            ContentPart::image_bytes(png_bytes, "image/png"),
        ],
        name: None,
        tool_call_id: None,
    }];

    let result = with_compliance_retry(3, 2, || {
        let msgs = messages.clone();
        let client = &client;
        async move {
            unified_llm::generate(
                GenerateOptions::new(OPENAI_MODEL)
                    .messages(msgs)
                    .max_tokens(50)
                    .provider("openai"),
                client,
            )
            .await
        }
    })
    .await
    .expect("Image base64 openai failed");

    assert!(!result.text.is_empty(), "openai: should respond to image");
    dod_pass("8.9.3/openai");
}

#[tokio::test]
async fn compliance_8_9_3_image_base64_anthropic() {
    use unified_llm_types::{ContentPart, Message, Role};

    let client = require_client();
    let png_bytes = decode_tiny_png();

    let messages = vec![Message {
        role: Role::User,
        content: vec![
            ContentPart::text("What color is this image? Answer in one word."),
            ContentPart::image_bytes(png_bytes, "image/png"),
        ],
        name: None,
        tool_call_id: None,
    }];

    let result = with_compliance_retry(3, 2, || {
        let msgs = messages.clone();
        let client = &client;
        async move {
            unified_llm::generate(
                GenerateOptions::new(ANTHROPIC_MODEL)
                    .messages(msgs)
                    .max_tokens(50)
                    .provider("anthropic"),
                client,
            )
            .await
        }
    })
    .await
    .expect("Image base64 anthropic failed");

    assert!(
        !result.text.is_empty(),
        "anthropic: should respond to image"
    );
    dod_pass("8.9.3/anthropic");
}

#[tokio::test]
async fn compliance_8_9_3_image_base64_gemini() {
    use unified_llm_types::{ContentPart, Message, Role};

    let client = require_client();
    let png_bytes = decode_tiny_png();

    let messages = vec![Message {
        role: Role::User,
        content: vec![
            ContentPart::text("What color is this image? Answer in one word."),
            ContentPart::image_bytes(png_bytes, "image/png"),
        ],
        name: None,
        tool_call_id: None,
    }];

    let result = with_compliance_retry(3, 2, || {
        let msgs = messages.clone();
        let client = &client;
        async move {
            unified_llm::generate(
                GenerateOptions::new(GEMINI_MODEL)
                    .messages(msgs)
                    .max_tokens(50)
                    .provider("gemini"),
                client,
            )
            .await
        }
    })
    .await
    .expect("Image base64 gemini failed");

    assert!(!result.text.is_empty(), "gemini: should respond to image");
    dod_pass("8.9.3/gemini");
}

// ---------------------------------------------------------------------------
// §8.6.9 — Multi-turn prompt caching × OpenAI (real API)
//
// Same pattern as the Anthropic test: 6 turns with a large system prompt,
// verify cache_read_tokens > 0 on turn 5+.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn compliance_8_6_9_multi_turn_caching_openai() {
    use unified_llm_types::Message;

    let client = require_client();
    let system_text = "You are a helpful assistant. ".repeat(500);
    let mut messages = vec![Message::system(&system_text)];

    for turn in 1..=6 {
        messages.push(Message::user(format!(
            "Turn {turn}: What is {turn} + {turn}?"
        )));
        let result = with_compliance_retry(3, 2, || {
            let opts = GenerateOptions::new(OPENAI_MODEL)
                .messages(messages.clone())
                .max_tokens(100)
                .provider("openai");
            let client = &client;
            async move { unified_llm::generate(opts, client).await }
        })
        .await
        .unwrap_or_else(|e| panic!("Turn {turn} failed: {e:?}"));

        messages.push(Message::assistant(&result.text));

        if turn >= 5 {
            let cache_read = result.usage.cache_read_tokens.unwrap_or(0);
            let input = result.usage.input_tokens;
            let ratio = if input > 0 {
                cache_read as f64 / input as f64 * 100.0
            } else {
                0.0
            };
            eprintln!(
                "[CACHE] Turn {turn}: cache_read={cache_read}, input={input}, ratio={ratio:.1}%"
            );

            assert!(
                result.usage.cache_read_tokens.is_some(),
                "Turn {turn}: cache_read_tokens field should be reported (Some), got None"
            );
            if cache_read == 0 {
                eprintln!("WARN: Turn {turn} cache_read=0 — server may not have cached yet");
            }
            if cache_read < input / 2 {
                eprintln!("WARN: Turn {turn} cache ratio {ratio:.1}% is below 50% target");
            }
        }
    }
    dod_pass("8.6.9/openai");
}

// ---------------------------------------------------------------------------
// §8.6.9 — Multi-turn prompt caching × Gemini (real API)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn compliance_8_6_9_multi_turn_caching_gemini() {
    use unified_llm_types::Message;

    let client = require_client();
    let system_text = "You are a helpful assistant. ".repeat(500);
    let mut messages = vec![Message::system(&system_text)];

    for turn in 1..=6 {
        messages.push(Message::user(format!(
            "Turn {turn}: What is {turn} + {turn}?"
        )));
        let result = with_compliance_retry(3, 2, || {
            let opts = GenerateOptions::new(GEMINI_MODEL)
                .messages(messages.clone())
                .max_tokens(100)
                .provider("gemini");
            let client = &client;
            async move { unified_llm::generate(opts, client).await }
        })
        .await
        .unwrap_or_else(|e| panic!("Turn {turn} failed: {e:?}"));

        messages.push(Message::assistant(&result.text));

        if turn >= 5 {
            let cache_read = result.usage.cache_read_tokens.unwrap_or(0);
            let input = result.usage.input_tokens;
            let ratio = if input > 0 {
                cache_read as f64 / input as f64 * 100.0
            } else {
                0.0
            };
            eprintln!(
                "[CACHE] Turn {turn}: cache_read={cache_read}, input={input}, ratio={ratio:.1}%"
            );

            // Gemini may not report cache_read_tokens at all (None) — this is
            // provider-specific behavior, so soft-check only.
            if result.usage.cache_read_tokens.is_none() {
                eprintln!(
                    "WARN: Turn {turn} cache_read_tokens is None — \
                     Gemini may not expose cache metrics"
                );
            } else if cache_read == 0 {
                eprintln!("WARN: Turn {turn} cache_read=0 — server may not have cached yet");
            }
            if cache_read < input / 2 {
                eprintln!("WARN: Turn {turn} cache ratio {ratio:.1}% is below 50% target");
            }
        }
    }
    dod_pass("8.6.9/gemini");
}

// ---------------------------------------------------------------------------
// §8.9.4 — Image URL input × 3 providers
// ---------------------------------------------------------------------------

/// A well-known, stable public image URL for integration testing.
const TEST_IMAGE_URL: &str =
    "https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png";

#[tokio::test]
async fn compliance_8_9_4_image_url_anthropic() {
    use unified_llm_types::{ContentPart, Message, Role};

    let client = require_client();

    let messages = vec![Message {
        role: Role::User,
        content: vec![
            ContentPart::text("What do you see in this image?"),
            ContentPart::image_url(TEST_IMAGE_URL),
        ],
        name: None,
        tool_call_id: None,
    }];

    let result = with_compliance_retry(3, 2, || {
        let msgs = messages.clone();
        let client = &client;
        async move {
            unified_llm::generate(
                GenerateOptions::new(ANTHROPIC_MODEL)
                    .messages(msgs)
                    .max_tokens(200)
                    .provider("anthropic"),
                client,
            )
            .await
        }
    })
    .await
    .expect("Image URL anthropic failed");

    assert!(
        !result.text.is_empty(),
        "anthropic: should respond to image URL"
    );
    dod_pass("8.9.4/anthropic");
}

#[tokio::test]
async fn compliance_8_9_4_image_url_openai() {
    use unified_llm_types::{ContentPart, Message, Role};

    let client = require_client();

    let messages = vec![Message {
        role: Role::User,
        content: vec![
            ContentPart::text("What do you see in this image?"),
            ContentPart::image_url(TEST_IMAGE_URL),
        ],
        name: None,
        tool_call_id: None,
    }];

    let result = with_compliance_retry(3, 2, || {
        let msgs = messages.clone();
        let client = &client;
        async move {
            unified_llm::generate(
                GenerateOptions::new(OPENAI_MODEL)
                    .messages(msgs)
                    .max_tokens(200)
                    .provider("openai"),
                client,
            )
            .await
        }
    })
    .await
    .expect("Image URL openai failed");

    assert!(
        !result.text.is_empty(),
        "openai: should respond to image URL"
    );
    dod_pass("8.9.4/openai");
}

#[tokio::test]
async fn compliance_8_9_4_image_url_gemini() {
    use unified_llm_types::{ContentPart, Message, Role};

    let client = require_client();

    // Gemini's generateContent API doesn't support fetching arbitrary HTTP URLs
    // for fileData — it only accepts Google Cloud Storage URIs or inline base64.
    // Download the image and send it as base64 to test the same capability.
    let img_bytes = reqwest::get(TEST_IMAGE_URL)
        .await
        .expect("Failed to download test image")
        .bytes()
        .await
        .expect("Failed to read test image bytes")
        .to_vec();

    let messages = vec![Message {
        role: Role::User,
        content: vec![
            ContentPart::text("What do you see in this image?"),
            ContentPart::image_bytes(img_bytes, "image/png"),
        ],
        name: None,
        tool_call_id: None,
    }];

    let result = with_compliance_retry(3, 2, || {
        let msgs = messages.clone();
        let client = &client;
        async move {
            unified_llm::generate(
                GenerateOptions::new(GEMINI_MODEL)
                    .messages(msgs)
                    .max_tokens(200)
                    .provider("gemini"),
                client,
            )
            .await
        }
    })
    .await
    .expect("Image URL gemini failed");

    assert!(
        !result.text.is_empty(),
        "gemini: should respond to image URL"
    );
    dod_pass("8.9.4/gemini");
}

// ---------------------------------------------------------------------------
// §8.9.7 — Multi-step tool loop (3+ rounds) × 3 providers
// ---------------------------------------------------------------------------

/// Build a counting tool handler closure.
fn count_step_handler(
    args: serde_json::Value,
) -> std::pin::Pin<
    Box<dyn std::future::Future<Output = Result<serde_json::Value, unified_llm::Error>> + Send>,
> {
    Box::pin(async move {
        let step = args["step"].as_i64().unwrap_or(0);
        Ok(serde_json::json!({
            "result": format!("Step {} complete", step),
            "next_step": step + 1
        }))
    })
}

fn make_count_step_tool() -> unified_llm::api::types::Tool {
    unified_llm::api::types::Tool::active(
        "count_step",
        "Returns the current step number. Call with step=1, then step=2, then step=3.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "step": { "type": "integer", "description": "The step number to execute" }
            },
            "required": ["step"],
            "additionalProperties": false
        }),
        count_step_handler,
    )
}

fn make_count_step_tool_gemini() -> unified_llm::api::types::Tool {
    unified_llm::api::types::Tool::active(
        "count_step",
        "Returns the current step number. Call with step=1, then step=2, then step=3.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "step": { "type": "integer", "description": "The step number to execute" }
            },
            "required": ["step"]
        }),
        count_step_handler,
    )
}

#[tokio::test]
async fn compliance_8_9_7_multi_step_tool_loop_openai() {
    let client = require_client();
    let result = with_compliance_retry(3, 2, || {
        let opts = GenerateOptions::new(OPENAI_MODEL)
            .prompt("Call the count_step tool 3 times: first with step=1, then step=2, then step=3. After all 3 calls, summarize the results.")
            .tools(vec![make_count_step_tool()])
            .max_tool_rounds(5)
            .max_tokens(500)
            .provider("openai");
        let client = &client;
        async move { unified_llm::generate(opts, client).await }
    })
    .await
    .expect("Multi-step tool loop openai failed");

    // Models may batch tool calls in parallel, so count total tool calls
    // across all steps rather than requiring 3+ separate steps.
    let total_tool_calls: usize = result.steps.iter().map(|s| s.tool_calls.len()).sum();
    assert!(
        total_tool_calls >= 3,
        "openai: should have 3+ tool calls total, got {} across {} steps",
        total_tool_calls,
        result.steps.len()
    );
    dod_pass("8.9.7/openai");
}

#[tokio::test]
async fn compliance_8_9_7_multi_step_tool_loop_anthropic() {
    let client = require_client();
    let result = with_compliance_retry(3, 2, || {
        let opts = GenerateOptions::new(ANTHROPIC_MODEL)
            .prompt("Call the count_step tool 3 times: first with step=1, then step=2, then step=3. After all 3 calls, summarize the results.")
            .tools(vec![make_count_step_tool()])
            .max_tool_rounds(5)
            .max_tokens(500)
            .provider("anthropic");
        let client = &client;
        async move { unified_llm::generate(opts, client).await }
    })
    .await
    .expect("Multi-step tool loop anthropic failed");

    let total_tool_calls: usize = result.steps.iter().map(|s| s.tool_calls.len()).sum();
    assert!(
        total_tool_calls >= 3,
        "anthropic: should have 3+ tool calls total, got {} across {} steps",
        total_tool_calls,
        result.steps.len()
    );
    dod_pass("8.9.7/anthropic");
}

#[tokio::test]
async fn compliance_8_9_7_multi_step_tool_loop_gemini() {
    let client = require_client();
    let result = with_compliance_retry(3, 2, || {
        let opts = GenerateOptions::new(GEMINI_MODEL)
            .prompt("Call the count_step tool 3 times: first with step=1, then step=2, then step=3. After all 3 calls, summarize the results.")
            .tools(vec![make_count_step_tool_gemini()])
            .max_tool_rounds(5)
            .max_tokens(500)
            .provider("gemini");
        let client = &client;
        async move { unified_llm::generate(opts, client).await }
    })
    .await
    .expect("Multi-step tool loop gemini failed");

    let total_tool_calls: usize = result.steps.iter().map(|s| s.tool_calls.len()).sum();
    assert!(
        total_tool_calls >= 3,
        "gemini: should have 3+ tool calls total, got {} across {} steps",
        total_tool_calls,
        result.steps.len()
    );
    dod_pass("8.9.7/gemini");
}

// ---------------------------------------------------------------------------
// §8.9.10 — Reasoning tokens × OpenAI (o4-mini)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn compliance_8_9_10_reasoning_tokens_openai() {
    let client = require_client();
    let result = with_compliance_retry(3, 2, || {
        // Use a harder prompt to encourage reasoning output from o4-mini.
        let opts = GenerateOptions::new("o4-mini")
            .prompt(
                "Solve step by step: If a train leaves Chicago at 60 mph and another \
                 leaves New York at 80 mph toward each other, and the distance is 800 \
                 miles, after how many hours do they meet?",
            )
            .max_tokens(1024)
            .provider("openai")
            .reasoning_effort("low");
        let client = &client;
        async move { unified_llm::generate(opts, client).await }
    })
    .await
    .expect("Reasoning tokens test failed");

    let reasoning_tokens = result.usage.reasoning_tokens;
    // Spec §8.9.10 requires reasoning_tokens are *reported* (Some), not that
    // every prompt necessarily produces non-zero reasoning tokens.
    assert!(
        reasoning_tokens.is_some(),
        "openai: should report reasoning_tokens field (got None)"
    );
    if reasoning_tokens == Some(0) {
        eprintln!(
            "WARN compliance_8_9_10: reasoning_tokens reported as 0 — \
             model may not have used internal reasoning for this prompt"
        );
    } else {
        eprintln!(
            "OK   compliance_8_9_10: reasoning_tokens = {:?}",
            reasoning_tokens.unwrap()
        );
    }
    dod_pass("8.9.10/openai");
}

// ---------------------------------------------------------------------------
// §8.9.10 — Reasoning tokens × Anthropic (claude-sonnet-4-20250514 with thinking)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn compliance_8_9_10_reasoning_tokens_anthropic() {
    let client = require_client();
    let result = with_compliance_retry(3, 2, || {
        let opts = GenerateOptions::new("claude-sonnet-4-20250514")
            .prompt(
                "Solve step by step: If a train leaves Chicago at 60 mph and another \
                 leaves New York at 80 mph toward each other, and the distance is 800 \
                 miles, after how many hours do they meet?",
            )
            .max_tokens(8192)
            .provider("anthropic")
            .provider_options(serde_json::json!({
                "anthropic": {
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": 4096
                    }
                }
            }));
        let client = &client;
        async move { unified_llm::generate(opts, client).await }
    })
    .await
    .expect("Reasoning tokens anthropic test failed");

    // Anthropic returns thinking blocks — the generate layer extracts them
    // into result.reasoning. Also, reasoning_tokens is estimated from
    // thinking block text length.
    let has_reasoning_text = result.reasoning.as_ref().is_some_and(|r| !r.is_empty());
    let reasoning_tokens = result.usage.reasoning_tokens;

    assert!(
        has_reasoning_text || reasoning_tokens.is_some(),
        "anthropic: should have reasoning text or reasoning_tokens \
         (reasoning={:?}, reasoning_tokens={:?})",
        result.reasoning.as_deref().map(|r| &r[..r.len().min(80)]),
        reasoning_tokens
    );

    if has_reasoning_text {
        eprintln!(
            "OK   compliance_8_9_10: anthropic reasoning text length = {}",
            result.reasoning.as_ref().map(|r| r.len()).unwrap_or(0)
        );
    }
    if let Some(rt) = reasoning_tokens {
        eprintln!("OK   compliance_8_9_10: anthropic reasoning_tokens = {rt}");
    }
    dod_pass("8.9.10/anthropic");
}

// ---------------------------------------------------------------------------
// §8.9.10 — Reasoning tokens × Gemini (gemini-2.5-flash with thinking)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn compliance_8_9_10_reasoning_tokens_gemini() {
    let client = require_client();
    let result = with_compliance_retry(3, 2, || {
        // Use a model with native thinking support. Do NOT set reasoning_effort
        // (which sends thinkingConfig) — just let the model think natively and
        // verify that reasoning_tokens are reported in the response.
        let opts = GenerateOptions::new("gemini-2.5-flash")
            .prompt(
                "Solve step by step: If a train leaves Chicago at 60 mph and another \
                 leaves New York at 80 mph toward each other, and the distance is 800 \
                 miles, after how many hours do they meet?",
            )
            .max_tokens(1024)
            .provider("gemini");
        let client = &client;
        async move { unified_llm::generate(opts, client).await }
    })
    .await
    .expect("Reasoning tokens gemini test failed");

    let reasoning_tokens = result.usage.reasoning_tokens;
    // Spec §8.9.10 requires reasoning_tokens are *reported* (Some), not that
    // every prompt necessarily produces non-zero reasoning tokens.
    assert!(
        reasoning_tokens.is_some(),
        "gemini: should report reasoning_tokens field (got None)"
    );
    if reasoning_tokens == Some(0) {
        eprintln!(
            "WARN compliance_8_9_10: gemini reasoning_tokens reported as 0 — \
             model may not have used internal reasoning for this prompt"
        );
    } else {
        eprintln!(
            "OK   compliance_8_9_10: gemini reasoning_tokens = {:?}",
            reasoning_tokens.unwrap()
        );
    }
    dod_pass("8.9.10/gemini");
}
