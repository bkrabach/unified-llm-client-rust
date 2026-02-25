# unified-llm-client-rust

A unified LLM client library in Rust. Provides a single async interface across
OpenAI, Anthropic, Gemini, and OpenAI-compatible endpoints. Built against the
[unified-llm-spec.md](https://github.com/strongdm/attractor/blob/main/unified-llm-spec.md)
natural language specification from [strongdm/attractor](https://github.com/strongdm/attractor).

---

## Specification

This project implements the **Unified LLM Client** natural language specification
(nlspec) from [strongdm/attractor](https://github.com/strongdm/attractor):

> **[unified-llm-spec.md](https://github.com/strongdm/attractor/blob/main/unified-llm-spec.md)**

The spec defines a provider-agnostic interface for LLM interaction organized as a
4-layer architecture (types, adapter, client, high-level API) with a 141-cell
Definition of Done covering every behavioral requirement. It prescribes unified
request/response models, a middleware chain, multi-step tool execution loops,
structured output with schema validation, streaming lifecycle events, retry
policies, and cancellation semantics.

---

## Architecture

The repository is a Cargo workspace with two crates:

```
unified-llm-client-rust/
├── Cargo.toml                     # Workspace root
└── crates/
    ├── unified-llm-types/         # Layer 1 — shared types, traits, error hierarchy
    └── unified-llm/               # Layers 2-4 — client, middleware, providers, API
```

| Crate | Role | Dependencies |
|-------|------|--------------|
| `unified-llm-types` | Shared types, the `ProviderAdapter` trait, error hierarchy. Zero business logic. Stability contract for adapter authors. | `serde`, `serde_json`, `futures-core` |
| `unified-llm` | Client routing, middleware chain, provider adapters, retry engine, tool execution loops, and the four high-level API functions. | `tokio`, `reqwest`, `serde`, `jsonschema`, `tracing`, and others |

### Provider Adapters

Each adapter is behind a feature flag. The three primary providers are enabled by default.

| Adapter | Feature Flag | Default | API |
|---------|-------------|---------|-----|
| Anthropic | `anthropic` | Yes | Messages API |
| OpenAI | `openai` | Yes | Responses API |
| Gemini | `gemini` | Yes | Native Gemini API |
| OpenAI-Compatible | `openai-compat` | No | Chat Completions API |

---

## Features

- **`generate()` / `stream()`** -- text generation with automatic provider routing
- **`generate_object()` / `stream_object()`** -- structured output with JSON schema validation
- **Multi-step tool calling** with parallel execution, active/passive tools, and configurable round limits
- **Middleware chain** (onion model) for request/response interception
- **Exponential backoff retry** with jitter, `Retry-After` header support, and per-step budgets
- **Anthropic auto-caching** -- 3-breakpoint cache control injection with opt-out
- **Streaming** with start/delta/end lifecycle events and thinking block preservation
- **Cancellation** via `CancellationToken` (abort signal)
- **Default client** with `_with_default()` convenience functions (`generate_with_default`, etc.)
- **Embedded model catalog** with `get_latest_model()`, `get_model_info()`, `list_models()`

---

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
unified-llm = "0.1"
tokio = { version = "1", features = ["full"] }
```

### Text Generation

```rust
use unified_llm::{Client, generate, GenerateOptions};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::from_env()?;

    let result = generate(
        GenerateOptions::new("gpt-4o-mini")
            .prompt("Say hello in one sentence.")
            .provider("openai"),
        &client,
    ).await?;

    println!("{}", result.text);
    Ok(())
}
```

### Streaming

```rust
use unified_llm::{Client, stream, GenerateOptions, StreamEventType};
use futures::StreamExt;

let client = Client::from_env()?;

let stream_result = stream(
    GenerateOptions::new("claude-sonnet-4-20250514")
        .prompt("Write a haiku about Rust.")
        .provider("anthropic"),
    &client,
).await?;

let mut stream = stream_result.text_stream;
while let Some(event) = stream.next().await {
    match event {
        Ok(delta) => print!("{delta}"),
        Err(e) => eprintln!("Stream error: {e}"),
    }
}
println!();
```

### Tool Calling

```rust
use unified_llm::{Client, generate, GenerateOptions, Tool};
use serde_json::json;

let weather_tool = Tool::active(
    "get_weather",
    "Get the current weather for a city.",
    json!({
        "type": "object",
        "properties": {
            "city": { "type": "string" }
        },
        "required": ["city"]
    }),
    |args, _ctx| Box::pin(async move {
        let city = args["city"].as_str().unwrap_or("unknown");
        Ok(json!({ "temp": "72F", "city": city }).to_string())
    }),
);

let result = generate(
    GenerateOptions::new("gpt-4o-mini")
        .prompt("What is the weather in Seattle?")
        .tools(vec![weather_tool])
        .provider("openai"),
    &client,
).await?;

println!("{}", result.text);
```

### Structured Output

```rust
use unified_llm::{Client, generate_object, GenerateOptions};
use serde_json::json;

let schema = json!({
    "type": "object",
    "properties": {
        "name":  { "type": "string" },
        "age":   { "type": "integer" },
        "email": { "type": "string", "format": "email" }
    },
    "required": ["name", "age", "email"]
});

let result = generate_object(
    GenerateOptions::new("gpt-4o-mini")
        .prompt("Generate a sample user profile.")
        .provider("openai"),
    schema,
    &client,
).await?;

println!("{}", serde_json::to_string_pretty(&result.output)?);
```

---

## Environment Variables

`Client::from_env()` reads the following variables to auto-register providers:

| Variable | Required For | Description |
|----------|-------------|-------------|
| `OPENAI_API_KEY` | OpenAI | API key for OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic | API key for Anthropic |
| `GEMINI_API_KEY` | Gemini | API key for Gemini (primary) |
| `GOOGLE_API_KEY` | Gemini | API key for Gemini (fallback) |
| `OPENAI_BASE_URL` | -- | Override OpenAI endpoint |
| `ANTHROPIC_BASE_URL` | -- | Override Anthropic endpoint |
| `GEMINI_BASE_URL` | -- | Override Gemini endpoint |
| `OPENAI_ORG_ID` | -- | OpenAI organization ID |
| `OPENAI_PROJECT_ID` | -- | OpenAI project ID |

Providers are registered only when their API key is present. If no key is set for
a provider, that provider is silently skipped.

---

## Testing

The test suite runs entirely without `#[ignore]` — every test executes on every run,
including 54 tests that call real provider APIs.

```bash
# Full suite (requires OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY)
cargo test --all-features -- --test-threads=1

# Offline-only (no API keys needed — wiremock mocks)
cargo test --all-features --lib --test conformance --test multimodal
```

| Category | Tests | Infrastructure |
|----------|-------|----------------|
| Types crate unit tests | ~200 | Inline `#[cfg(test)]` |
| Provider adapter unit tests | ~250 | Wiremock |
| API layer unit tests | ~110 | MockProvider |
| Cross-provider conformance | 45 | Wiremock × 3 providers |
| Multimodal conformance | 22 | Wiremock × 3 providers |
| Live API compliance | 48 | Real OpenAI, Anthropic, Gemini |
| Live API smoke tests | 6 | Real APIs, all 3 providers |
| Doc-tests | 3 | Compile-checked |
| **Total** | **~925** | **0 ignored, 0 stubs** |

### Examples

Seven runnable examples, all tested against live APIs:

```bash
cargo run --example basic_generate      # Text generation across all 3 providers
cargo run --example streaming           # Real-time streaming output
cargo run --example tool_calling        # Active tool with auto-execution loop
cargo run --example structured_output   # JSON schema-validated output
cargo run --example multi_turn          # Low-level Client.complete() conversation
cargo run --example provider_fallback   # Error-based provider switching
cargo run --example multi_provider      # Same prompt across all 3 providers
```

---

## Spec Conformance

This implementation has been certified as an **exemplar** of the
[unified-llm-spec.md](https://github.com/strongdm/attractor/blob/main/unified-llm-spec.md)
through adversarial multi-agent review.

| Metric | Value |
|--------|-------|
| DoD cells passing | **141 / 141** |
| §1–§7 behavioral checks | **79 / 79** |
| Review rounds | 12 |
| Independent audit agents | 100+ |
| Evidence standard | Line-number code + test + behavioral |
| Live API tests | 54 (all 3 providers) |
| Open spec violations | **0** |
| Spec bugs found | 8 (implementation correct, spec tables wrong) |

The audit identified 8 places where the spec's reference tables use Chat
Completions API field names that don't match the mandated Responses API. In all
cases, this implementation correctly targets the actual API.

### Documented Rust-Idiomatic Deviations

These are additive extensions that never remove spec-required functionality:

| Deviation | Rationale |
|-----------|-----------|
| Flat `ErrorKind` enum + `is_provider_error()` | Idiomatic Rust; functionally equivalent to class hierarchy |
| `generate()` / `generate_with_default()` split | Rust ownership semantics; documented |
| Anthropic `reasoning_tokens` estimated (~chars/4) | API limitation; warning emitted on every such response |
| `StepFinish` + `Unknown(String)` on StreamEventType | Forward-compatible extensions for tool loops and new events |
| `ThinkingData.data` field | Preserves Anthropic redacted thinking opaque payload |

---

## License

See [LICENSE](LICENSE) for details.
