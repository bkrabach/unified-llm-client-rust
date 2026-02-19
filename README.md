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

```bash
# Unit and integration tests (no API keys required)
cargo test --all-features
```

The full test suite contains 817 unit and integration tests covering types,
serialization, provider adapters (via wiremock), middleware, retry logic, tool
execution, streaming, and structured output.

### Real-API Tests

These tests hit live provider APIs and require valid API keys:

```bash
# Smoke tests — 6 tests across all 3 providers
cargo test --test smoke_test -- --test-threads=1

# Compliance harness — 27 tests across Anthropic, OpenAI, and Gemini
cargo test --test compliance_harness -- --ignored --test-threads=1
```

---

## Certification

This implementation has been certified through adversarial review:

| Metric | Value |
|--------|-------|
| Review agents | 35 |
| Compliance checks | 750+ |
| Review rounds | 4 |
| DoD cells passing | 141 / 141 |
| Evidence standard | Three-part (code + test + behavioral) |
| Real-API compliance tests | 27 passing (Anthropic, OpenAI, Gemini) |
| Open spec violations | 0 |

Every cell in the 141-cell Definition of Done passes with three-part evidence:
source code demonstrating the behavior, a test exercising it, and a behavioral
argument linking the two to the spec requirement.

---

## License

See [LICENSE](LICENSE) for details.
