// unified-llm: Layers 2–4 — client, providers, high-level API
#![allow(clippy::result_large_err)]

pub mod api;
pub mod catalog_data;
pub mod client;
pub mod default_client;
pub mod middleware;
pub mod providers;
#[cfg(any(test, feature = "testing"))]
pub mod testing;
pub mod util;

// --- Curated re-exports from unified-llm-types (Layer 1) ---
// We avoid `pub use unified_llm_types::*` to keep the public API surface
// intentional and prevent internal types from leaking to consumers.
pub use unified_llm_types::{
    AdapterTimeout,
    ArgumentValue,
    AudioData,
    // Type aliases
    BoxFuture,
    BoxStream,
    ContentKind,
    ContentPart,
    DocumentData,
    // Errors
    Error,
    ErrorKind,
    FinishReason,
    // Content data types
    ImageData,
    // Messages and content
    Message,
    // Catalog
    ModelInfo,
    // Provider trait
    ProviderAdapter,
    RateLimitInfo,
    // Request/Response
    Request,
    Response,
    ResponseFormat,
    // Config
    RetryPolicy,
    Role,
    // Streaming
    StreamEvent,
    StreamEventType,
    ThinkingData,
    TimeoutConfig,
    ToolCall,
    ToolCallData,
    ToolChoice,
    // Tools
    ToolDefinition,
    ToolResult,
    ToolResultData,
    Usage,
    Warning,
};

// --- Core client types at crate root ---
pub use client::{Client, ClientBuilder};

// --- High-level API types (Layer 4) ---
pub use api::generate::generate;
pub use api::generate::generate_with_default;
pub use api::generate_object::generate_object;
pub use api::generate_object::generate_object_with_default;
pub use api::generate_types::{GenerateResult, StepResult};
pub use api::stream::{stream, stream_with_default, StreamResult, TextDeltaStream};
pub use api::stream_object::{
    stream_object, stream_object_with_default, PartialObject, StreamObjectResult,
};
pub use api::types::{
    GenerateOptions, RepairToolCallFn, Tool, ToolContext, ToolExecuteFn, ToolExecuteWithContextFn,
};

// Catalog functions at crate root.
pub use catalog_data::{get_latest_model, get_model_info, list_models};

// Default client functions at crate root.
#[cfg(any(test, feature = "testing"))]
pub use default_client::reset_default_client;
pub use default_client::{get_default_client, set_default_client};

// Middleware trait at crate root.
pub use middleware::Middleware;

// Retry utility at crate root.
pub use util::retry::with_retry;

// Provider adapters at crate root (behind feature flags).
#[cfg(feature = "openai-compat")]
pub use providers::openai_compat::OpenAICompatibleAdapter;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_with_retry_reexported_at_crate_root() {
        // Verify with_retry is accessible at crate root by actually calling it.
        let policy = unified_llm_types::config::RetryPolicy::default();
        let result = with_retry(&policy, || async { Ok::<_, Error>(42) }).await;
        assert_eq!(result.unwrap(), 42);
    }

    /// H-1: Verify Client and ClientBuilder are importable from the crate root.
    #[test]
    fn test_client_importable_from_crate_root() {
        // These lines compile only if Client and ClientBuilder are re-exported.
        let _: fn() -> Result<Client, Error> = Client::from_env;
        let _: fn() -> ClientBuilder = Client::builder;
    }

    /// H-1: Verify curated re-exports cover the essential public types.
    #[test]
    fn test_curated_reexports_available() {
        // Message types
        let _ = Role::User;
        let _ = Message::user("test");

        // Error types
        let _ = ErrorKind::Configuration;

        // Config types
        let _ = AdapterTimeout::default();
        let _ = RetryPolicy::default();
        let _ = TimeoutConfig::default();

        // Stream event types
        let _ = StreamEventType::TextDelta;

        // Generate result types (moved from types crate to unified-llm)
        let _: fn(Vec<StepResult>) -> Result<GenerateResult, Error> = GenerateResult::from_steps;
    }
}
