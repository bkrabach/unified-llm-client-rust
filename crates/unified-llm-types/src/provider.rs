// ProviderAdapter trait — the contract every provider adapter must implement.

use std::future::Future;
use std::pin::Pin;

use futures_core::Stream;

use crate::error::Error;
use crate::request::Request;
use crate::response::Response;
use crate::stream::StreamEvent;

/// A boxed future that is Send.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// A boxed stream that is Send.
pub type BoxStream<'a, T> = Pin<Box<dyn Stream<Item = T> + Send + 'a>>;

/// The contract every provider adapter must implement.
///
/// Uses explicit BoxFuture/BoxStream return types instead of the `async-trait`
/// macro for two reasons:
/// 1. No hidden heap allocations from macro expansion
/// 2. Explicit control over lifetime bounds
pub trait ProviderAdapter: Send + Sync {
    /// Provider name (e.g., "openai", "anthropic", "gemini").
    fn name(&self) -> &str;

    /// Send a request, return the full response.
    fn complete(&self, request: Request) -> BoxFuture<'_, Result<Response, Error>>;

    /// Send a request, return a stream of events.
    fn stream(&self, request: Request) -> BoxStream<'_, Result<StreamEvent, Error>>;

    /// Release resources. Called by Client::close().
    fn close(&self) -> BoxFuture<'_, Result<(), Error>> {
        Box::pin(async { Ok(()) })
    }

    /// Validate configuration on startup. Called on registration.
    fn initialize(&self) -> BoxFuture<'_, Result<(), Error>> {
        Box::pin(async { Ok(()) })
    }

    /// Query whether this provider supports a specific tool choice mode.
    /// Default returns true for all modes. Override in adapters with caveats.
    fn supports_tool_choice(&self, _mode: &str) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Compile-time verification: a mock can implement the trait
    struct TestAdapter;

    impl ProviderAdapter for TestAdapter {
        fn name(&self) -> &str {
            "test"
        }

        fn complete(&self, _request: Request) -> BoxFuture<'_, Result<Response, Error>> {
            Box::pin(async { Err(Error::configuration("not implemented")) })
        }

        fn stream(&self, _request: Request) -> BoxStream<'_, Result<StreamEvent, Error>> {
            // Return an empty stream — use a custom stream that yields nothing
            Box::pin(EmptyStream)
        }
    }

    /// A stream that immediately returns Poll::Ready(None).
    struct EmptyStream;

    impl futures_core::Stream for EmptyStream {
        type Item = Result<StreamEvent, Error>;

        fn poll_next(
            self: std::pin::Pin<&mut Self>,
            _cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Option<Self::Item>> {
            std::task::Poll::Ready(None)
        }
    }

    #[test]
    fn test_provider_adapter_trait_object() {
        let adapter: Box<dyn ProviderAdapter> = Box::new(TestAdapter);
        assert_eq!(adapter.name(), "test");
    }

    #[tokio::test]
    async fn test_provider_adapter_default_close() {
        let adapter = TestAdapter;
        let result = adapter.close().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_provider_adapter_default_initialize() {
        let adapter = TestAdapter;
        let result = adapter.initialize().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_provider_adapter_complete_returns_error() {
        let adapter: Box<dyn ProviderAdapter> = Box::new(TestAdapter);
        let req = Request::default();
        let result = adapter.complete(req).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.kind, crate::error::ErrorKind::Configuration);
    }

    #[test]
    fn test_provider_adapter_default_supports_tool_choice() {
        let adapter = TestAdapter;
        // Default implementation returns true for all modes
        assert!(adapter.supports_tool_choice("auto"));
        assert!(adapter.supports_tool_choice("any"));
        assert!(adapter.supports_tool_choice("none"));
        assert!(adapter.supports_tool_choice("tool"));
    }
}
