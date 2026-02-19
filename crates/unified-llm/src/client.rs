// Client and ClientBuilder — Layer 3: Core Client with provider routing.

use std::collections::HashMap;
use std::sync::Arc;

use unified_llm_types::{
    AdapterTimeout, BoxFuture, BoxStream, Error, ProviderAdapter, Request, Response, StreamEvent,
};

use crate::middleware::{Middleware, Next};

/// The core client holding registered providers and routing requests.
pub struct Client {
    providers: HashMap<String, Box<dyn ProviderAdapter>>,
    default_provider: Option<String>,
    middleware: Vec<Arc<dyn Middleware>>,
}

/// Builder for constructing a Client with providers.
pub struct ClientBuilder {
    providers: HashMap<String, Box<dyn ProviderAdapter>>,
    default_provider: Option<String>,
    middleware: Vec<Arc<dyn Middleware>>,
}

impl ClientBuilder {
    pub fn new() -> Self {
        Self {
            providers: HashMap::new(),
            default_provider: None,
            middleware: Vec::new(),
        }
    }

    /// Register a provider adapter under the given name.
    /// The first registered provider becomes the default automatically.
    pub fn provider(mut self, name: &str, adapter: Box<dyn ProviderAdapter>) -> Self {
        if self.providers.is_empty() && self.default_provider.is_none() {
            self.default_provider = Some(name.to_string());
        }
        self.providers.insert(name.to_string(), adapter);
        self
    }

    /// Set the default provider by name.
    pub fn default_provider(mut self, name: &str) -> Self {
        self.default_provider = Some(name.to_string());
        self
    }

    /// Register a middleware. Middleware executes in registration order for the
    /// request phase and reverse order for the response phase (onion pattern).
    pub fn middleware(mut self, mw: Arc<dyn Middleware>) -> Self {
        self.middleware.push(mw);
        self
    }

    /// Build the Client. Returns ConfigurationError if no providers are registered.
    pub fn build(self) -> Result<Client, Error> {
        if self.providers.is_empty() {
            return Err(Error::configuration("No providers configured"));
        }
        Ok(Client {
            providers: self.providers,
            default_provider: self.default_provider,
            middleware: self.middleware,
        })
    }
}

impl Default for ClientBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Client {
    /// Create a new ClientBuilder.
    pub fn builder() -> ClientBuilder {
        ClientBuilder::new()
    }

    /// Resolve which provider to use for this request.
    fn resolve_provider(&self, request: &Request) -> Result<&dyn ProviderAdapter, Error> {
        let provider_name = request
            .provider
            .as_deref()
            .or(self.default_provider.as_deref());

        match provider_name {
            Some(name) => self
                .providers
                .get(name)
                .map(|p| p.as_ref())
                .ok_or_else(|| Error::configuration(format!("Unknown provider: {name}"))),
            None => Err(Error::configuration(
                "No provider specified and no default provider configured",
            )),
        }
    }

    /// Auto-register providers whose API keys are found in environment variables.
    ///
    /// **Registration order** (first registered becomes default):
    /// 1. `ANTHROPIC_API_KEY` → Anthropic adapter
    /// 2. `OPENAI_API_KEY` → OpenAI adapter
    /// 3. `GEMINI_API_KEY` (fallback: `GOOGLE_API_KEY`) → Gemini adapter
    ///
    /// **Timeout configuration** (optional, shared across all adapters):
    /// - `UNIFIED_LLM_CONNECT_TIMEOUT` — connection timeout in seconds (default: 10)
    /// - `UNIFIED_LLM_REQUEST_TIMEOUT` — request timeout in seconds (default: 120)
    /// - `UNIFIED_LLM_STREAM_READ_TIMEOUT` — per-chunk stream read timeout in seconds (default: 30)
    ///
    /// Only providers whose keys are present are registered. If no keys are
    /// found, returns `ConfigurationError`.
    ///
    /// To control which provider is default when multiple keys are present,
    /// use `ClientBuilder` with explicit `.default_provider()` instead.
    pub fn from_env() -> Result<Self, Error> {
        let mut builder = ClientBuilder::new();
        let timeout = Self::timeout_from_env();

        #[cfg(feature = "anthropic")]
        if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
            let adapter = if let Ok(base_url) = std::env::var("ANTHROPIC_BASE_URL") {
                crate::providers::anthropic::AnthropicAdapter::new_with_base_url_and_timeout(
                    secrecy::SecretString::from(key),
                    base_url,
                    timeout.clone(),
                )
            } else {
                crate::providers::anthropic::AnthropicAdapter::new_with_timeout(
                    secrecy::SecretString::from(key),
                    timeout.clone(),
                )
            };
            builder = builder.provider("anthropic", Box::new(adapter));
        }

        #[cfg(feature = "openai")]
        if let Ok(key) = std::env::var("OPENAI_API_KEY") {
            let mut adapter_builder =
                crate::providers::openai::OpenAiAdapter::builder(secrecy::SecretString::from(key));
            adapter_builder = adapter_builder.timeout(timeout.clone());
            if let Ok(base_url) = std::env::var("OPENAI_BASE_URL") {
                // The OpenAI adapter appends "/v1/responses" internally, so strip
                // a trailing "/v1" or "/v1/" from the env var to avoid double-path
                // (e.g. "https://api.openai.com/v1" → "https://api.openai.com").
                let normalized = base_url
                    .strip_suffix("/v1/")
                    .or_else(|| base_url.strip_suffix("/v1"))
                    .unwrap_or(&base_url);
                adapter_builder = adapter_builder.base_url(normalized);
            }
            // Wire OPENAI_ORG_ID and OPENAI_PROJECT_ID as default headers (H-8 enables this).
            let org_id = std::env::var("OPENAI_ORG_ID").ok();
            let project_id = std::env::var("OPENAI_PROJECT_ID").ok();
            if org_id.is_some() || project_id.is_some() {
                let mut headers = reqwest::header::HeaderMap::new();
                if let Some(org) = org_id {
                    if let Ok(val) = org.parse() {
                        headers.insert("OpenAI-Organization", val);
                    }
                }
                if let Some(proj) = project_id {
                    if let Ok(val) = proj.parse() {
                        headers.insert("OpenAI-Project", val);
                    }
                }
                adapter_builder = adapter_builder.default_headers(headers);
            }
            builder = builder.provider("openai", Box::new(adapter_builder.build()));
        }

        #[cfg(feature = "gemini")]
        if let Ok(key) =
            std::env::var("GEMINI_API_KEY").or_else(|_| std::env::var("GOOGLE_API_KEY"))
        {
            let adapter = if let Ok(base_url) = std::env::var("GEMINI_BASE_URL") {
                crate::providers::gemini::GeminiAdapter::new_with_base_url_and_timeout(
                    secrecy::SecretString::from(key),
                    base_url,
                    timeout,
                )
            } else {
                crate::providers::gemini::GeminiAdapter::new_with_timeout(
                    secrecy::SecretString::from(key),
                    timeout,
                )
            };
            builder = builder.provider("gemini", Box::new(adapter));
        }

        builder.build()
    }

    /// Parse timeout configuration from environment variables.
    /// Falls back to `AdapterTimeout::default()` for any unset or unparseable values.
    fn timeout_from_env() -> AdapterTimeout {
        let defaults = AdapterTimeout::default();
        AdapterTimeout {
            connect: std::env::var("UNIFIED_LLM_CONNECT_TIMEOUT")
                .ok()
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(defaults.connect),
            request: std::env::var("UNIFIED_LLM_REQUEST_TIMEOUT")
                .ok()
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(defaults.request),
            stream_read: std::env::var("UNIFIED_LLM_STREAM_READ_TIMEOUT")
                .ok()
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(defaults.stream_read),
        }
    }

    /// Send a completion request, routing to the appropriate provider.
    /// If middleware is registered, runs the onion chain (request phase in
    /// registration order, response phase in reverse order).
    pub async fn complete(&self, request: Request) -> Result<Response, Error> {
        let provider = self.resolve_provider(&request)?;

        if self.middleware.is_empty() {
            return provider.complete(request).await;
        }

        self.run_complete_chain(request, provider, 0).await
    }

    /// Recursively build the middleware onion for complete requests.
    /// At `index == middleware.len()`, calls the provider directly.
    fn run_complete_chain<'a>(
        &'a self,
        request: Request,
        provider: &'a dyn ProviderAdapter,
        index: usize,
    ) -> BoxFuture<'a, Result<Response, Error>> {
        if index >= self.middleware.len() {
            return provider.complete(request);
        }
        let mw = &self.middleware[index];
        let next = Next {
            complete_fn: Box::new(move |req| self.run_complete_chain(req, provider, index + 1)),
            stream_fn: Box::new(|_| {
                Box::pin(async {
                    Err(Error::configuration(
                        "stream not available in complete context",
                    ))
                })
            }),
        };
        mw.process(request, next)
    }

    /// Send a streaming request, routing to the appropriate provider.
    /// If middleware is registered, runs the onion chain for streaming.
    pub fn stream(
        &self,
        request: Request,
    ) -> Result<BoxStream<'_, Result<StreamEvent, Error>>, Error> {
        let provider = self.resolve_provider(&request)?;

        if self.middleware.is_empty() {
            return Ok(provider.stream(request));
        }

        // Return a stream that lazily evaluates the middleware chain.
        Ok(Box::pin(async_stream::stream! {
            use futures::StreamExt;
            match self.run_stream_chain(request, provider, 0).await {
                Ok(mut inner) => {
                    while let Some(event) = inner.next().await {
                        yield event;
                    }
                }
                Err(e) => {
                    yield Err(e);
                }
            }
        }))
    }

    /// Recursively build the middleware onion for stream requests.
    /// At `index == middleware.len()`, calls the provider directly.
    fn run_stream_chain<'a>(
        &'a self,
        request: Request,
        provider: &'a dyn ProviderAdapter,
        index: usize,
    ) -> BoxFuture<'a, Result<BoxStream<'a, Result<StreamEvent, Error>>, Error>> {
        if index >= self.middleware.len() {
            return Box::pin(async move { Ok(provider.stream(request)) });
        }
        let mw = &self.middleware[index];
        let next = Next {
            complete_fn: Box::new(|_| {
                Box::pin(async {
                    Err(Error::configuration(
                        "complete not available in stream context",
                    ))
                })
            }),
            stream_fn: Box::new(move |req| self.run_stream_chain(req, provider, index + 1)),
        };
        mw.process_stream(request, next)
    }

    /// Close all providers, releasing resources.
    pub async fn close(&self) -> Result<(), Error> {
        for provider in self.providers.values() {
            provider.close().await?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::MockProvider;
    use serial_test::serial;
    use unified_llm_types::*;

    fn make_test_response(text: &str, provider: &str) -> Response {
        Response {
            id: "resp_test".into(),
            model: "test-model".into(),
            provider: provider.into(),
            message: Message::assistant(text),
            finish_reason: FinishReason::stop(),
            usage: Usage::default(),
            raw: None,
            warnings: vec![],
            rate_limit: None,
        }
    }

    // --- ClientBuilder tests ---

    #[test]
    fn test_builder_no_providers_returns_error() {
        let result = ClientBuilder::new().build();
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert_eq!(err.kind, ErrorKind::Configuration);
    }

    #[test]
    fn test_builder_with_one_provider() {
        let mock = MockProvider::new("mock").with_response(make_test_response("Hello", "mock"));
        let client = ClientBuilder::new()
            .provider("mock", Box::new(mock))
            .build();
        assert!(client.is_ok());
    }

    #[test]
    fn test_builder_first_provider_becomes_default() {
        let mock = MockProvider::new("first");
        let client = ClientBuilder::new()
            .provider("first", Box::new(mock))
            .build()
            .unwrap();
        assert_eq!(client.default_provider, Some("first".to_string()));
    }

    #[test]
    fn test_builder_explicit_default_overrides() {
        let mock_a = MockProvider::new("a");
        let mock_b = MockProvider::new("b");
        let client = ClientBuilder::new()
            .provider("a", Box::new(mock_a))
            .provider("b", Box::new(mock_b))
            .default_provider("b")
            .build()
            .unwrap();
        assert_eq!(client.default_provider, Some("b".to_string()));
    }

    #[test]
    fn test_client_builder_shortcut() {
        let mock = MockProvider::new("mock");
        let client = Client::builder().provider("mock", Box::new(mock)).build();
        assert!(client.is_ok());
    }

    // --- Client::complete routing tests ---

    #[tokio::test]
    async fn test_client_complete_uses_default_provider() {
        let mock =
            MockProvider::new("default").with_response(make_test_response("Hello", "default"));
        let client = ClientBuilder::new()
            .provider("default", Box::new(mock))
            .build()
            .unwrap();

        let resp = client
            .complete(Request::default().model("test"))
            .await
            .unwrap();
        assert_eq!(resp.text(), "Hello");
        assert_eq!(resp.provider, "default");
    }

    #[tokio::test]
    async fn test_client_complete_routes_by_provider_field() {
        let mock_a = MockProvider::new("a").with_response(make_test_response("From A", "a"));
        let mock_b = MockProvider::new("b").with_response(make_test_response("From B", "b"));
        let client = ClientBuilder::new()
            .provider("a", Box::new(mock_a))
            .provider("b", Box::new(mock_b))
            .default_provider("a")
            .build()
            .unwrap();

        // Route to b explicitly
        let req = Request::default().model("test").provider(Some("b".into()));
        let resp = client.complete(req).await.unwrap();
        assert_eq!(resp.provider, "b");
        assert_eq!(resp.text(), "From B");
    }

    #[tokio::test]
    async fn test_client_complete_unknown_provider_error() {
        let mock = MockProvider::new("only").with_response(make_test_response("Hello", "only"));
        let client = ClientBuilder::new()
            .provider("only", Box::new(mock))
            .build()
            .unwrap();

        let req = Request::default()
            .model("test")
            .provider(Some("nonexistent".into()));
        let err = client.complete(req).await.unwrap_err();
        assert_eq!(err.kind, ErrorKind::Configuration);
        assert!(err.message.contains("Unknown provider"));
    }

    #[tokio::test]
    async fn test_client_complete_no_default_no_provider_error() {
        let mock = MockProvider::new("only");
        // Build without setting default, then override the auto-default
        let client = Client {
            providers: {
                let mut map = HashMap::new();
                map.insert(
                    "only".to_string(),
                    Box::new(mock) as Box<dyn ProviderAdapter>,
                );
                map
            },
            default_provider: None,
            middleware: Vec::new(),
        };

        let err = client
            .complete(Request::default().model("test"))
            .await
            .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Configuration);
        assert!(err.message.contains("No provider specified"));
    }

    // --- Client::from_env tests ---

    #[test]
    #[serial]
    fn test_from_env_no_keys_returns_error() {
        // Safety: Tests run serially via #[serial], no concurrent env access.
        unsafe {
            std::env::remove_var("ANTHROPIC_API_KEY");
            std::env::remove_var("OPENAI_API_KEY");
            std::env::remove_var("GEMINI_API_KEY");
            std::env::remove_var("GOOGLE_API_KEY");
        }
        let err = Client::from_env().err().unwrap();
        assert_eq!(err.kind, ErrorKind::Configuration);
    }

    #[test]
    #[serial]
    fn test_from_env_with_anthropic_key_creates_client() {
        // Safety: Tests run serially via #[serial], no concurrent env access.
        unsafe {
            std::env::remove_var("OPENAI_API_KEY");
            std::env::remove_var("GEMINI_API_KEY");
            std::env::remove_var("GOOGLE_API_KEY");
            std::env::set_var("ANTHROPIC_API_KEY", "test-key-123");
        }
        let client = Client::from_env().unwrap();
        assert_eq!(client.default_provider, Some("anthropic".to_string()));
        assert!(client.providers.contains_key("anthropic"));
        // Safety: Tests run serially via #[serial], no concurrent env access.
        unsafe {
            std::env::remove_var("ANTHROPIC_API_KEY");
        }
    }

    #[tokio::test]
    async fn test_client_complete_with_multiple_providers_default_selection() {
        let mock_a = MockProvider::new("a").with_response(make_test_response("From A", "a"));
        let mock_b = MockProvider::new("b").with_response(make_test_response("From B", "b"));
        let client = ClientBuilder::new()
            .provider("a", Box::new(mock_a))
            .provider("b", Box::new(mock_b))
            .default_provider("a")
            .build()
            .unwrap();

        // No provider specified, should use default "a"
        let resp = client
            .complete(Request::default().model("test"))
            .await
            .unwrap();
        assert_eq!(resp.provider, "a");
    }

    // --- H-4: Optional env var reading tests ---

    #[test]
    #[serial]
    fn test_from_env_with_openai_base_url() {
        // Safety: Tests run serially via #[serial], no concurrent env access.
        unsafe {
            std::env::remove_var("ANTHROPIC_API_KEY");
            std::env::remove_var("GEMINI_API_KEY");
            std::env::remove_var("GOOGLE_API_KEY");
            std::env::set_var("OPENAI_API_KEY", "test-key");
            std::env::set_var("OPENAI_BASE_URL", "https://custom.openai.com");
        }
        let client = Client::from_env().unwrap();
        assert!(client.providers.contains_key("openai"));
        unsafe {
            std::env::remove_var("OPENAI_API_KEY");
            std::env::remove_var("OPENAI_BASE_URL");
        }
    }

    #[test]
    #[serial]
    fn test_from_env_with_anthropic_base_url() {
        unsafe {
            std::env::remove_var("OPENAI_API_KEY");
            std::env::remove_var("GEMINI_API_KEY");
            std::env::remove_var("GOOGLE_API_KEY");
            std::env::set_var("ANTHROPIC_API_KEY", "test-key");
            std::env::set_var("ANTHROPIC_BASE_URL", "https://custom.anthropic.com");
        }
        let client = Client::from_env().unwrap();
        assert!(client.providers.contains_key("anthropic"));
        unsafe {
            std::env::remove_var("ANTHROPIC_API_KEY");
            std::env::remove_var("ANTHROPIC_BASE_URL");
        }
    }

    #[test]
    #[serial]
    fn test_from_env_with_gemini_base_url() {
        unsafe {
            std::env::remove_var("OPENAI_API_KEY");
            std::env::remove_var("ANTHROPIC_API_KEY");
            std::env::set_var("GEMINI_API_KEY", "test-key");
            std::env::set_var("GEMINI_BASE_URL", "https://custom.gemini.com");
        }
        let client = Client::from_env().unwrap();
        assert!(client.providers.contains_key("gemini"));
        unsafe {
            std::env::remove_var("GEMINI_API_KEY");
            std::env::remove_var("GEMINI_BASE_URL");
        }
    }

    #[test]
    #[serial]
    fn test_from_env_with_openai_org_and_project() {
        unsafe {
            std::env::remove_var("ANTHROPIC_API_KEY");
            std::env::remove_var("GEMINI_API_KEY");
            std::env::remove_var("GOOGLE_API_KEY");
            std::env::set_var("OPENAI_API_KEY", "test-key");
            std::env::set_var("OPENAI_ORG_ID", "org-test123");
            std::env::set_var("OPENAI_PROJECT_ID", "proj-test456");
        }
        // Should not error — org/project headers are wired via the builder.
        let client = Client::from_env().unwrap();
        assert!(client.providers.contains_key("openai"));
        unsafe {
            std::env::remove_var("OPENAI_API_KEY");
            std::env::remove_var("OPENAI_ORG_ID");
            std::env::remove_var("OPENAI_PROJECT_ID");
        }
    }
}
