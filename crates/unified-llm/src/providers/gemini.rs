// Gemini (Google AI) generateContent API adapter.

use std::collections::HashMap;
use std::sync::Mutex;

use futures::StreamExt;
use secrecy::{ExposeSecret, SecretString};

use unified_llm_types::{
    AdapterTimeout, ArgumentValue, BoxFuture, BoxStream, ContentPart, Error, ErrorKind,
    FinishReason, Message, ProviderAdapter, Request, Response, Role, StreamEvent, StreamEventType,
    ThinkingData, ToolCall, ToolCallData, Usage, Warning,
};

use crate::util::sse::SseParser;

/// Default Gemini API base URL.
const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com";

/// Gemini generateContent API adapter.
pub struct GeminiAdapter {
    api_key: SecretString,
    base_url: String,
    http_client: reqwest::Client,
    /// Maps synthetic tool call IDs to function names for functionResponse routing.
    tool_call_map: Mutex<HashMap<String, String>>,
    /// Per-chunk timeout for streaming responses (from AdapterTimeout.stream_read).
    stream_read_timeout: std::time::Duration,
}

impl GeminiAdapter {
    /// Create a new GeminiAdapter with the given API key.
    ///
    /// Uses default timeouts: connect=10s, request=120s.
    pub fn new(api_key: SecretString) -> Self {
        let timeout = AdapterTimeout::default();
        Self {
            api_key,
            base_url: DEFAULT_BASE_URL.to_string(),
            http_client: Self::build_http_client(&timeout),
            tool_call_map: Mutex::new(HashMap::new()),
            stream_read_timeout: std::time::Duration::from_secs_f64(timeout.stream_read),
        }
    }

    /// Create a new GeminiAdapter with a custom base URL (for testing with wiremock).
    ///
    /// Uses default timeouts: connect=10s, request=120s.
    pub fn new_with_base_url(api_key: SecretString, base_url: impl Into<String>) -> Self {
        let timeout = AdapterTimeout::default();
        Self {
            api_key,
            base_url: crate::util::normalize_base_url(&base_url.into()),
            http_client: Self::build_http_client(&timeout),
            tool_call_map: Mutex::new(HashMap::new()),
            stream_read_timeout: std::time::Duration::from_secs_f64(timeout.stream_read),
        }
    }

    /// Create a new GeminiAdapter with custom timeouts.
    pub fn new_with_timeout(api_key: SecretString, timeout: AdapterTimeout) -> Self {
        Self {
            api_key,
            base_url: DEFAULT_BASE_URL.to_string(),
            http_client: Self::build_http_client(&timeout),
            tool_call_map: Mutex::new(HashMap::new()),
            stream_read_timeout: std::time::Duration::from_secs_f64(timeout.stream_read),
        }
    }

    /// Create with custom base URL and timeouts.
    pub fn new_with_base_url_and_timeout(
        api_key: SecretString,
        base_url: impl Into<String>,
        timeout: AdapterTimeout,
    ) -> Self {
        Self {
            api_key,
            base_url: crate::util::normalize_base_url(&base_url.into()),
            http_client: Self::build_http_client(&timeout),
            tool_call_map: Mutex::new(HashMap::new()),
            stream_read_timeout: std::time::Duration::from_secs_f64(timeout.stream_read),
        }
    }

    /// Create from environment variable GEMINI_API_KEY, falling back to GOOGLE_API_KEY.
    pub fn from_env() -> Result<Self, Error> {
        let api_key = std::env::var("GEMINI_API_KEY")
            .or_else(|_| std::env::var("GOOGLE_API_KEY"))
            .map_err(|_| Error::configuration("GEMINI_API_KEY or GOOGLE_API_KEY not set"))?;
        Ok(Self::new(SecretString::from(api_key)))
    }

    /// Create a builder for fine-grained configuration.
    pub fn builder(api_key: SecretString) -> GeminiAdapterBuilder {
        GeminiAdapterBuilder::new(api_key)
    }

    /// Build an HTTP client with the given timeout configuration.
    ///
    /// Wires `connect` → `connect_timeout()` and `request` → `timeout()`.
    /// Note: `stream_read` requires a custom per-chunk timeout implementation
    /// and is not wired here.
    fn build_http_client(timeout: &AdapterTimeout) -> reqwest::Client {
        Self::build_http_client_with_headers(timeout, None)
    }

    fn build_http_client_with_headers(
        timeout: &AdapterTimeout,
        default_headers: Option<reqwest::header::HeaderMap>,
    ) -> reqwest::Client {
        let mut builder = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs_f64(timeout.connect))
            .timeout(std::time::Duration::from_secs_f64(timeout.request));
        if let Some(headers) = default_headers {
            builder = builder.default_headers(headers);
        }
        builder.build().expect("Failed to build HTTP client")
    }

    /// Build common HTTP headers for Gemini API requests.
    ///
    // DEVIATION FROM SPEC: Spec §7 suggests query parameter authentication for Gemini.
    // We use x-goog-api-key header instead — this is Google's recommended approach
    // and avoids leaking API keys in server access logs, proxy logs, and browser history.
    // Both methods are supported by the Gemini API.
    fn build_headers(&self) -> Result<reqwest::header::HeaderMap, Error> {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            "x-goog-api-key",
            self.api_key.expose_secret().parse().map_err(|_| {
                Error::configuration("Invalid API key: contains non-ASCII or control characters")
            })?,
        );
        headers.insert("content-type", "application/json".parse().unwrap());
        Ok(headers)
    }

    /// Build the URL for a non-streaming generateContent request.
    fn build_url(&self, model: &str) -> String {
        format!("{}/v1beta/models/{}:generateContent", self.base_url, model)
    }

    /// Build the URL for a streaming generateContent request.
    fn build_stream_url(&self, model: &str) -> String {
        format!(
            "{}/v1beta/models/{}:streamGenerateContent?alt=sse",
            self.base_url, model
        )
    }

    /// Clear the tool call ID → function name mapping.
    /// Called at the start of each request to prevent cross-request pollution.
    pub fn clear_tool_call_map(&self) {
        self.tool_call_map
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
    }

    /// Perform the actual HTTP request for complete().
    async fn do_complete(&self, mut request: Request) -> Result<Response, Error> {
        // H-4: Pre-resolve local file images to avoid blocking I/O in translate_request.
        crate::util::image::pre_resolve_local_images(&mut request.messages).await?;

        // Pre-populate tool_call_map from conversation history.
        // This makes the adapter stateless across requests — each request
        // carries enough context to resolve synthetic IDs.
        {
            let mut map = self.tool_call_map.lock().unwrap_or_else(|e| e.into_inner());
            map.clear();
            for msg in &request.messages {
                if msg.role == unified_llm_types::message::Role::Assistant {
                    for part in &msg.content {
                        if let unified_llm_types::ContentPart::ToolCall { tool_call } = part {
                            map.insert(tool_call.id.clone(), tool_call.name.clone());
                        }
                    }
                }
            }
        }
        let url = self.build_url(&request.model);
        let (body, translation_warnings) = {
            let map = self.tool_call_map.lock().unwrap_or_else(|e| e.into_inner());
            translate_request(&request, Some(&map))
        };
        let request_headers = self.build_headers()?;

        let http_response = self
            .http_client
            .post(&url)
            .headers(request_headers)
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::network(format!("HTTP request failed: {e}"), e))?;

        let status = http_response.status().as_u16();
        let headers = http_response.headers().clone();

        if status >= 400 {
            let error_body: serde_json::Value = http_response.json().await.unwrap_or(
                serde_json::json!({"error": {"message": "Failed to parse error response"}}),
            );
            return Err(parse_error(status, &headers, error_body));
        }

        let response_body: serde_json::Value = http_response
            .json()
            .await
            .map_err(|e| Error::network(format!("Failed to parse response: {e}"), e))?;

        let mut response = parse_response(response_body, &headers)?;
        response.warnings = translation_warnings;

        // Store tool call ID → function name mappings for future functionResponse routing
        for part in &response.message.content {
            if let ContentPart::ToolCall { tool_call } = part {
                if let Ok(mut map) = self.tool_call_map.lock() {
                    map.insert(tool_call.id.clone(), tool_call.name.clone());
                }
            }
        }

        Ok(response)
    }

    /// Perform the HTTP request for stream() and return a stream of events.
    fn do_stream(&self, mut request: Request) -> BoxStream<'_, Result<StreamEvent, Error>> {
        // Pre-populate tool_call_map from conversation history.
        {
            let mut map = self.tool_call_map.lock().unwrap_or_else(|e| e.into_inner());
            map.clear();
            for msg in &request.messages {
                if msg.role == unified_llm_types::message::Role::Assistant {
                    for part in &msg.content {
                        if let unified_llm_types::ContentPart::ToolCall { tool_call } = part {
                            map.insert(tool_call.id.clone(), tool_call.name.clone());
                        }
                    }
                }
            }
        }
        let stream = async_stream::stream! {
            // H-4: Pre-resolve local file images to avoid blocking I/O in translate_request.
            if let Err(e) = crate::util::image::pre_resolve_local_images(&mut request.messages).await {
                yield Err(e);
                return;
            }

            let url = self.build_stream_url(&request.model);
            let (body, _translation_warnings) = {
                let map = self.tool_call_map.lock().unwrap_or_else(|e| e.into_inner());
                translate_request(&request, Some(&map))
            };
            let request_headers = match self.build_headers() {
                Ok(h) => h,
                Err(e) => {
                    yield Err(e);
                    return;
                }
            };

            let http_response = match self
                .http_client
                .post(&url)
                .headers(request_headers)
                .json(&body)
                .send()
                .await
            {
                Ok(resp) => resp,
                Err(e) => {
                    yield Err(Error::network(format!("HTTP request failed: {e}"), e));
                    return;
                }
            };

            let status = http_response.status().as_u16();
            let headers = http_response.headers().clone();

            if status >= 400 {
                let error_body: serde_json::Value = http_response
                    .json()
                    .await
                    .unwrap_or(serde_json::json!({"error": {"message": "Failed to parse error response"}}));
                yield Err(parse_error(status, &headers, error_body));
                return;
            }

            // Incremental streaming: read chunks as they arrive
            let mut parser = SseParser::new();
            let mut byte_stream = http_response.bytes_stream();
            let mut translator = GeminiStreamTranslator::new();
            let stream_read_timeout = self.stream_read_timeout;

            loop {
                // C-2/stream_read: enforce per-chunk timeout
                let chunk_result = match tokio::time::timeout(
                    stream_read_timeout,
                    byte_stream.next(),
                ).await {
                    Ok(Some(result)) => result,
                    Ok(None) => break, // stream ended
                    Err(_elapsed) => {
                        yield Err(Error::stream(
                            format!("Stream read timed out after {:?}", stream_read_timeout),
                            std::io::Error::new(std::io::ErrorKind::TimedOut, "stream read timeout"),
                        ));
                        return;
                    }
                };

                let chunk = match chunk_result {
                    Ok(bytes) => bytes,
                    Err(e) => {
                        yield Err(Error::stream(format!("Stream read error: {e}"), e));
                        return;
                    }
                };

                let chunk_str = match std::str::from_utf8(&chunk) {
                    Ok(s) => s,
                    Err(_) => continue,
                };

                let sse_events = parser.feed(chunk_str);

                for sse_event in sse_events {
                    // Gemini SSE has no event type field — just data lines.
                    // Process the data directly as JSON.
                    let data: serde_json::Value = match serde_json::from_str(&sse_event.data) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };

                    for evt in translator.process(&data) {
                        // Store tool call mappings from stream events
                        if evt.event_type == StreamEventType::ToolCallStart {
                            if let Some(ref tc) = evt.tool_call {
                                if let Ok(mut map) = self.tool_call_map.lock() {
                                    map.insert(tc.id.clone(), tc.name.clone());
                                }
                            }
                        }
                        yield Ok(evt);
                    }
                }
            }

            // Emit TextEnd and Finish if stream ended without finishReason
            for evt in translator.finalize() {
                yield Ok(evt);
            }
        };
        Box::pin(stream)
    }
}

/// Builder for constructing a `GeminiAdapter` with fine-grained configuration.
pub struct GeminiAdapterBuilder {
    api_key: SecretString,
    base_url: Option<String>,
    timeout: Option<AdapterTimeout>,
    default_headers: Option<reqwest::header::HeaderMap>,
}

impl GeminiAdapterBuilder {
    /// Create a new builder with the required API key.
    pub fn new(api_key: SecretString) -> Self {
        Self {
            api_key,
            base_url: None,
            timeout: None,
            default_headers: None,
        }
    }

    /// Set a custom base URL (e.g., for proxies or testing).
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Set custom timeout configuration.
    pub fn timeout(mut self, timeout: AdapterTimeout) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Set default HTTP headers sent with every request.
    pub fn default_headers(mut self, headers: reqwest::header::HeaderMap) -> Self {
        self.default_headers = Some(headers);
        self
    }

    /// Build the `GeminiAdapter`.
    pub fn build(self) -> GeminiAdapter {
        let timeout = self.timeout.unwrap_or_default();
        let base_url = self
            .base_url
            .map(|u| crate::util::normalize_base_url(&u))
            .unwrap_or_else(|| DEFAULT_BASE_URL.to_string());
        GeminiAdapter {
            api_key: self.api_key,
            base_url,
            http_client: GeminiAdapter::build_http_client_with_headers(
                &timeout,
                self.default_headers,
            ),
            tool_call_map: Mutex::new(HashMap::new()),
            stream_read_timeout: std::time::Duration::from_secs_f64(timeout.stream_read),
        }
    }
}

impl ProviderAdapter for GeminiAdapter {
    fn name(&self) -> &str {
        "gemini"
    }

    fn complete(&self, request: Request) -> BoxFuture<'_, Result<Response, Error>> {
        Box::pin(self.do_complete(request))
    }

    fn stream(&self, request: Request) -> BoxStream<'_, Result<StreamEvent, Error>> {
        self.do_stream(request)
    }
}

// === Request Translation ===

/// Translate a unified Request into a Gemini generateContent JSON body.
pub(crate) fn translate_request(
    request: &Request,
    tool_call_map: Option<&std::collections::HashMap<String, String>>,
) -> (serde_json::Value, Vec<Warning>) {
    let mut body = serde_json::Map::new();
    let mut warnings: Vec<Warning> = Vec::new();

    // All system messages are extracted regardless of conversation position per provider API
    // conventions. Mid-conversation system messages are repositioned to the system prompt area.
    let mut system_parts: Vec<String> = Vec::new();
    let mut contents: Vec<serde_json::Value> = Vec::new();

    for msg in &request.messages {
        match msg.role {
            Role::System | Role::Developer => {
                system_parts.push(msg.text());
            }
            Role::User => {
                let (parts, mut part_warnings) = translate_content_parts(&msg.content);
                warnings.append(&mut part_warnings);
                contents.push(serde_json::json!({
                    "role": "user",
                    "parts": parts,
                }));
            }
            Role::Assistant => {
                let has_tool_calls = msg
                    .content
                    .iter()
                    .any(|p| matches!(p, ContentPart::ToolCall { .. }));

                if has_tool_calls {
                    // Assistant tool calls → model role with functionCall parts
                    let mut parts = Vec::new();
                    for part in &msg.content {
                        match part {
                            ContentPart::ToolCall { tool_call } => {
                                let args = match &tool_call.arguments {
                                    ArgumentValue::Dict(map) => {
                                        serde_json::Value::Object(map.clone())
                                    }
                                    ArgumentValue::Raw(s) => {
                                        serde_json::from_str(s).unwrap_or(serde_json::json!({}))
                                    }
                                };
                                parts.push(serde_json::json!({
                                    "functionCall": {
                                        "name": tool_call.name,
                                        "args": args,
                                    }
                                }));
                            }
                            ContentPart::Text { text } => {
                                parts.push(serde_json::json!({"text": text}));
                            }
                            _ => {}
                        }
                    }
                    contents.push(serde_json::json!({
                        "role": "model",
                        "parts": parts,
                    }));
                } else {
                    let (parts, mut part_warnings) = translate_content_parts(&msg.content);
                    warnings.append(&mut part_warnings);
                    contents.push(serde_json::json!({
                        "role": "model",
                        "parts": parts,
                    }));
                }
            }
            Role::Tool => {
                // Tool results → user role with functionResponse parts
                let mut parts = Vec::new();
                for part in &msg.content {
                    if let ContentPart::ToolResult { tool_result } = part {
                        // Look up function name from tool_call_id
                        // For request translation, we use the tool_call_id as the name
                        // since Gemini uses name-based routing.
                        // The caller should ensure tool_call_id maps to function name.
                        let mut response_value = match &tool_result.content {
                            serde_json::Value::String(s) => {
                                // Gemini expects object, not string — wrap in {"result": "..."}
                                serde_json::json!({"result": s})
                            }
                            serde_json::Value::Object(_) => tool_result.content.clone(),
                            other => serde_json::json!({"result": other}),
                        };
                        // Preserve is_error flag in the response object
                        if tool_result.is_error {
                            if let Some(obj) = response_value.as_object_mut() {
                                obj.insert("is_error".to_string(), serde_json::Value::Bool(true));
                            }
                        }
                        let resolved_name = tool_call_map
                            .and_then(|m| m.get(&tool_result.tool_call_id))
                            .cloned()
                            .unwrap_or_else(|| tool_result.tool_call_id.clone());
                        parts.push(serde_json::json!({
                            "functionResponse": {
                                "name": resolved_name,
                                "response": response_value,
                            }
                        }));
                    }
                }
                contents.push(serde_json::json!({
                    "role": "user",
                    "parts": parts,
                }));
            }
        }
    }

    // systemInstruction
    if !system_parts.is_empty() {
        let text = system_parts.join("\n\n");
        body.insert(
            "systemInstruction".into(),
            serde_json::json!({
                "parts": [{"text": text}]
            }),
        );
    }

    // contents
    body.insert("contents".into(), serde_json::Value::Array(contents));

    // generationConfig
    let mut gen_config = serde_json::Map::new();
    if let Some(max_tokens) = request.max_tokens {
        gen_config.insert("maxOutputTokens".into(), serde_json::json!(max_tokens));
    }
    if let Some(temp) = request.temperature {
        gen_config.insert("temperature".into(), serde_json::json!(temp));
    }
    if let Some(top_p) = request.top_p {
        gen_config.insert("topP".into(), serde_json::json!(top_p));
    }
    if let Some(ref stop) = request.stop_sequences {
        gen_config.insert("stopSequences".into(), serde_json::json!(stop));
    }

    // Structured output: response_format → responseMimeType + responseSchema
    if let Some(ref fmt) = request.response_format {
        if fmt.r#type == "json_schema" {
            gen_config.insert(
                "responseMimeType".into(),
                serde_json::Value::String("application/json".to_string()),
            );
            if let Some(ref schema) = fmt.json_schema {
                gen_config.insert("responseSchema".into(), schema.clone());
            }
        }
    }

    // Thinking config from provider_options.gemini.thinkingConfig
    // Must be at request root level, NOT inside generationConfig.
    if let Some(gemini_opts) =
        crate::util::provider_options::get_provider_options(&request.provider_options, "gemini")
    {
        if let Some(thinking_config) = gemini_opts.get("thinkingConfig") {
            body.insert("thinkingConfig".into(), thinking_config.clone());
        }
    }

    if !gen_config.is_empty() {
        body.insert(
            "generationConfig".into(),
            serde_json::Value::Object(gen_config),
        );
    }

    // Tools — functionDeclarations wrapper
    if let Some(ref tools) = request.tools {
        let fn_decls: Vec<serde_json::Value> = tools
            .iter()
            .map(|t| {
                serde_json::json!({
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                })
            })
            .collect();
        body.insert(
            "tools".into(),
            serde_json::json!([{"functionDeclarations": fn_decls}]),
        );
    }

    // Tool choice → toolConfig.functionCallingConfig
    if let Some(ref tc) = request.tool_choice {
        let config = match tc.mode.as_str() {
            "auto" => serde_json::json!({"mode": "AUTO"}),
            "none" => serde_json::json!({"mode": "NONE"}),
            "required" => serde_json::json!({"mode": "ANY"}),
            "named" => {
                if let Some(ref name) = tc.tool_name {
                    serde_json::json!({
                        "mode": "ANY",
                        "allowedFunctionNames": [name],
                    })
                } else {
                    serde_json::json!({"mode": "AUTO"})
                }
            }
            _ => serde_json::json!({"mode": "AUTO"}),
        };
        body.insert(
            "toolConfig".into(),
            serde_json::json!({"functionCallingConfig": config}),
        );
    }

    // Map reasoning_effort to Gemini's thinkingConfig.
    // Only inject if thinkingConfig is not already in provider_options.
    if let Some(ref effort) = request.reasoning_effort {
        let has_explicit_thinking = request
            .provider_options
            .as_ref()
            .and_then(|opts| opts.get("gemini"))
            .and_then(|g| g.get("thinkingConfig"))
            .is_some();

        if !has_explicit_thinking {
            let level = match effort.as_str() {
                "none" => "THINKING_BUDGET_NONE",
                "low" => "LOW",
                "medium" => "MEDIUM",
                "high" => "HIGH",
                _ => "MEDIUM",
            };
            body.insert(
                "thinkingConfig".into(),
                serde_json::json!({
                    "thinkingLevel": level
                }),
            );
        }
    }

    // Provider options passthrough (filter internal keys)
    const INTERNAL_KEYS: &[&str] = &["thinkingConfig"];
    if let Some(opts) =
        crate::util::provider_options::get_provider_options(&request.provider_options, "gemini")
    {
        let mut body_val = serde_json::Value::Object(body);
        crate::util::provider_options::merge_provider_options(&mut body_val, &opts, INTERNAL_KEYS);
        return (body_val, warnings);
    }

    (serde_json::Value::Object(body), warnings)
}

/// Translate unified ContentParts into Gemini parts array.
/// Returns the translated parts and any warnings for dropped/unsupported content.
fn translate_content_parts(parts: &[ContentPart]) -> (Vec<serde_json::Value>, Vec<Warning>) {
    let mut warnings = Vec::new();
    let blocks = parts
        .iter()
        .filter_map(|part| match part {
            ContentPart::Text { text } => Some(serde_json::json!({"text": text})),
            ContentPart::Image { image } => {
                if let Some(ref data) = image.data {
                    let b64 = crate::util::image::base64_encode(data);
                    let media_type = image.media_type.as_deref().unwrap_or("image/png");
                    Some(serde_json::json!({
                        "inlineData": {
                            "mimeType": media_type,
                            "data": b64,
                        }
                    }))
                } else {
                    image.url.as_ref().map(|url| {
                        if crate::util::image::is_local_path(url) {
                            match crate::util::image::resolve_local_file(url) {
                                Ok((data, mime)) => {
                                    let b64 = crate::util::image::base64_encode(&data);
                                    serde_json::json!({
                                        "inlineData": {
                                            "mimeType": mime,
                                            "data": b64,
                                        }
                                    })
                                }
                                Err(e) => {
                                    tracing::warn!(
                                        "Failed to resolve local image '{}': {}",
                                        url,
                                        e.message
                                    );
                                    let media_type =
                                        image.media_type.as_deref().unwrap_or_else(|| {
                                            crate::util::image::infer_mime_type(url)
                                        });
                                    serde_json::json!({
                                        "fileData": {
                                            "mimeType": media_type,
                                            "fileUri": url,
                                        }
                                    })
                                }
                            }
                        } else {
                            let media_type = image
                                .media_type
                                .as_deref()
                                .unwrap_or_else(|| crate::util::image::infer_mime_type(url));
                            serde_json::json!({
                                "fileData": {
                                    "mimeType": media_type,
                                    "fileUri": url,
                                }
                            })
                        }
                    })
                }
            }
            ContentPart::Thinking { thinking } => {
                let mut part = serde_json::json!({
                    "thought": true,
                    "text": thinking.text,
                });
                if let Some(ref sig) = thinking.signature {
                    part["thoughtSignature"] = serde_json::Value::String(sig.clone());
                }
                Some(part)
            }
            ContentPart::RedactedThinking { thinking } => {
                // Gemini doesn't have a direct equivalent of redacted thinking.
                // Preserve as opaque thought part with empty text.
                let mut part = serde_json::json!({
                    "thought": true,
                    "text": "",
                });
                if let Some(ref sig) = thinking.signature {
                    part["thoughtSignature"] = serde_json::Value::String(sig.clone());
                }
                Some(part)
            }
            other => {
                let msg = format!(
                    "Dropped unsupported content part kind={:?} for provider=gemini",
                    other.kind()
                );
                tracing::warn!("{}", msg);
                warnings.push(Warning {
                    message: msg,
                    code: Some("dropped_content_part".to_string()),
                });
                None
            }
        })
        .collect();
    (blocks, warnings)
}

// === Response Translation ===

/// Map Gemini finishReason to unified FinishReason.
/// If any functionCall parts are present, returns "tool_calls" regardless of the raw reason.
fn map_finish_reason(reason: &str, has_function_calls: bool) -> FinishReason {
    if has_function_calls {
        return FinishReason {
            reason: "tool_calls".to_string(),
            raw: Some(reason.to_string()),
        };
    }
    let unified = match reason {
        "STOP" => "stop",
        "MAX_TOKENS" => "length",
        "SAFETY" | "RECITATION" => "content_filter",
        _ => "other",
    };
    FinishReason {
        reason: unified.to_string(),
        raw: Some(reason.to_string()),
    }
}

/// Parse a Gemini generateContent response JSON into a unified Response.
pub(crate) fn parse_response(
    raw: serde_json::Value,
    headers: &reqwest::header::HeaderMap,
) -> Result<Response, Error> {
    let id = raw
        .get("responseId")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let model = raw
        .get("modelVersion")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    // Parse candidates[0].content.parts
    let mut content_parts: Vec<ContentPart> = Vec::new();
    let mut has_function_calls = false;

    if let Some(candidate) = raw
        .get("candidates")
        .and_then(|v| v.as_array())
        .and_then(|a| a.first())
    {
        if let Some(parts) = candidate
            .get("content")
            .and_then(|c| c.get("parts"))
            .and_then(|p| p.as_array())
        {
            for part in parts {
                // Thinking part: {thought: true, text: "...", thoughtSignature: "..."}
                if part.get("thought").and_then(|v| v.as_bool()) == Some(true) {
                    let text = part
                        .get("text")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let signature = part
                        .get("thoughtSignature")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    content_parts.push(ContentPart::Thinking {
                        thinking: ThinkingData {
                            text,
                            signature,
                            redacted: false,
                            data: None,
                        },
                    });
                    continue;
                }

                // Function call part: {functionCall: {name, args, id?}}
                if let Some(fc) = part.get("functionCall") {
                    has_function_calls = true;
                    let name = fc
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let call_id = fc
                        .get("id")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| format!("call_{}", uuid::Uuid::new_v4()));
                    let args = fc
                        .get("args")
                        .and_then(|v| v.as_object())
                        .cloned()
                        .unwrap_or_default();
                    content_parts.push(ContentPart::ToolCall {
                        tool_call: ToolCallData {
                            id: call_id,
                            name,
                            arguments: ArgumentValue::Dict(args),
                            r#type: "function".to_string(),
                        },
                    });
                    continue;
                }

                // Text part: {text: "..."}
                if let Some(text) = part.get("text").and_then(|v| v.as_str()) {
                    content_parts.push(ContentPart::Text {
                        text: text.to_string(),
                    });
                }
            }
        }
    }

    // Finish reason
    let raw_finish_reason = raw
        .get("candidates")
        .and_then(|v| v.as_array())
        .and_then(|a| a.first())
        .and_then(|c| c.get("finishReason"))
        .and_then(|v| v.as_str())
        .unwrap_or("STOP");
    let finish_reason = map_finish_reason(raw_finish_reason, has_function_calls);

    // Usage from usageMetadata
    let usage_obj = raw.get("usageMetadata");
    let input_tokens = usage_obj
        .and_then(|u| u.get("promptTokenCount"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    let output_tokens = usage_obj
        .and_then(|u| u.get("candidatesTokenCount"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    let reasoning_tokens = usage_obj
        .and_then(|u| u.get("thoughtsTokenCount"))
        .and_then(|v| v.as_u64())
        .map(|v| v as u32);
    let cache_read_tokens = usage_obj
        .and_then(|u| u.get("cachedContentTokenCount"))
        .and_then(|v| v.as_u64())
        .map(|v| v as u32);

    // M-11: Gemini does not expose a cache write token count in its API.
    // The `cachedContentTokenCount` field only reports cache *read* tokens.
    // Gemini handles caching via explicit CachedContent resources rather than
    // per-request write accounting, so cache_write_tokens is always None.
    let cache_write_tokens: Option<u32> = None;

    let usage = Usage {
        input_tokens,
        output_tokens,
        // F-10: always recompute total_tokens instead of trusting provider
        total_tokens: input_tokens + output_tokens,
        reasoning_tokens,
        cache_read_tokens,
        cache_write_tokens,
        raw: usage_obj.cloned(),
    };

    Ok(Response {
        id,
        model,
        provider: "gemini".to_string(),
        message: Message {
            role: Role::Assistant,
            content: content_parts,
            name: None,
            tool_call_id: None,
        },
        finish_reason,
        usage,
        raw: Some(raw),
        warnings: vec![],
        rate_limit: crate::util::http::parse_rate_limit_headers(headers),
    })
}

// === Error Translation ===

/// Parse a Gemini error response into a unified Error.
pub(crate) fn parse_error(
    status: u16,
    headers: &reqwest::header::HeaderMap,
    body: serde_json::Value,
) -> Error {
    let (error_message, error_code) = crate::util::http::parse_provider_error_message(
        &body,
        &["error", "message"],
        &["error", "status"],
    );

    let retry_after = crate::util::http::parse_retry_after(headers);

    let mut err = Error::from_http_status(status, error_message, "gemini", Some(body), retry_after);
    err.error_code = error_code.clone();

    // F-5: Override ErrorKind based on gRPC status codes (spec §6.4)
    // GAP-3: All 8 spec-required gRPC codes mapped.
    //
    // Important: gRPC code mapping runs first, then message-based reclassification
    // can further refine (e.g. INVALID_ARGUMENT + "API key not valid" → Authentication).
    if let Some(ref code) = error_code {
        match code.as_str() {
            "DEADLINE_EXCEEDED" => err.kind = ErrorKind::RequestTimeout,
            "PERMISSION_DENIED" => err.kind = ErrorKind::AccessDenied,
            "RESOURCE_EXHAUSTED" => err.kind = ErrorKind::RateLimit,
            "UNAUTHENTICATED" => err.kind = ErrorKind::Authentication,
            "NOT_FOUND" => err.kind = ErrorKind::NotFound,
            "INVALID_ARGUMENT" => err.kind = ErrorKind::InvalidRequest,
            "UNAVAILABLE" => err.kind = ErrorKind::Server,
            "INTERNAL" => err.kind = ErrorKind::Server,
            _ => {}
        }
    }

    // GAP-3: Apply message-based reclassification AFTER gRPC override.
    // This catches cases like HTTP 400 + INVALID_ARGUMENT + "API key not valid" → Authentication.
    err.kind = Error::classify_by_message_pub(&err.message, err.kind);

    // GAP-3: Recalculate retryable based on the final kind.
    err.retryable = matches!(
        err.kind,
        ErrorKind::RateLimit
            | ErrorKind::Server
            | ErrorKind::RequestTimeout
            | ErrorKind::Network
            | ErrorKind::Stream
    );

    err
}

// === Stream Translation ===

/// Holds streaming state and translates Gemini SSE chunks into unified StreamEvents.
struct GeminiStreamTranslator {
    started: bool,
    text_started: bool,
    /// S-1: text_id generated on TextStart, propagated to TextDelta/TextEnd.
    current_text_id: Option<String>,
    reasoning_started: bool,
    /// Accumulated thoughtSignature from streaming thought parts for round-trip.
    active_thinking_signature: Option<String>,
    finished: bool,
    response_id: String,
    model: String,
}

impl GeminiStreamTranslator {
    fn new() -> Self {
        Self {
            started: false,
            text_started: false,
            current_text_id: None,
            reasoning_started: false,
            active_thinking_signature: None,
            finished: false,
            response_id: String::new(),
            model: String::new(),
        }
    }

    /// Process a single Gemini SSE data chunk (a complete GenerateContentResponse JSON).
    fn process(&mut self, data: &serde_json::Value) -> Vec<StreamEvent> {
        let mut events = Vec::new();

        // Emit StreamStart on first chunk
        if !self.started {
            self.started = true;
            self.response_id = data
                .get("responseId")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            self.model = data
                .get("modelVersion")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            events.push(StreamEvent {
                event_type: StreamEventType::StreamStart,
                id: if self.response_id.is_empty() {
                    None
                } else {
                    Some(self.response_id.clone())
                },
                ..Default::default()
            });
        }

        // Extract candidates[0]
        let candidate = match data
            .get("candidates")
            .and_then(|v| v.as_array())
            .and_then(|a| a.first())
        {
            Some(c) => c,
            None => return events,
        };

        // Process parts
        if let Some(parts) = candidate
            .get("content")
            .and_then(|c| c.get("parts"))
            .and_then(|p| p.as_array())
        {
            for part in parts {
                // Thinking part
                if part.get("thought").and_then(|v| v.as_bool()) == Some(true) {
                    if !self.reasoning_started {
                        self.reasoning_started = true;
                        self.active_thinking_signature = None; // reset for new thinking block
                        events.push(StreamEvent {
                            event_type: StreamEventType::ReasoningStart,
                            ..Default::default()
                        });
                    }
                    // Capture thoughtSignature for round-trip (last one wins)
                    if let Some(sig) = part.get("thoughtSignature").and_then(|v| v.as_str()) {
                        self.active_thinking_signature = Some(sig.to_string());
                    }
                    let text = part
                        .get("text")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    events.push(StreamEvent {
                        event_type: StreamEventType::ReasoningDelta,
                        delta: Some(text.clone()),
                        reasoning_delta: Some(text),
                        ..Default::default()
                    });
                    continue;
                }

                // Function call part — arrives COMPLETE, emit Start + End together
                if let Some(fc) = part.get("functionCall") {
                    let name = fc
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let call_id = fc
                        .get("id")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| format!("call_{}", uuid::Uuid::new_v4()));
                    let args = fc
                        .get("args")
                        .and_then(|v| v.as_object())
                        .cloned()
                        .unwrap_or_default();
                    let raw_args =
                        serde_json::to_string(&serde_json::Value::Object(args.clone())).ok();

                    let tool_call = ToolCall {
                        id: call_id,
                        name,
                        arguments: args,
                        raw_arguments: raw_args,
                    };

                    events.push(StreamEvent {
                        event_type: StreamEventType::ToolCallStart,
                        tool_call: Some(tool_call.clone()),
                        ..Default::default()
                    });
                    events.push(StreamEvent {
                        event_type: StreamEventType::ToolCallEnd,
                        tool_call: Some(tool_call),
                        ..Default::default()
                    });
                    continue;
                }

                // Text part
                if let Some(text) = part.get("text").and_then(|v| v.as_str()) {
                    // Emit ReasoningEnd when transitioning from reasoning to text
                    if self.reasoning_started {
                        events.push(StreamEvent {
                            event_type: StreamEventType::ReasoningEnd,
                            raw: self
                                .active_thinking_signature
                                .take()
                                .map(|sig| serde_json::json!({"signature": sig})),
                            ..Default::default()
                        });
                        self.reasoning_started = false;
                    }
                    // NOTE: Single boolean limits to one concurrent text segment. Current providers
                    // send text sequentially. For concurrent segments, replace with counter or
                    // segment-id tracking.
                    if !self.text_started {
                        self.text_started = true;
                        let text_id = format!("txt_{}", uuid::Uuid::new_v4());
                        self.current_text_id = Some(text_id.clone());
                        events.push(StreamEvent {
                            event_type: StreamEventType::TextStart,
                            text_id: Some(text_id),
                            ..Default::default()
                        });
                    }
                    events.push(StreamEvent {
                        event_type: StreamEventType::TextDelta,
                        delta: Some(text.to_string()),
                        text_id: self.current_text_id.clone(),
                        ..Default::default()
                    });
                }
            }
        }

        // Check for finishReason
        if let Some(finish_reason_str) = candidate.get("finishReason").and_then(|v| v.as_str()) {
            self.finished = true;

            // Emit ReasoningEnd if reasoning was streaming
            if self.reasoning_started {
                events.push(StreamEvent {
                    event_type: StreamEventType::ReasoningEnd,
                    raw: self
                        .active_thinking_signature
                        .take()
                        .map(|sig| serde_json::json!({"signature": sig})),
                    ..Default::default()
                });
                self.reasoning_started = false;
            }

            // Emit TextEnd if text was streaming
            if self.text_started {
                events.push(StreamEvent {
                    event_type: StreamEventType::TextEnd,
                    text_id: self.current_text_id.take(),
                    ..Default::default()
                });
                self.text_started = false;
            }

            // Check if any function calls were in this response
            let has_function_calls = candidate
                .get("content")
                .and_then(|c| c.get("parts"))
                .and_then(|p| p.as_array())
                .map(|parts| parts.iter().any(|p| p.get("functionCall").is_some()))
                .unwrap_or(false);

            let finish_reason = map_finish_reason(finish_reason_str, has_function_calls);

            // Parse usage from usageMetadata
            let usage_obj = data.get("usageMetadata");
            let input_tokens = usage_obj
                .and_then(|u| u.get("promptTokenCount"))
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32;
            let output_tokens = usage_obj
                .and_then(|u| u.get("candidatesTokenCount"))
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32;
            let reasoning_tokens = usage_obj
                .and_then(|u| u.get("thoughtsTokenCount"))
                .and_then(|v| v.as_u64())
                .map(|v| v as u32);
            let cache_read_tokens = usage_obj
                .and_then(|u| u.get("cachedContentTokenCount"))
                .and_then(|v| v.as_u64())
                .map(|v| v as u32);

            // M-11: Gemini does not expose cache write tokens (see parse_response comment).
            let cache_write_tokens: Option<u32> = None;

            let usage = Usage {
                input_tokens,
                output_tokens,
                // F-10: always recompute total_tokens
                total_tokens: input_tokens + output_tokens,
                reasoning_tokens,
                cache_read_tokens,
                cache_write_tokens,
                raw: usage_obj.cloned(),
            };

            events.push(StreamEvent {
                event_type: StreamEventType::Finish,
                finish_reason: Some(finish_reason),
                usage: Some(usage),
                ..Default::default()
            });
        }

        events
    }

    /// Called when the stream connection closes — emit final events if not already finished.
    fn finalize(&mut self) -> Vec<StreamEvent> {
        let mut events = Vec::new();
        if !self.finished && self.started {
            if self.reasoning_started {
                events.push(StreamEvent {
                    event_type: StreamEventType::ReasoningEnd,
                    raw: self
                        .active_thinking_signature
                        .take()
                        .map(|sig| serde_json::json!({"signature": sig})),
                    ..Default::default()
                });
            }
            if self.text_started {
                events.push(StreamEvent {
                    event_type: StreamEventType::TextEnd,
                    text_id: self.current_text_id.take(),
                    ..Default::default()
                });
            }
            events.push(StreamEvent {
                event_type: StreamEventType::Finish,
                finish_reason: Some(FinishReason::stop()),
                ..Default::default()
            });
        }
        events
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use unified_llm_types::*;

    // === Builder Tests (H-8) ===

    #[test]
    fn test_gemini_adapter_builder_defaults() {
        let adapter = GeminiAdapterBuilder::new(SecretString::from("key".to_string())).build();
        assert_eq!(adapter.base_url, DEFAULT_BASE_URL);
        assert_eq!(adapter.name(), "gemini");
    }

    #[test]
    fn test_gemini_adapter_builder_with_all_options() {
        let adapter = GeminiAdapterBuilder::new(SecretString::from("key".to_string()))
            .base_url("https://custom.gemini.com")
            .timeout(AdapterTimeout {
                connect: 5.0,
                request: 60.0,
                stream_read: 15.0,
            })
            .default_headers(reqwest::header::HeaderMap::new())
            .build();
        assert_eq!(adapter.base_url, "https://custom.gemini.com");
    }

    #[test]
    fn test_gemini_adapter_builder_shortcut() {
        let adapter = GeminiAdapter::builder(SecretString::from("key".to_string()))
            .base_url("https://test.com")
            .build();
        assert_eq!(adapter.base_url, "https://test.com");
    }

    // === T01: Adapter skeleton + constructors ===

    #[test]
    fn test_adapter_name() {
        let adapter = GeminiAdapter::new(SecretString::from("test-key".to_string()));
        assert_eq!(adapter.name(), "gemini");
    }

    #[test]
    fn test_new_sets_default_base_url() {
        let adapter = GeminiAdapter::new(SecretString::from("test-key".to_string()));
        assert_eq!(adapter.base_url, DEFAULT_BASE_URL);
    }

    #[test]
    fn test_new_with_base_url() {
        let adapter = GeminiAdapter::new_with_base_url(
            SecretString::from("test-key".to_string()),
            "http://localhost:8080",
        );
        assert_eq!(adapter.base_url, "http://localhost:8080");
    }

    #[test]
    fn test_build_headers_has_api_key() {
        let adapter = GeminiAdapter::new(SecretString::from("test-key-123".to_string()));
        let headers = adapter.build_headers().unwrap();
        assert_eq!(headers.get("x-goog-api-key").unwrap(), "test-key-123");
        assert_eq!(headers.get("content-type").unwrap(), "application/json");
    }

    #[test]
    fn test_build_url() {
        let adapter = GeminiAdapter::new(SecretString::from("k".to_string()));
        let url = adapter.build_url("gemini-2.0-flash");
        assert_eq!(
            url,
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        );
    }

    #[test]
    fn test_build_stream_url() {
        let adapter = GeminiAdapter::new(SecretString::from("k".to_string()));
        let url = adapter.build_stream_url("gemini-2.0-flash");
        assert_eq!(
            url,
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:streamGenerateContent?alt=sse"
        );
    }

    #[test]
    fn test_build_headers_rejects_invalid_key() {
        let adapter = GeminiAdapter::new(SecretString::from("bad\x00key".to_string()));
        let result = adapter.build_headers();
        assert!(result.is_err());
    }

    // === T02: Request translation — messages and system ===

    #[test]
    fn test_translate_request_system_to_system_instruction() {
        let request = Request::default().model("gemini-2.0-flash").messages(vec![
            Message::system("You are helpful."),
            Message::user("Hello"),
        ]);
        let (body, _) = translate_request(&request, None);

        let sys = &body["systemInstruction"];
        assert_eq!(sys["parts"][0]["text"], "You are helpful.");

        let contents = body["contents"].as_array().unwrap();
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0]["role"], "user");
        assert_eq!(contents[0]["parts"][0]["text"], "Hello");
    }

    #[test]
    fn test_translate_request_developer_merged_with_system() {
        let request = Request::default().model("gemini-2.0-flash").messages(vec![
            Message::system("System prompt."),
            Message {
                role: Role::Developer,
                content: vec![ContentPart::Text {
                    text: "Developer instructions.".to_string(),
                }],
                name: None,
                tool_call_id: None,
            },
            Message::user("Hi"),
        ]);
        let (body, _) = translate_request(&request, None);

        let sys_text = body["systemInstruction"]["parts"][0]["text"]
            .as_str()
            .unwrap();
        assert!(sys_text.contains("System prompt."));
        assert!(sys_text.contains("Developer instructions."));
    }

    #[test]
    fn test_translate_request_user_and_assistant_roles() {
        let request = Request::default().model("gemini-2.0-flash").messages(vec![
            Message::user("Hello"),
            Message::assistant("Hi there!"),
            Message::user("How are you?"),
        ]);
        let (body, _) = translate_request(&request, None);

        let contents = body["contents"].as_array().unwrap();
        assert_eq!(contents.len(), 3);
        assert_eq!(contents[0]["role"], "user");
        assert_eq!(contents[1]["role"], "model"); // NOT "assistant"
        assert_eq!(contents[2]["role"], "user");
        assert_eq!(contents[1]["parts"][0]["text"], "Hi there!");
    }

    #[test]
    fn test_translate_request_no_model_in_body() {
        let request = Request::default()
            .model("gemini-2.0-flash")
            .messages(vec![Message::user("Hi")]);
        let (body, _) = translate_request(&request, None);
        // Model goes in URL, not body
        assert!(body.get("model").is_none());
    }

    // === T03: Tool definitions and tool_choice ===

    #[test]
    fn test_translate_request_tools() {
        let request = Request::default()
            .model("gemini-2.0-flash")
            .messages(vec![Message::user("Weather?")])
            .tools(vec![ToolDefinition {
                name: "get_weather".to_string(),
                description: "Get weather".to_string(),
                parameters: serde_json::json!({"type": "object", "properties": {"city": {"type": "string"}}}),
                strict: None,
            }]);
        let (body, _) = translate_request(&request, None);

        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        let fn_decls = tools[0]["functionDeclarations"].as_array().unwrap();
        assert_eq!(fn_decls.len(), 1);
        assert_eq!(fn_decls[0]["name"], "get_weather");
        assert_eq!(fn_decls[0]["description"], "Get weather");
    }

    #[test]
    fn test_translate_request_tool_choice_auto() {
        let request = Request::default()
            .model("gemini-2.0-flash")
            .messages(vec![Message::user("Hi")])
            .tool_choice(ToolChoice {
                mode: "auto".to_string(),
                tool_name: None,
            });
        let (body, _) = translate_request(&request, None);

        assert_eq!(body["toolConfig"]["functionCallingConfig"]["mode"], "AUTO");
    }

    #[test]
    fn test_translate_request_tool_choice_none() {
        let request = Request::default()
            .model("gemini-2.0-flash")
            .messages(vec![Message::user("Hi")])
            .tool_choice(ToolChoice {
                mode: "none".to_string(),
                tool_name: None,
            });
        let (body, _) = translate_request(&request, None);
        assert_eq!(body["toolConfig"]["functionCallingConfig"]["mode"], "NONE");
    }

    #[test]
    fn test_translate_request_tool_choice_required() {
        let request = Request::default()
            .model("gemini-2.0-flash")
            .messages(vec![Message::user("Hi")])
            .tool_choice(ToolChoice {
                mode: "required".to_string(),
                tool_name: None,
            });
        let (body, _) = translate_request(&request, None);
        assert_eq!(body["toolConfig"]["functionCallingConfig"]["mode"], "ANY");
    }

    #[test]
    fn test_translate_request_tool_choice_named() {
        let request = Request::default()
            .model("gemini-2.0-flash")
            .messages(vec![Message::user("Hi")])
            .tool_choice(ToolChoice {
                mode: "named".to_string(),
                tool_name: Some("get_weather".to_string()),
            });
        let (body, _) = translate_request(&request, None);
        assert_eq!(body["toolConfig"]["functionCallingConfig"]["mode"], "ANY");
        let allowed = body["toolConfig"]["functionCallingConfig"]["allowedFunctionNames"]
            .as_array()
            .unwrap();
        assert_eq!(allowed[0], "get_weather");
    }

    // === T04: Image/multimodal parts ===

    #[test]
    fn test_translate_request_image_base64() {
        let request = Request::default()
            .model("gemini-2.0-flash")
            .messages(vec![Message {
                role: Role::User,
                content: vec![
                    ContentPart::Text {
                        text: "What's this?".to_string(),
                    },
                    ContentPart::Image {
                        image: ImageData {
                            url: None,
                            data: Some(vec![0x89, 0x50, 0x4E, 0x47]),
                            media_type: Some("image/png".to_string()),
                            detail: None,
                        },
                    },
                ],
                name: None,
                tool_call_id: None,
            }]);
        let (body, _) = translate_request(&request, None);

        let parts = body["contents"][0]["parts"].as_array().unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["text"], "What's this?");
        assert_eq!(parts[1]["inlineData"]["mimeType"], "image/png");
        assert!(parts[1]["inlineData"]["data"].as_str().unwrap().len() > 0);
    }

    #[test]
    fn test_translate_request_image_url() {
        let request = Request::default()
            .model("gemini-2.0-flash")
            .messages(vec![Message {
                role: Role::User,
                content: vec![ContentPart::Image {
                    image: ImageData {
                        url: Some("https://example.com/cat.jpg".to_string()),
                        data: None,
                        media_type: None,
                        detail: None,
                    },
                }],
                name: None,
                tool_call_id: None,
            }]);
        let (body, _) = translate_request(&request, None);

        let parts = body["contents"][0]["parts"].as_array().unwrap();
        assert_eq!(
            parts[0]["fileData"]["fileUri"],
            "https://example.com/cat.jpg"
        );
        // Should infer MIME type from URL extension
        assert_eq!(parts[0]["fileData"]["mimeType"], "image/jpeg");
    }

    // === T04: Tool calls and tool results in messages ===

    #[test]
    fn test_translate_request_assistant_tool_calls() {
        let request = Request::default()
            .model("gemini-2.0-flash")
            .messages(vec![Message {
                role: Role::Assistant,
                content: vec![ContentPart::ToolCall {
                    tool_call: ToolCallData {
                        id: "call_123".to_string(),
                        name: "get_weather".to_string(),
                        arguments: ArgumentValue::Dict(
                            serde_json::from_str(r#"{"city": "NYC"}"#).unwrap(),
                        ),
                        r#type: "function".to_string(),
                    },
                }],
                name: None,
                tool_call_id: None,
            }]);
        let (body, _) = translate_request(&request, None);

        let contents = body["contents"].as_array().unwrap();
        assert_eq!(contents[0]["role"], "model");
        let fc = &contents[0]["parts"][0]["functionCall"];
        assert_eq!(fc["name"], "get_weather");
        assert_eq!(fc["args"]["city"], "NYC");
    }

    #[test]
    fn test_translate_request_tool_results() {
        let request = Request::default()
            .model("gemini-2.0-flash")
            .messages(vec![Message::tool_result("get_weather", "72°F", false)]);
        let (body, _) = translate_request(&request, None);

        let contents = body["contents"].as_array().unwrap();
        assert_eq!(contents[0]["role"], "user");
        let fr = &contents[0]["parts"][0]["functionResponse"];
        assert_eq!(fr["name"], "get_weather");
        // String results are wrapped in {"result": "..."}
        assert_eq!(fr["response"]["result"], "72°F");
    }

    #[test]
    fn test_translate_request_tool_result_object_not_wrapped() {
        // Build tool result message manually with object content
        let request = Request::default()
            .model("gemini-2.0-flash")
            .messages(vec![Message {
                role: Role::Tool,
                content: vec![ContentPart::ToolResult {
                    tool_result: ToolResultData {
                        tool_call_id: "get_weather".to_string(),
                        content: serde_json::json!({"temp": 72, "unit": "F"}),
                        is_error: false,
                        image_data: None,
                        image_media_type: None,
                    },
                }],
                name: None,
                tool_call_id: None,
            }]);
        let (body, _) = translate_request(&request, None);

        let fr = &body["contents"][0]["parts"][0]["functionResponse"];
        assert_eq!(fr["response"]["temp"], 72);
        assert_eq!(fr["response"]["unit"], "F");
    }

    #[test]
    fn test_translate_request_tool_result_error_flag_preserved() {
        // When a tool result has is_error: true, the error flag must be
        // preserved in the Gemini functionResponse format.
        let request = Request::default()
            .model("gemini-2.0-flash")
            .messages(vec![Message {
                role: Role::Tool,
                content: vec![ContentPart::ToolResult {
                    tool_result: ToolResultData {
                        tool_call_id: "get_weather".to_string(),
                        content: serde_json::json!({"error": "City not found"}),
                        is_error: true,
                        image_data: None,
                        image_media_type: None,
                    },
                }],
                name: None,
                tool_call_id: None,
            }]);
        let (body, _) = translate_request(&request, None);

        let fr = &body["contents"][0]["parts"][0]["functionResponse"];
        assert_eq!(fr["name"], "get_weather");
        // The error flag must be present in the response
        assert_eq!(
            fr["response"]["is_error"],
            true,
            "is_error flag must be preserved in functionResponse. Got: {}",
            serde_json::to_string_pretty(&fr["response"]).unwrap()
        );
    }

    // === T05: Generation params ===

    #[test]
    fn test_translate_request_generation_config() {
        let request = Request::default()
            .model("gemini-2.0-flash")
            .messages(vec![Message::user("Hi")])
            .max_tokens(1000)
            .temperature(0.7)
            .top_p(0.9)
            .stop_sequences(vec!["STOP".to_string()]);
        let (body, _) = translate_request(&request, None);

        let gc = &body["generationConfig"];
        assert_eq!(gc["maxOutputTokens"], 1000);
        assert_eq!(gc["temperature"], 0.7);
        assert_eq!(gc["topP"], 0.9);
        assert_eq!(gc["stopSequences"][0], "STOP");
    }

    #[test]
    fn test_translate_request_structured_output() {
        let request = Request::default()
            .model("gemini-2.0-flash")
            .messages(vec![Message::user("Hi")])
            .response_format(ResponseFormat {
                r#type: "json_schema".to_string(),
                json_schema: Some(serde_json::json!({"type": "object", "properties": {"name": {"type": "string"}}})),
                strict: true,
            });
        let (body, _) = translate_request(&request, None);

        let gc = &body["generationConfig"];
        assert_eq!(gc["responseMimeType"], "application/json");
        assert!(gc["responseSchema"].is_object());
    }

    #[test]
    fn test_translate_request_thinking_config() {
        let request = Request::default()
            .model("gemini-2.5-flash")
            .messages(vec![Message::user("Think about this")])
            .provider_options(Some(serde_json::json!({
                "gemini": {
                    "thinkingConfig": {
                        "thinkingLevel": "HIGH",
                        "includeThoughts": true,
                    }
                }
            })));
        let (body, _) = translate_request(&request, None);

        // thinkingConfig must be at the request root, NOT inside generationConfig
        assert_eq!(body["thinkingConfig"]["thinkingLevel"], "HIGH");
        assert_eq!(body["thinkingConfig"]["includeThoughts"], true);
        // Must NOT be nested inside generationConfig
        assert!(
            body.get("generationConfig")
                .and_then(|gc| gc.get("thinkingConfig"))
                .is_none(),
            "thinkingConfig must NOT be inside generationConfig"
        );
    }

    #[test]
    fn test_translate_request_provider_options_passthrough() {
        let request = Request::default()
            .model("gemini-2.0-flash")
            .messages(vec![Message::user("Hi")])
            .provider_options(Some(serde_json::json!({
                "gemini": {
                    "safetySettings": [{"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"}],
                    "thinkingConfig": {"thinkingLevel": "HIGH"},
                }
            })));
        let (body, _) = translate_request(&request, None);

        // safetySettings should be passed through
        assert!(body.get("safetySettings").is_some());
        // thinkingConfig must be at request root level (not inside generationConfig)
        assert_eq!(body["thinkingConfig"]["thinkingLevel"], "HIGH");
        // Must NOT be nested inside generationConfig
        assert!(
            body.get("generationConfig")
                .and_then(|gc| gc.get("thinkingConfig"))
                .is_none(),
            "thinkingConfig must NOT be inside generationConfig"
        );
    }

    // === T06: Response parsing ===

    #[test]
    fn test_parse_response_simple_text() {
        let raw = serde_json::json!({
            "responseId": "resp_123",
            "modelVersion": "gemini-2.0-flash",
            "candidates": [{
                "content": {
                    "parts": [{"text": "Hello!"}],
                    "role": "model"
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 2,
                "totalTokenCount": 7
            }
        });

        let resp = parse_response(raw, &reqwest::header::HeaderMap::new()).unwrap();
        assert_eq!(resp.id, "resp_123");
        assert_eq!(resp.model, "gemini-2.0-flash");
        assert_eq!(resp.provider, "gemini");
        assert_eq!(resp.text(), "Hello!");
        assert_eq!(resp.finish_reason.reason, "stop");
        assert_eq!(resp.finish_reason.raw, Some("STOP".to_string()));
        assert_eq!(resp.usage.input_tokens, 5);
        assert_eq!(resp.usage.output_tokens, 2);
        assert_eq!(resp.usage.total_tokens, 7);
    }

    #[test]
    fn test_parse_response_tool_call_with_id() {
        let raw = serde_json::json!({
            "candidates": [{
                "content": {
                    "parts": [{
                        "functionCall": {
                            "name": "get_weather",
                            "args": {"city": "NYC"},
                            "id": "server_id_1"
                        }
                    }],
                    "role": "model"
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15}
        });

        let resp = parse_response(raw, &reqwest::header::HeaderMap::new()).unwrap();
        assert_eq!(resp.finish_reason.reason, "tool_calls");
        let tool_calls = resp.tool_calls();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "server_id_1");
        assert_eq!(tool_calls[0].name, "get_weather");
    }

    #[test]
    fn test_parse_response_tool_call_without_id_generates_synthetic() {
        let raw = serde_json::json!({
            "candidates": [{
                "content": {
                    "parts": [{
                        "functionCall": {
                            "name": "get_weather",
                            "args": {"city": "NYC"}
                        }
                    }],
                    "role": "model"
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15}
        });

        let resp = parse_response(raw, &reqwest::header::HeaderMap::new()).unwrap();
        let tool_calls = resp.tool_calls();
        assert_eq!(tool_calls.len(), 1);
        assert!(tool_calls[0].id.starts_with("call_"));
        assert_eq!(tool_calls[0].name, "get_weather");
    }

    #[test]
    fn test_parse_response_thinking_part() {
        let raw = serde_json::json!({
            "candidates": [{
                "content": {
                    "parts": [
                        {"thought": true, "text": "Let me think...", "thoughtSignature": "sig_abc"},
                        {"text": "The answer is 42."}
                    ],
                    "role": "model"
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 10,
                "totalTokenCount": 15,
                "thoughtsTokenCount": 8
            }
        });

        let resp = parse_response(raw, &reqwest::header::HeaderMap::new()).unwrap();
        assert_eq!(resp.usage.reasoning_tokens, Some(8));

        // First part should be Thinking
        match &resp.message.content[0] {
            ContentPart::Thinking { thinking } => {
                assert_eq!(thinking.text, "Let me think...");
                assert_eq!(thinking.signature, Some("sig_abc".to_string()));
            }
            _ => panic!("Expected Thinking content part"),
        }
        // Second part should be Text
        assert_eq!(resp.text(), "The answer is 42.");
    }

    #[test]
    fn test_parse_response_finish_reason_mapping() {
        // STOP → stop
        assert_eq!(map_finish_reason("STOP", false).reason, "stop");
        // MAX_TOKENS → length
        assert_eq!(map_finish_reason("MAX_TOKENS", false).reason, "length");
        // SAFETY → content_filter
        assert_eq!(map_finish_reason("SAFETY", false).reason, "content_filter");
        // RECITATION → content_filter
        assert_eq!(
            map_finish_reason("RECITATION", false).reason,
            "content_filter"
        );
        // has_function_calls overrides
        assert_eq!(map_finish_reason("STOP", true).reason, "tool_calls");
        // Unknown → other
        assert_eq!(map_finish_reason("SOMETHING_ELSE", false).reason, "other");
    }

    #[test]
    fn test_parse_response_usage_with_cache_tokens() {
        let raw = serde_json::json!({
            "candidates": [{
                "content": {"parts": [{"text": "Hi"}], "role": "model"},
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 100,
                "candidatesTokenCount": 10,
                "totalTokenCount": 110,
                "cachedContentTokenCount": 50
            }
        });

        let resp = parse_response(raw, &reqwest::header::HeaderMap::new()).unwrap();
        assert_eq!(resp.usage.cache_read_tokens, Some(50));
    }

    // --- M-11: cache_write_tokens is always None (Gemini provider limitation) ---

    #[test]
    fn test_cache_write_tokens_always_none() {
        // M-11: Gemini does not expose cache write tokens in its API.
        // cachedContentTokenCount maps to cache_read_tokens only.
        let raw = serde_json::json!({
            "candidates": [{
                "content": {"parts": [{"text": "Hi"}], "role": "model"},
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 100,
                "candidatesTokenCount": 10,
                "totalTokenCount": 110,
                "cachedContentTokenCount": 50
            }
        });

        let resp = parse_response(raw, &reqwest::header::HeaderMap::new()).unwrap();
        assert_eq!(resp.usage.cache_read_tokens, Some(50));
        assert_eq!(
            resp.usage.cache_write_tokens, None,
            "M-11: Gemini does not provide cache write tokens"
        );
    }

    #[test]
    fn test_cache_write_tokens_none_without_any_cache_fields() {
        // Even when no cache fields are present, cache_write_tokens is None
        let raw = serde_json::json!({
            "candidates": [{
                "content": {"parts": [{"text": "Hi"}], "role": "model"},
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15
            }
        });

        let resp = parse_response(raw, &reqwest::header::HeaderMap::new()).unwrap();
        assert_eq!(resp.usage.cache_read_tokens, None);
        assert_eq!(resp.usage.cache_write_tokens, None);
    }

    #[test]
    fn test_stream_cache_write_tokens_always_none() {
        // M-11: Streaming also has cache_write_tokens as None
        let mut translator = GeminiStreamTranslator::new();
        let chunk = serde_json::json!({
            "candidates": [{
                "content": {"parts": [{"text": "Hi"}], "role": "model"},
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 100,
                "candidatesTokenCount": 10,
                "totalTokenCount": 110,
                "cachedContentTokenCount": 50
            }
        });

        let events = translator.process(&chunk);
        let finish = events
            .iter()
            .find(|e| e.event_type == StreamEventType::Finish)
            .unwrap();
        let usage = finish.usage.as_ref().unwrap();
        assert_eq!(usage.cache_read_tokens, Some(50));
        assert_eq!(
            usage.cache_write_tokens, None,
            "M-11: Gemini streaming does not provide cache write tokens"
        );
    }

    // === T08: Error translation ===

    #[test]
    fn test_parse_error_400() {
        let body = serde_json::json!({
            "error": {
                "code": 400,
                "message": "Invalid value at 'contents'",
                "status": "INVALID_ARGUMENT"
            }
        });
        let err = parse_error(400, &reqwest::header::HeaderMap::new(), body);
        assert_eq!(err.kind, ErrorKind::InvalidRequest);
        assert!(err.message.contains("Invalid value"));
        assert_eq!(err.error_code, Some("INVALID_ARGUMENT".to_string()));
        assert_eq!(err.provider, Some("gemini".to_string()));
    }

    #[test]
    fn test_parse_error_400_api_key_not_valid_maps_to_authentication() {
        // Gemini returns HTTP 400 (not 401) with INVALID_ARGUMENT when the API
        // key is syntactically wrong.  Message-based reclassification in
        // Error::from_http_status should promote this to Authentication.
        let body = serde_json::json!({
            "error": {
                "code": 400,
                "message": "API key not valid. Please pass a valid API key.",
                "status": "INVALID_ARGUMENT"
            }
        });
        let err = parse_error(400, &reqwest::header::HeaderMap::new(), body);
        assert_eq!(
            err.kind,
            ErrorKind::Authentication,
            "400 + 'API key not valid' should be reclassified to Authentication"
        );
        assert!(!err.retryable);
        assert_eq!(err.error_code, Some("INVALID_ARGUMENT".to_string()));
    }

    #[test]
    fn test_parse_error_401() {
        let body = serde_json::json!({
            "error": {
                "code": 401,
                "message": "API key not valid",
                "status": "UNAUTHENTICATED"
            }
        });
        let err = parse_error(401, &reqwest::header::HeaderMap::new(), body);
        assert_eq!(err.kind, ErrorKind::Authentication);
    }

    #[test]
    fn test_parse_error_403() {
        let body = serde_json::json!({
            "error": {
                "code": 403,
                "message": "Permission denied",
                "status": "PERMISSION_DENIED"
            }
        });
        let err = parse_error(403, &reqwest::header::HeaderMap::new(), body);
        assert_eq!(err.kind, ErrorKind::AccessDenied);
    }

    #[test]
    fn test_parse_error_404() {
        let body = serde_json::json!({
            "error": {
                "code": 404,
                "message": "Model not found",
                "status": "NOT_FOUND"
            }
        });
        let err = parse_error(404, &reqwest::header::HeaderMap::new(), body);
        assert_eq!(err.kind, ErrorKind::NotFound);
    }

    #[test]
    fn test_parse_error_429_with_retry_after() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("retry-after", "30".parse().unwrap());
        let body = serde_json::json!({
            "error": {
                "code": 429,
                "message": "Resource exhausted",
                "status": "RESOURCE_EXHAUSTED"
            }
        });
        let err = parse_error(429, &headers, body);
        assert_eq!(err.kind, ErrorKind::RateLimit);
        assert!(err.retryable);
        assert_eq!(err.retry_after, Some(std::time::Duration::from_secs(30)));
    }

    #[test]
    fn test_parse_error_500() {
        let body = serde_json::json!({
            "error": {
                "code": 500,
                "message": "Internal error",
                "status": "INTERNAL"
            }
        });
        let err = parse_error(500, &reqwest::header::HeaderMap::new(), body);
        assert_eq!(err.kind, ErrorKind::Server);
        assert!(err.retryable);
    }

    #[test]
    fn test_parse_error_preserves_raw() {
        let body = serde_json::json!({
            "error": {"code": 400, "message": "Bad request", "status": "INVALID_ARGUMENT"}
        });
        let err = parse_error(400, &reqwest::header::HeaderMap::new(), body.clone());
        assert_eq!(err.raw, Some(body));
    }

    // === F-5: gRPC status code override tests ===

    #[test]
    fn test_parse_error_grpc_resource_exhausted_overrides_http_status() {
        let headers = reqwest::header::HeaderMap::new();
        // Gemini sometimes returns 400 with RESOURCE_EXHAUSTED gRPC status
        let body = serde_json::json!({
            "error": {
                "message": "Resource exhausted",
                "status": "RESOURCE_EXHAUSTED"
            }
        });
        let err = parse_error(400, &headers, body);
        assert_eq!(
            err.kind,
            ErrorKind::RateLimit,
            "gRPC RESOURCE_EXHAUSTED should map to RateLimit regardless of HTTP status"
        );
    }

    #[test]
    fn test_parse_error_grpc_deadline_exceeded_maps_to_request_timeout() {
        let headers = reqwest::header::HeaderMap::new();
        let body = serde_json::json!({
            "error": {
                "message": "Deadline exceeded",
                "status": "DEADLINE_EXCEEDED"
            }
        });
        let err = parse_error(504, &headers, body);
        assert_eq!(err.kind, ErrorKind::RequestTimeout);
    }

    #[test]
    fn test_parse_error_grpc_permission_denied_maps_to_access_denied() {
        let headers = reqwest::header::HeaderMap::new();
        let body = serde_json::json!({
            "error": {
                "message": "Permission denied",
                "status": "PERMISSION_DENIED"
            }
        });
        let err = parse_error(403, &headers, body);
        assert_eq!(err.kind, ErrorKind::AccessDenied);
    }

    #[test]
    fn test_parse_error_grpc_unauthenticated_maps_to_authentication() {
        let headers = reqwest::header::HeaderMap::new();
        let body = serde_json::json!({
            "error": {
                "message": "Invalid API key",
                "status": "UNAUTHENTICATED"
            }
        });
        let err = parse_error(401, &headers, body);
        assert_eq!(err.kind, ErrorKind::Authentication);
    }

    #[test]
    fn test_parse_error_grpc_not_found_maps_to_not_found() {
        let headers = reqwest::header::HeaderMap::new();
        let body = serde_json::json!({
            "error": {
                "message": "Model not found",
                "status": "NOT_FOUND"
            }
        });
        let err = parse_error(404, &headers, body);
        assert_eq!(err.kind, ErrorKind::NotFound);
    }

    // === T09: Stream translation ===

    #[test]
    fn test_stream_text_basic() {
        let mut translator = GeminiStreamTranslator::new();

        let chunk1 = serde_json::json!({
            "responseId": "resp_s1",
            "modelVersion": "gemini-2.0-flash",
            "candidates": [{
                "content": {"parts": [{"text": "Hello"}], "role": "model"}
            }]
        });

        let events = translator.process(&chunk1);
        assert_eq!(events[0].event_type, StreamEventType::StreamStart);
        assert_eq!(events[0].id, Some("resp_s1".to_string()));
        assert_eq!(events[1].event_type, StreamEventType::TextStart);
        assert_eq!(events[2].event_type, StreamEventType::TextDelta);
        assert_eq!(events[2].delta, Some("Hello".to_string()));
    }

    #[test]
    fn test_stream_text_multiple_chunks() {
        let mut translator = GeminiStreamTranslator::new();

        let chunk1 = serde_json::json!({
            "candidates": [{
                "content": {"parts": [{"text": "Hello"}], "role": "model"}
            }]
        });
        let chunk2 = serde_json::json!({
            "candidates": [{
                "content": {"parts": [{"text": " world"}], "role": "model"}
            }]
        });
        let chunk3 = serde_json::json!({
            "candidates": [{
                "content": {"parts": [{"text": "!"}], "role": "model"},
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 3,
                "totalTokenCount": 8
            }
        });

        let events1 = translator.process(&chunk1);
        assert!(events1
            .iter()
            .any(|e| e.event_type == StreamEventType::StreamStart));
        assert!(events1
            .iter()
            .any(|e| e.event_type == StreamEventType::TextStart));

        let events2 = translator.process(&chunk2);
        assert_eq!(events2.len(), 1);
        assert_eq!(events2[0].event_type, StreamEventType::TextDelta);
        assert_eq!(events2[0].delta, Some(" world".to_string()));

        let events3 = translator.process(&chunk3);
        // Should have: TextDelta("!"), TextEnd, Finish
        let types: Vec<_> = events3.iter().map(|e| &e.event_type).collect();
        assert!(types.contains(&&StreamEventType::TextDelta));
        assert!(types.contains(&&StreamEventType::TextEnd));
        assert!(types.contains(&&StreamEventType::Finish));

        let finish = events3
            .iter()
            .find(|e| e.event_type == StreamEventType::Finish)
            .unwrap();
        assert_eq!(finish.finish_reason.as_ref().unwrap().reason, "stop");
        assert_eq!(finish.usage.as_ref().unwrap().input_tokens, 5);
        assert_eq!(finish.usage.as_ref().unwrap().output_tokens, 3);
    }

    #[test]
    fn test_stream_tool_call_complete_in_one_chunk() {
        let mut translator = GeminiStreamTranslator::new();

        let chunk = serde_json::json!({
            "candidates": [{
                "content": {
                    "parts": [{
                        "functionCall": {
                            "name": "get_weather",
                            "args": {"city": "NYC"}
                        }
                    }],
                    "role": "model"
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15}
        });

        let events = translator.process(&chunk);
        let types: Vec<_> = events.iter().map(|e| &e.event_type).collect();

        // Should have: StreamStart, ToolCallStart, ToolCallEnd, Finish
        assert!(types.contains(&&StreamEventType::StreamStart));
        assert!(types.contains(&&StreamEventType::ToolCallStart));
        assert!(types.contains(&&StreamEventType::ToolCallEnd));
        assert!(types.contains(&&StreamEventType::Finish));

        // ToolCallStart should have the complete tool_call
        let start = events
            .iter()
            .find(|e| e.event_type == StreamEventType::ToolCallStart)
            .unwrap();
        let tc = start.tool_call.as_ref().unwrap();
        assert_eq!(tc.name, "get_weather");
        assert!(tc.id.starts_with("call_"));

        // Finish reason should be tool_calls
        let finish = events
            .iter()
            .find(|e| e.event_type == StreamEventType::Finish)
            .unwrap();
        assert_eq!(finish.finish_reason.as_ref().unwrap().reason, "tool_calls");
    }

    #[test]
    fn test_stream_finalize_without_finish_reason() {
        let mut translator = GeminiStreamTranslator::new();

        let chunk = serde_json::json!({
            "candidates": [{
                "content": {"parts": [{"text": "Hello"}], "role": "model"}
            }]
        });

        translator.process(&chunk);
        let final_events = translator.finalize();

        let types: Vec<_> = final_events.iter().map(|e| &e.event_type).collect();
        assert!(types.contains(&&StreamEventType::TextEnd));
        assert!(types.contains(&&StreamEventType::Finish));
    }

    #[test]
    fn test_stream_finalize_noop_when_already_finished() {
        let mut translator = GeminiStreamTranslator::new();

        let chunk = serde_json::json!({
            "candidates": [{
                "content": {"parts": [{"text": "Hi"}], "role": "model"},
                "finishReason": "STOP"
            }],
            "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2}
        });

        translator.process(&chunk);
        let final_events = translator.finalize();
        assert!(final_events.is_empty());
    }

    // === Wiremock integration tests ===

    #[tokio::test]
    async fn test_gemini_complete_roundtrip() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path_regex(
                r"/v1beta/models/.+:generateContent",
            ))
            .and(wiremock::matchers::header_exists("x-goog-api-key"))
            .respond_with(
                wiremock::ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "responseId": "resp_test",
                    "modelVersion": "gemini-2.0-flash",
                    "candidates": [{
                        "content": {"parts": [{"text": "Gemini says hello!"}], "role": "model"},
                        "finishReason": "STOP"
                    }],
                    "usageMetadata": {
                        "promptTokenCount": 5,
                        "candidatesTokenCount": 4,
                        "totalTokenCount": 9
                    }
                })),
            )
            .mount(&server)
            .await;

        let adapter = GeminiAdapter::new_with_base_url(
            SecretString::from("test-key".to_string()),
            server.uri(),
        );
        let resp = adapter
            .do_complete(
                Request::default()
                    .model("gemini-2.0-flash")
                    .messages(vec![Message::user("Hello")]),
            )
            .await
            .unwrap();

        assert_eq!(resp.text(), "Gemini says hello!");
        assert_eq!(resp.provider, "gemini");
        assert_eq!(resp.usage.total_tokens, 9);
    }

    #[tokio::test]
    async fn test_gemini_error_roundtrip() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .respond_with(
                wiremock::ResponseTemplate::new(429).set_body_json(serde_json::json!({
                    "error": {
                        "code": 429,
                        "message": "Resource has been exhausted",
                        "status": "RESOURCE_EXHAUSTED"
                    }
                })),
            )
            .mount(&server)
            .await;

        let adapter = GeminiAdapter::new_with_base_url(
            SecretString::from("test-key".to_string()),
            server.uri(),
        );
        let err = adapter
            .do_complete(
                Request::default()
                    .model("gemini-2.0-flash")
                    .messages(vec![Message::user("Hello")]),
            )
            .await
            .unwrap_err();

        assert_eq!(err.kind, ErrorKind::RateLimit);
        assert!(err.message.contains("exhausted"));
    }

    #[tokio::test]
    async fn test_gemini_request_body_structure() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .respond_with(
                wiremock::ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "candidates": [{
                        "content": {"parts": [{"text": "ok"}], "role": "model"},
                        "finishReason": "STOP"
                    }],
                    "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2}
                })),
            )
            .mount(&server)
            .await;

        let adapter = GeminiAdapter::new_with_base_url(
            SecretString::from("test-key".to_string()),
            server.uri(),
        );
        adapter
            .do_complete(
                Request::default()
                    .model("gemini-2.0-flash")
                    .messages(vec![Message::system("Be helpful"), Message::user("Hi")])
                    .temperature(0.5),
            )
            .await
            .unwrap();

        // Verify the request was made to the right path
        let requests = server.received_requests().await.unwrap();
        assert_eq!(requests.len(), 1);
        assert!(requests[0]
            .url
            .path()
            .contains("gemini-2.0-flash:generateContent"));

        // Verify x-goog-api-key header (NOT in URL)
        assert_eq!(
            requests[0]
                .headers
                .get("x-goog-api-key")
                .unwrap()
                .to_str()
                .unwrap(),
            "test-key"
        );
        assert!(!requests[0].url.to_string().contains("test-key"));

        // Verify body structure
        let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
        assert!(body.get("systemInstruction").is_some());
        assert!(body.get("contents").is_some());
        assert!(body.get("generationConfig").is_some());
        // model should NOT be in body
        assert!(body.get("model").is_none());
    }

    #[tokio::test]
    async fn test_gemini_stream_roundtrip() {
        let sse_body = [
            "data: {\"responseId\":\"resp_s1\",\"modelVersion\":\"gemini-2.0-flash\",\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hello\"}],\"role\":\"model\"}}]}\n\n",
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\" world\"}],\"role\":\"model\"}}]}\n\n",
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"!\"}],\"role\":\"model\"},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":5,\"candidatesTokenCount\":3,\"totalTokenCount\":8}}\n\n",
        ].join("");

        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path_regex(
                r"/v1beta/models/.+:streamGenerateContent",
            ))
            .respond_with(
                wiremock::ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse_body),
            )
            .mount(&server)
            .await;

        let adapter = GeminiAdapter::new_with_base_url(
            SecretString::from("test-key".to_string()),
            server.uri(),
        );

        let stream = adapter.do_stream(
            Request::default()
                .model("gemini-2.0-flash")
                .messages(vec![Message::user("Hello")]),
        );

        let events: Vec<StreamEvent> = stream
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .filter_map(|r| r.ok())
            .collect();

        let types: Vec<_> = events.iter().map(|e| &e.event_type).collect();
        assert!(types.contains(&&StreamEventType::StreamStart));
        assert!(types.contains(&&StreamEventType::TextStart));
        assert!(types.contains(&&StreamEventType::TextDelta));
        assert!(types.contains(&&StreamEventType::TextEnd));
        assert!(types.contains(&&StreamEventType::Finish));

        // Verify text deltas
        let deltas: Vec<_> = events
            .iter()
            .filter(|e| e.event_type == StreamEventType::TextDelta)
            .filter_map(|e| e.delta.as_deref())
            .collect();
        assert_eq!(deltas, vec!["Hello", " world", "!"]);

        // Verify finish
        let finish = events
            .iter()
            .find(|e| e.event_type == StreamEventType::Finish)
            .unwrap();
        assert_eq!(finish.finish_reason.as_ref().unwrap().reason, "stop");
        assert_eq!(finish.usage.as_ref().unwrap().total_tokens, 8);
    }

    #[tokio::test]
    async fn test_gemini_stream_error_response() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .respond_with(
                wiremock::ResponseTemplate::new(401).set_body_json(serde_json::json!({
                    "error": {
                        "code": 401,
                        "message": "API key not valid",
                        "status": "UNAUTHENTICATED"
                    }
                })),
            )
            .mount(&server)
            .await;

        let adapter = GeminiAdapter::new_with_base_url(
            SecretString::from("bad-key".to_string()),
            server.uri(),
        );

        let stream = adapter.do_stream(
            Request::default()
                .model("gemini-2.0-flash")
                .messages(vec![Message::user("Hello")]),
        );

        let events: Vec<Result<StreamEvent, Error>> = stream.collect::<Vec<_>>().await;
        assert_eq!(events.len(), 1);
        let err = events[0].as_ref().unwrap_err();
        assert_eq!(err.kind, ErrorKind::Authentication);
    }

    #[tokio::test]
    async fn test_gemini_stream_tool_call() {
        let sse_body = "data: {\"responseId\":\"resp_tc\",\"candidates\":[{\"content\":{\"parts\":[{\"functionCall\":{\"name\":\"get_weather\",\"args\":{\"city\":\"NYC\"}}}],\"role\":\"model\"},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":10,\"candidatesTokenCount\":5,\"totalTokenCount\":15}}\n\n";

        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .respond_with(
                wiremock::ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse_body),
            )
            .mount(&server)
            .await;

        let adapter = GeminiAdapter::new_with_base_url(
            SecretString::from("test-key".to_string()),
            server.uri(),
        );

        let stream = adapter.do_stream(
            Request::default()
                .model("gemini-2.0-flash")
                .messages(vec![Message::user("Weather in NYC?")]),
        );

        let events: Vec<StreamEvent> = stream
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .filter_map(|r| r.ok())
            .collect();

        let types: Vec<_> = events.iter().map(|e| &e.event_type).collect();
        assert!(types.contains(&&StreamEventType::ToolCallStart));
        assert!(types.contains(&&StreamEventType::ToolCallEnd));
        assert!(types.contains(&&StreamEventType::Finish));

        let tc_start = events
            .iter()
            .find(|e| e.event_type == StreamEventType::ToolCallStart)
            .unwrap();
        assert_eq!(tc_start.tool_call.as_ref().unwrap().name, "get_weather");

        let finish = events
            .iter()
            .find(|e| e.event_type == StreamEventType::Finish)
            .unwrap();
        assert_eq!(finish.finish_reason.as_ref().unwrap().reason, "tool_calls");
    }

    #[tokio::test]
    async fn test_gemini_tool_call_id_stored_in_map() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .respond_with(
                wiremock::ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "candidates": [{
                        "content": {
                            "parts": [{
                                "functionCall": {"name": "get_weather", "args": {"city": "NYC"}}
                            }],
                            "role": "model"
                        },
                        "finishReason": "STOP"
                    }],
                    "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15}
                })),
            )
            .mount(&server)
            .await;

        let adapter = GeminiAdapter::new_with_base_url(
            SecretString::from("test-key".to_string()),
            server.uri(),
        );

        let resp = adapter
            .do_complete(
                Request::default()
                    .model("gemini-2.0-flash")
                    .messages(vec![Message::user("Weather?")]),
            )
            .await
            .unwrap();

        let tc = &resp.tool_calls()[0];
        let map = adapter.tool_call_map.lock().unwrap();
        assert_eq!(map.get(&tc.id), Some(&"get_weather".to_string()));
    }

    #[test]
    fn test_stream_thinking_has_reasoning_start_and_end() {
        let mut translator = GeminiStreamTranslator::new();

        // Chunk 1: thinking part
        let chunk1 = serde_json::json!({
            "candidates": [{
                "content": {"parts": [{"text": "Let me think...", "thought": true}], "role": "model"}
            }]
        });

        // Chunk 2: more thinking
        let chunk2 = serde_json::json!({
            "candidates": [{
                "content": {"parts": [{"text": "Analyzing data...", "thought": true}], "role": "model"}
            }]
        });

        // Chunk 3: regular text + finish
        let chunk3 = serde_json::json!({
            "candidates": [{
                "content": {"parts": [{"text": "The answer is 42."}], "role": "model"},
                "finishReason": "STOP"
            }],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 10, "totalTokenCount": 15}
        });

        let events1 = translator.process(&chunk1);
        let types1: Vec<_> = events1.iter().map(|e| &e.event_type).collect();
        assert!(types1.contains(&&StreamEventType::StreamStart));
        assert!(types1.contains(&&StreamEventType::ReasoningStart));
        assert!(types1.contains(&&StreamEventType::ReasoningDelta));

        let events2 = translator.process(&chunk2);
        let types2: Vec<_> = events2.iter().map(|e| &e.event_type).collect();
        // Second chunk should NOT have another ReasoningStart
        assert!(!types2.contains(&&StreamEventType::ReasoningStart));
        assert!(types2.contains(&&StreamEventType::ReasoningDelta));

        let events3 = translator.process(&chunk3);
        let types3: Vec<_> = events3.iter().map(|e| &e.event_type).collect();
        // Transition to text: ReasoningEnd, then TextStart, TextDelta, TextEnd, Finish
        assert!(types3.contains(&&StreamEventType::ReasoningEnd));
        assert!(types3.contains(&&StreamEventType::TextStart));
        assert!(types3.contains(&&StreamEventType::TextDelta));
        assert!(types3.contains(&&StreamEventType::TextEnd));
        assert!(types3.contains(&&StreamEventType::Finish));

        // ReasoningEnd must come BEFORE TextStart
        let re_pos = types3
            .iter()
            .position(|t| **t == StreamEventType::ReasoningEnd)
            .unwrap();
        let ts_pos = types3
            .iter()
            .position(|t| **t == StreamEventType::TextStart)
            .unwrap();
        assert!(re_pos < ts_pos);
    }

    #[test]
    fn test_stream_finalize_emits_reasoning_end() {
        // If stream ends during reasoning (no finishReason), finalize() should emit ReasoningEnd
        let mut translator = GeminiStreamTranslator::new();

        let chunk = serde_json::json!({
            "candidates": [{
                "content": {"parts": [{"text": "thinking...", "thought": true}], "role": "model"}
            }]
        });

        translator.process(&chunk);
        let final_events = translator.finalize();
        let types: Vec<_> = final_events.iter().map(|e| &e.event_type).collect();
        assert!(types.contains(&&StreamEventType::ReasoningEnd));
        assert!(types.contains(&&StreamEventType::Finish));
        // No TextEnd since text was never started
        assert!(!types.contains(&&StreamEventType::TextEnd));
    }

    #[test]
    fn test_translate_request_tool_result_resolves_via_map() {
        // When a tool_call_map is provided, tool_result.tool_call_id should be
        // looked up to find the real function name for functionResponse.name.
        let mut map = std::collections::HashMap::new();
        map.insert("call_abc123".to_string(), "get_weather".to_string());

        let request = Request::default()
            .model("gemini-2.0-flash")
            .messages(vec![Message {
                role: Role::Tool,
                content: vec![ContentPart::ToolResult {
                    tool_result: unified_llm_types::ToolResultData {
                        tool_call_id: "call_abc123".into(),
                        content: serde_json::json!({"temp": "72F"}),
                        is_error: false,
                        image_data: None,
                        image_media_type: None,
                    },
                }],
                name: None,
                tool_call_id: None,
            }]);

        let (body, _) = translate_request(&request, Some(&map));
        let parts = body["contents"][0]["parts"].as_array().unwrap();
        let fr = &parts[0]["functionResponse"];
        // Must resolve to the function name, NOT the synthetic ID
        assert_eq!(fr["name"], "get_weather");
        assert_eq!(fr["response"]["temp"], "72F");
    }

    #[test]
    fn test_translate_request_tool_result_fallback_when_not_in_map() {
        // When tool_call_id is NOT in the map, fall back to using the ID as-is
        let map = std::collections::HashMap::new(); // empty map

        let request = Request::default()
            .model("gemini-2.0-flash")
            .messages(vec![Message {
                role: Role::Tool,
                content: vec![ContentPart::ToolResult {
                    tool_result: unified_llm_types::ToolResultData {
                        tool_call_id: "get_weather".into(), // already a real name
                        content: serde_json::json!({"temp": "72F"}),
                        is_error: false,
                        image_data: None,
                        image_media_type: None,
                    },
                }],
                name: None,
                tool_call_id: None,
            }]);

        let (body, _) = translate_request(&request, Some(&map));
        let parts = body["contents"][0]["parts"].as_array().unwrap();
        // Falls back to tool_call_id verbatim
        assert_eq!(parts[0]["functionResponse"]["name"], "get_weather");
    }

    // === AF-07: Outbound thinking arms tests ===

    #[test]
    fn test_gemini_outbound_thinking_part() {
        let parts = vec![ContentPart::Thinking {
            thinking: ThinkingData {
                text: "Let me reason...".to_string(),
                signature: None,
                redacted: false,
                data: None,
            },
        }];
        let (translated, _) = translate_content_parts(&parts);
        assert_eq!(translated.len(), 1, "Thinking block should not be dropped");
        let part = &translated[0];
        assert_eq!(part["thought"], true);
        assert_eq!(part["text"], "Let me reason...");
    }

    #[test]
    fn test_gemini_outbound_thinking_with_signature() {
        let parts = vec![ContentPart::Thinking {
            thinking: ThinkingData {
                text: "Deep thought...".to_string(),
                signature: Some("sig_gem_abc".to_string()),
                redacted: false,
                data: None,
            },
        }];
        let (translated, _) = translate_content_parts(&parts);
        assert_eq!(translated.len(), 1, "Thinking block should not be dropped");
        let part = &translated[0];
        assert_eq!(part["thought"], true);
        assert_eq!(part["text"], "Deep thought...");
        assert_eq!(part["thoughtSignature"], "sig_gem_abc");
    }

    // --- AF-19: tool_call_map cleared between requests ---

    #[test]
    fn test_gemini_tool_result_resolves_from_conversation_history() {
        // Regression test for FP-14 (C-18 regression):
        // After clear_tool_call_map(), an empty map is passed to translate_request,
        // causing functionResponse.name to fall back to the tool_call_id instead
        // of the actual function name. The fix pre-populates the map from
        // conversation history before calling translate_request.
        //
        // This test verifies the complete do_complete() flow by simulating
        // what should happen: the adapter pre-populates the map from conversation
        // history, so translate_request gets the correct name mappings.

        use unified_llm_types::content::{ArgumentValue, ToolCallData, ToolResultData};
        use unified_llm_types::message::Role;

        let request = Request::default().model("gemini-2.5-flash").messages(vec![
            Message::user("What's the weather?"),
            Message {
                role: Role::Assistant,
                content: vec![ContentPart::ToolCall {
                    tool_call: ToolCallData {
                        id: "call_abc123".into(),
                        name: "get_weather".into(),
                        arguments: ArgumentValue::Dict(serde_json::Map::new()),
                        r#type: "function".into(),
                    },
                }],
                name: None,
                tool_call_id: None,
            },
            Message {
                role: Role::Tool,
                content: vec![ContentPart::ToolResult {
                    tool_result: ToolResultData {
                        tool_call_id: "call_abc123".into(),
                        content: serde_json::json!({"temp": "72F"}),
                        is_error: false,
                        image_data: None,
                        image_media_type: None,
                    },
                }],
                name: None,
                tool_call_id: None,
            },
        ]);

        // Simulate the pre-population logic that do_complete/do_stream should perform
        let mut map = std::collections::HashMap::new();
        for msg in &request.messages {
            if msg.role == Role::Assistant {
                for part in &msg.content {
                    if let ContentPart::ToolCall { tool_call } = part {
                        map.insert(tool_call.id.clone(), tool_call.name.clone());
                    }
                }
            }
        }

        let (body, _) = translate_request(&request, Some(&map));
        let contents = body["contents"].as_array().unwrap();

        // Find the functionResponse part
        let tool_msg = contents
            .iter()
            .find(|c| {
                c["parts"].as_array().map_or(false, |parts| {
                    parts.iter().any(|p| p.get("functionResponse").is_some())
                })
            })
            .expect("Should have a functionResponse message");

        let fn_response = &tool_msg["parts"][0]["functionResponse"];
        assert_eq!(
            fn_response["name"], "get_weather",
            "Should resolve to function name, not synthetic ID"
        );
    }

    #[tokio::test]
    async fn test_gemini_do_complete_prepopulates_tool_call_map() {
        // Regression test for FP-14:
        // After clear_tool_call_map() (C-18 fix), the map is empty when
        // translate_request tries to resolve tool_call_id → function name.
        // This causes functionResponse.name = "call_abc123" (the synthetic ID)
        // instead of "get_weather" (the actual function name).
        //
        // The fix: do_complete() pre-populates the map from conversation history
        // BEFORE calling translate_request.
        //
        // We verify the actual request body sent to the server.

        use unified_llm_types::content::{ArgumentValue, ToolCallData, ToolResultData};
        use unified_llm_types::message::Role;

        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path_regex(
                r"/v1beta/models/.+:generateContent",
            ))
            .respond_with(
                wiremock::ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "candidates": [{
                        "content": {"parts": [{"text": "It's 72F"}], "role": "model"},
                        "finishReason": "STOP"
                    }],
                    "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 5, "totalTokenCount": 25}
                })),
            )
            .mount(&server)
            .await;

        let adapter = GeminiAdapter::new_with_base_url(
            SecretString::from("test-key".to_string()),
            server.uri(),
        );

        // Manually pollute the map with stale data from a "previous request"
        adapter
            .tool_call_map
            .lock()
            .unwrap()
            .insert("stale_id".to_string(), "stale_fn".to_string());

        let request = Request::default().model("gemini-2.5-flash").messages(vec![
            Message::user("What's the weather?"),
            Message {
                role: Role::Assistant,
                content: vec![ContentPart::ToolCall {
                    tool_call: ToolCallData {
                        id: "call_abc123".into(),
                        name: "get_weather".into(),
                        arguments: ArgumentValue::Dict(serde_json::Map::new()),
                        r#type: "function".into(),
                    },
                }],
                name: None,
                tool_call_id: None,
            },
            Message {
                role: Role::Tool,
                content: vec![ContentPart::ToolResult {
                    tool_result: ToolResultData {
                        tool_call_id: "call_abc123".into(),
                        content: serde_json::json!({"temp": "72F"}),
                        is_error: false,
                        image_data: None,
                        image_media_type: None,
                    },
                }],
                name: None,
                tool_call_id: None,
            },
        ]);

        adapter.do_complete(request).await.unwrap();

        // Inspect the actual request body sent to the server
        let requests = server.received_requests().await.unwrap();
        assert_eq!(requests.len(), 1);
        let sent_body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();

        // Find the functionResponse in the sent body
        let contents = sent_body["contents"].as_array().unwrap();
        let tool_msg = contents
            .iter()
            .find(|c| {
                c["parts"].as_array().map_or(false, |parts| {
                    parts.iter().any(|p| p.get("functionResponse").is_some())
                })
            })
            .expect("Should have a functionResponse message");

        let fn_response = &tool_msg["parts"][0]["functionResponse"];
        assert_eq!(
            fn_response["name"], "get_weather",
            "functionResponse.name should be the actual function name, not the synthetic ID 'call_abc123'"
        );

        // Verify stale data was cleared
        let map = adapter.tool_call_map.lock().unwrap();
        assert!(
            !map.contains_key("stale_id"),
            "Stale entries should be cleared"
        );
    }

    #[test]
    fn test_gemini_tool_call_map_cleared_on_complete() {
        // Verify that the tool_call_map is cleared at the start of do_complete().
        // Simulate a previous request by manually inserting into the map.
        let adapter =
            GeminiAdapter::new_with_base_url(SecretString::from("test-key"), "http://localhost:0");

        // Manually insert into the map to simulate a previous request
        adapter
            .tool_call_map
            .lock()
            .unwrap()
            .insert("old_call_id".to_string(), "old_function_name".to_string());
        assert!(
            !adapter.tool_call_map.lock().unwrap().is_empty(),
            "Map should have old entries"
        );

        // The clear happens at the start of do_complete/do_stream.
        // We can't easily call do_complete without a real server, so
        // verify the clear_tool_call_map helper works.
        adapter.clear_tool_call_map();
        assert!(
            adapter.tool_call_map.lock().unwrap().is_empty(),
            "Map should be cleared by clear_tool_call_map()"
        );
    }

    // === FP-15: reasoning_effort → Gemini thinkingConfig ===

    #[test]
    fn test_gemini_reasoning_effort_maps_to_thinking_level() {
        let request = Request::default()
            .model("gemini-2.5-flash")
            .messages(vec![Message::user("think hard")])
            .reasoning_effort("high");
        let (body, _) = translate_request(&request, None);
        assert_eq!(body["thinkingConfig"]["thinkingLevel"], "HIGH");
    }

    #[test]
    fn test_gemini_reasoning_effort_low() {
        let request = Request::default()
            .model("gemini-2.5-flash")
            .messages(vec![Message::user("quick answer")])
            .reasoning_effort("low");
        let (body, _) = translate_request(&request, None);
        assert_eq!(body["thinkingConfig"]["thinkingLevel"], "LOW");
    }

    #[test]
    fn test_gemini_reasoning_effort_medium() {
        let request = Request::default()
            .model("gemini-2.5-flash")
            .messages(vec![Message::user("think")])
            .reasoning_effort("medium");
        let (body, _) = translate_request(&request, None);
        assert_eq!(body["thinkingConfig"]["thinkingLevel"], "MEDIUM");
    }

    #[test]
    fn test_gemini_reasoning_effort_none() {
        let request = Request::default()
            .model("gemini-2.5-flash")
            .messages(vec![Message::user("no thinking")])
            .reasoning_effort("none");
        let (body, _) = translate_request(&request, None);
        assert_eq!(
            body["thinkingConfig"]["thinkingLevel"], "THINKING_BUDGET_NONE",
            "reasoning_effort 'none' must map to THINKING_BUDGET_NONE, not NONE"
        );
    }

    #[test]
    fn test_gemini_reasoning_effort_does_not_override_explicit_thinking_config() {
        let request = Request::default()
            .model("gemini-2.5-flash")
            .messages(vec![Message::user("think")])
            .reasoning_effort("high")
            .provider_options(Some(serde_json::json!({
                "gemini": {
                    "thinkingConfig": {
                        "thinkingLevel": "LOW"
                    }
                }
            })));
        let (body, _) = translate_request(&request, None);
        // Explicit provider_options.gemini.thinkingConfig should take precedence
        // thinkingConfig is at request root level
        assert_eq!(body["thinkingConfig"]["thinkingLevel"], "LOW");
    }

    // === S-1: text_id propagation to TextDelta/TextEnd ===

    #[test]
    fn test_gemini_stream_text_id_consistent_across_start_delta_end() {
        let mut translator = GeminiStreamTranslator::new();

        // Chunk 1: text part → StreamStart + TextStart + TextDelta
        let chunk1 = serde_json::json!({
            "candidates": [{
                "content": {"parts": [{"text": "Hello"}], "role": "model"}
            }]
        });
        let events1 = translator.process(&chunk1);
        let text_start = events1
            .iter()
            .find(|e| e.event_type == StreamEventType::TextStart)
            .expect("Should emit TextStart");
        let text_id = text_start
            .text_id
            .clone()
            .expect("TextStart must have text_id");
        assert!(
            text_id.starts_with("txt_"),
            "text_id should start with txt_"
        );

        let text_delta1 = events1
            .iter()
            .find(|e| e.event_type == StreamEventType::TextDelta)
            .expect("Should emit TextDelta");
        assert_eq!(
            text_delta1.text_id.as_ref(),
            Some(&text_id),
            "TextDelta must carry same text_id as TextStart"
        );

        // Chunk 2: more text → TextDelta only
        let chunk2 = serde_json::json!({
            "candidates": [{
                "content": {"parts": [{"text": " world"}], "role": "model"}
            }]
        });
        let events2 = translator.process(&chunk2);
        let text_delta2 = events2
            .iter()
            .find(|e| e.event_type == StreamEventType::TextDelta)
            .expect("Should emit TextDelta");
        assert_eq!(
            text_delta2.text_id.as_ref(),
            Some(&text_id),
            "Second TextDelta must carry same text_id"
        );

        // Chunk 3: finish → TextEnd with text_id
        let chunk3 = serde_json::json!({
            "candidates": [{
                "content": {"parts": [{"text": "!"}], "role": "model"},
                "finishReason": "STOP"
            }],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3, "totalTokenCount": 8}
        });
        let events3 = translator.process(&chunk3);
        let text_end = events3
            .iter()
            .find(|e| e.event_type == StreamEventType::TextEnd)
            .expect("Should emit TextEnd");
        assert_eq!(
            text_end.text_id.as_ref(),
            Some(&text_id),
            "TextEnd must carry same text_id as TextStart"
        );
    }

    // === C-2: AdapterTimeout wiring ===

    #[tokio::test]
    async fn test_connect_timeout_is_wired() {
        use std::time::{Duration, Instant};

        // 10.255.255.1 is a non-routable address — connection will hang until timeout
        let adapter = GeminiAdapter::new_with_base_url_and_timeout(
            SecretString::from("test-key".to_string()),
            "http://10.255.255.1",
            AdapterTimeout {
                connect: 1.0,
                request: 5.0,
                stream_read: 5.0,
            },
        );

        let req = Request::default()
            .model("test")
            .messages(vec![Message::user("hello")]);

        let start = Instant::now();
        let result = adapter.complete(req).await;
        let elapsed = start.elapsed();

        assert!(result.is_err(), "Should fail due to connect timeout");
        assert!(
            elapsed < Duration::from_secs(3),
            "Should timeout within ~1s, took {:?}",
            elapsed
        );
    }
}
