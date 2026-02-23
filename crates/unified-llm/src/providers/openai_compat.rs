// OpenAI-compatible Chat Completions adapter for third-party services.

use futures::StreamExt;
use secrecy::{ExposeSecret, SecretString};
use serde_json::json;

use crate::util::sse::SseParser;

use unified_llm_types::{
    AdapterTimeout, ArgumentValue, BoxFuture, BoxStream, ContentPart, Error, FinishReason, Message,
    ProviderAdapter, Request, Response, Role, StreamEvent, StreamEventType, ToolCall, ToolCallData,
    Usage, Warning,
};

/// OpenAI-compatible Chat Completions adapter for third-party services.
///
/// Uses `/v1/chat/completions` — the standard Chat Completions protocol
/// supported by vLLM, Ollama, Together AI, Groq, and other services.
///
/// This is distinct from `OpenAiAdapter` which uses the Responses API.
pub struct OpenAICompatibleAdapter {
    api_key: SecretString,
    base_url: String,
    http_client: reqwest::Client,
    /// Per-chunk timeout for streaming responses (from AdapterTimeout.stream_read).
    stream_read_timeout: std::time::Duration,
}

impl OpenAICompatibleAdapter {
    /// Create a new adapter with API key and base URL.
    ///
    /// Uses default timeouts: connect=10s, request=120s.
    ///
    /// # Examples
    /// ```ignore
    /// let adapter = OpenAICompatibleAdapter::new(
    ///     SecretString::from("sk-..."),
    ///     "https://my-vllm-instance.example.com",
    /// );
    /// ```
    pub fn new(api_key: SecretString, base_url: impl Into<String>) -> Self {
        let timeout = AdapterTimeout::default();
        Self {
            api_key,
            base_url: crate::util::normalize_base_url(&base_url.into()),
            stream_read_timeout: std::time::Duration::from_secs_f64(timeout.stream_read),
            http_client: Self::build_http_client(&timeout),
        }
    }

    /// Create a new adapter with API key, base URL, and custom timeouts.
    pub fn new_with_timeout(
        api_key: SecretString,
        base_url: impl Into<String>,
        timeout: AdapterTimeout,
    ) -> Self {
        Self {
            api_key,
            base_url: crate::util::normalize_base_url(&base_url.into()),
            stream_read_timeout: std::time::Duration::from_secs_f64(timeout.stream_read),
            http_client: Self::build_http_client(&timeout),
        }
    }

    // NO from_env() — per spec §7.10, this is always programmatic.
    // The base_url is required and there's no standard env var for it.

    /// Create a builder for fine-grained configuration.
    pub fn builder(
        api_key: SecretString,
        base_url: impl Into<String>,
    ) -> OpenAICompatibleAdapterBuilder {
        OpenAICompatibleAdapterBuilder::new(api_key, base_url)
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

    /// Perform the actual HTTP request for complete().
    async fn do_complete(&self, mut request: Request) -> Result<Response, Error> {
        // H-4: Pre-resolve local file images to avoid blocking I/O in translate_request.
        crate::util::image::pre_resolve_local_images(&mut request.messages).await?;

        let url = format!("{}/v1/chat/completions", self.base_url);
        let (body, translation_warnings) = translate_request(&request);

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
            let error_body: serde_json::Value = http_response
                .json()
                .await
                .unwrap_or(json!({"error": {"message": "Failed to parse error response"}}));
            return Err(parse_error(status, &headers, error_body));
        }

        let response_body: serde_json::Value = http_response
            .json()
            .await
            .map_err(|e| Error::network(format!("Failed to parse response: {e}"), e))?;

        let mut response = parse_response(response_body, &headers)?;
        response.warnings = translation_warnings;
        Ok(response)
    }

    /// Build common HTTP headers for Chat Completions API requests.
    fn build_headers(&self) -> Result<reqwest::header::HeaderMap, Error> {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            "authorization",
            format!("Bearer {}", self.api_key.expose_secret())
                .parse()
                .map_err(|_| {
                    Error::configuration(
                        "Invalid API key: contains non-ASCII or control characters",
                    )
                })?,
        );
        headers.insert("content-type", "application/json".parse().unwrap());
        Ok(headers)
    }
}

/// Builder for constructing an `OpenAICompatibleAdapter` with fine-grained configuration.
///
/// Unlike the other adapter builders, `base_url` is required (set in `new()`).
pub struct OpenAICompatibleAdapterBuilder {
    api_key: SecretString,
    base_url: String,
    timeout: Option<AdapterTimeout>,
    default_headers: Option<reqwest::header::HeaderMap>,
}

impl OpenAICompatibleAdapterBuilder {
    /// Create a new builder with the required API key and base URL.
    pub fn new(api_key: SecretString, base_url: impl Into<String>) -> Self {
        Self {
            api_key,
            base_url: crate::util::normalize_base_url(&base_url.into()),
            timeout: None,
            default_headers: None,
        }
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

    /// Build the `OpenAICompatibleAdapter`.
    pub fn build(self) -> OpenAICompatibleAdapter {
        let timeout = self.timeout.unwrap_or_default();
        OpenAICompatibleAdapter {
            api_key: self.api_key,
            base_url: self.base_url,
            stream_read_timeout: std::time::Duration::from_secs_f64(timeout.stream_read),
            http_client: OpenAICompatibleAdapter::build_http_client_with_headers(
                &timeout,
                self.default_headers,
            ),
        }
    }
}

impl ProviderAdapter for OpenAICompatibleAdapter {
    fn name(&self) -> &str {
        "openai-compatible"
    }

    fn complete(&self, request: Request) -> BoxFuture<'_, Result<Response, Error>> {
        Box::pin(self.do_complete(request))
    }

    fn stream(&self, mut request: Request) -> BoxStream<'_, Result<StreamEvent, Error>> {
        let stream = async_stream::stream! {
            // H-4: Pre-resolve local file images to avoid blocking I/O in translate_request.
            if let Err(e) = crate::util::image::pre_resolve_local_images(&mut request.messages).await {
                yield Err(e);
                return;
            }

            let url = format!("{}/v1/chat/completions", self.base_url);
            let (mut body, translation_warnings) = translate_request(&request);
            // L-4: Log warnings that can't be attached to streaming responses
            for w in &translation_warnings {
                tracing::warn!("Translation warning (streaming): {}", w.message);
            }

            // Enable streaming + request usage in stream
            if let Some(obj) = body.as_object_mut() {
                obj.insert("stream".into(), json!(true));
                obj.insert("stream_options".into(), json!({"include_usage": true}));
            }

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
                    .unwrap_or(json!({"error": {"message": "Failed to parse error response"}}));
                yield Err(parse_error(status, &headers, error_body));
                return;
            }

            // True incremental streaming: read chunks as they arrive
            let mut parser = SseParser::new();
            let mut translator = ChatCompletionsStreamTranslator::new();
            let mut byte_stream = http_response.bytes_stream();
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
                    Err(_) => continue, // Skip invalid UTF-8 chunks
                };

                let sse_events = parser.feed(chunk_str);

                for sse_event in sse_events {
                    // Chat Completions SSE doesn't use named event types —
                    // all data is in "data:" lines. Check for [DONE] sentinel.
                    let data_str = sse_event.data.trim();
                    if data_str == "[DONE]" {
                        return;  // Stream complete
                    }

                    match serde_json::from_str::<serde_json::Value>(data_str) {
                        Ok(data) => {
                            for event in translator.process(&data) {
                                yield Ok(event);
                            }
                        }
                        Err(e) => {
                            tracing::warn!("Failed to parse SSE data: {e}");
                        }
                    }
                }
            }
        };
        Box::pin(stream)
    }
}

// === Request Translation ===

/// Translate a unified Request into a Chat Completions JSON body.
pub(crate) fn translate_request(request: &Request) -> (serde_json::Value, Vec<Warning>) {
    let mut body = serde_json::Map::new();
    let mut warnings: Vec<Warning> = Vec::new();
    body.insert("model".into(), json!(request.model));

    let mut messages: Vec<serde_json::Value> = Vec::new();

    for msg in &request.messages {
        match msg.role {
            Role::System => {
                messages.push(json!({
                    "role": "system",
                    "content": msg.text(),
                }));
            }
            Role::Developer => {
                messages.push(json!({
                    "role": "developer",
                    "content": msg.text(),
                }));
            }
            Role::User => {
                let (content, mut part_warnings) = translate_user_content(&msg.content);
                warnings.append(&mut part_warnings);
                messages.push(json!({
                    "role": "user",
                    "content": content,
                }));
            }
            Role::Assistant => {
                let mut assistant_msg = json!({
                    "role": "assistant",
                });

                // Text content
                let text = msg.text();
                if !text.is_empty() {
                    assistant_msg["content"] = json!(text);
                }

                // Tool calls — nested format on the assistant message
                let tool_calls: Vec<serde_json::Value> = msg
                    .content
                    .iter()
                    .filter_map(|p| {
                        if let ContentPart::ToolCall { tool_call } = p {
                            let arguments_str = match &tool_call.arguments {
                                ArgumentValue::Dict(map) => {
                                    serde_json::to_string(&serde_json::Value::Object(map.clone()))
                                        .unwrap_or_else(|_| "{}".to_string())
                                }
                                ArgumentValue::Raw(s) => s.clone(),
                            };
                            Some(json!({
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.name,
                                    "arguments": arguments_str,
                                }
                            }))
                        } else {
                            None
                        }
                    })
                    .collect();

                if !tool_calls.is_empty() {
                    assistant_msg["tool_calls"] = json!(tool_calls);
                }

                messages.push(assistant_msg);
            }
            Role::Tool => {
                // Each tool result is a separate message with role "tool"
                for part in &msg.content {
                    if let ContentPart::ToolResult { tool_result } = part {
                        let content = match &tool_result.content {
                            serde_json::Value::String(s) => s.clone(),
                            other => other.to_string(),
                        };
                        messages.push(json!({
                            "role": "tool",
                            "tool_call_id": tool_result.tool_call_id,
                            "content": content,
                        }));
                    }
                }
            }
        }
    }

    body.insert("messages".into(), json!(messages));

    // --- Optional parameters ---

    // max_tokens passes through directly (NOT max_output_tokens)
    if let Some(max_tokens) = request.max_tokens {
        body.insert("max_tokens".into(), json!(max_tokens));
    }

    if let Some(temp) = request.temperature {
        body.insert("temperature".into(), json!(temp));
    }
    if let Some(top_p) = request.top_p {
        body.insert("top_p".into(), json!(top_p));
    }
    if let Some(ref stop) = request.stop_sequences {
        body.insert("stop".into(), json!(stop));
    }

    // P1-4: Warn when reasoning_effort is present but dropped
    if request.reasoning_effort.is_some() {
        tracing::warn!(
            "reasoning_effort is set but the OpenAI-compatible adapter (Chat Completions) \
             does not support it. Use the native OpenAI adapter for reasoning models."
        );
    }
    // The spec explicitly says: "does not support reasoning tokens"

    // Tools — nested {type: "function", function: {...}} format
    if let Some(ref tools) = request.tools {
        let tool_defs: Vec<serde_json::Value> = tools
            .iter()
            .map(|t| {
                json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                        "strict": t.strict.unwrap_or(false),
                    }
                })
            })
            .collect();
        body.insert("tools".into(), json!(tool_defs));
    }

    // Tool choice — same string modes, but named uses nested format
    if let Some(ref tc) = request.tool_choice {
        let choice_val = match tc.mode.as_str() {
            "auto" => json!("auto"),
            "none" => json!("none"),
            "required" => json!("required"),
            "named" => {
                if let Some(ref name) = tc.tool_name {
                    json!({"type": "function", "function": {"name": name}})
                } else {
                    json!("auto")
                }
            }
            _ => json!("auto"),
        };
        body.insert("tool_choice".into(), choice_val);
    }

    // Structured output — response_format (Chat Completions format)
    if let Some(ref fmt) = request.response_format {
        if fmt.r#type == "json_schema" {
            let schema = fmt.json_schema.clone().unwrap_or(json!({}));
            body.insert(
                "response_format".into(),
                json!({
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response",
                        "schema": schema,
                        "strict": fmt.strict,
                    }
                }),
            );
        } else if fmt.r#type == "json_object" {
            body.insert(
                "response_format".into(),
                json!({
                    "type": "json_object",
                }),
            );
        }
    }

    // Provider options passthrough
    const INTERNAL_KEYS: &[&str] = &[];
    if let Some(opts) = crate::util::provider_options::get_provider_options(
        &request.provider_options,
        "openai-compatible",
    ) {
        let mut body_val = serde_json::Value::Object(body);
        crate::util::provider_options::merge_provider_options(&mut body_val, &opts, INTERNAL_KEYS);
        return (body_val, warnings);
    }

    (serde_json::Value::Object(body), warnings)
}

/// Translate user content parts to Chat Completions format.
///
/// For text-only messages, returns the text as a plain string (more compatible).
/// For multimodal messages, returns an array of content parts.
fn translate_user_content(parts: &[ContentPart]) -> (serde_json::Value, Vec<Warning>) {
    let mut warnings = Vec::new();
    let has_non_text = parts.iter().any(|p| !matches!(p, ContentPart::Text { .. }));

    if !has_non_text && parts.len() == 1 {
        // Simple text — return as string for maximum compatibility
        if let ContentPart::Text { text } = &parts[0] {
            return (json!(text), warnings);
        }
    }

    // Multimodal or multiple parts — return as array
    let content_parts: Vec<serde_json::Value> = parts
        .iter()
        .filter_map(|part| match part {
            ContentPart::Text { text } => Some(json!({"type": "text", "text": text})),
            ContentPart::Image { image } => {
                if let Some(ref data) = image.data {
                    let b64 = crate::util::image::base64_encode(data);
                    let media_type = image.media_type.as_deref().unwrap_or("image/png");
                    let data_uri = crate::util::image::encode_data_uri(media_type, &b64);
                    Some(json!({
                        "type": "image_url",
                        "image_url": {"url": data_uri, "detail": image.detail.as_deref().unwrap_or("auto")}
                    }))
                } else {
                    image.url.as_ref().map(|url| {
                        if crate::util::image::is_local_path(url) {
                            match crate::util::image::resolve_local_file(url) {
                                Ok((data, mime)) => {
                                    let b64 = crate::util::image::base64_encode(&data);
                                    let data_uri = crate::util::image::encode_data_uri(&mime, &b64);
                                    json!({
                                        "type": "image_url",
                                        "image_url": {"url": data_uri, "detail": image.detail.as_deref().unwrap_or("auto")}
                                    })
                                }
                                Err(e) => {
                                    tracing::warn!("Failed to resolve local image '{}': {}", url, e.message);
                                    json!({
                                        "type": "image_url",
                                        "image_url": {"url": url, "detail": image.detail.as_deref().unwrap_or("auto")}
                                    })
                                }
                            }
                        } else {
                            json!({
                                "type": "image_url",
                                "image_url": {"url": url, "detail": image.detail.as_deref().unwrap_or("auto")}
                            })
                        }
                    })
                }
            }
            other => {
                let msg = format!(
                    "Dropped unsupported content part kind={:?} for provider=openai-compatible",
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

    (json!(content_parts), warnings)
}

// === Response Translation ===

/// Parse a Chat Completions response into a unified Response.
pub(crate) fn parse_response(
    raw: serde_json::Value,
    headers: &reqwest::header::HeaderMap,
) -> Result<Response, Error> {
    let id = raw
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let model = raw
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let choice = raw
        .get("choices")
        .and_then(|v| v.as_array())
        .and_then(|a| a.first());

    let mut content_parts: Vec<ContentPart> = Vec::new();
    let mut has_tool_calls = false;

    if let Some(choice) = choice {
        let message = choice.get("message");

        // Text content
        if let Some(content) = message
            .and_then(|m| m.get("content"))
            .and_then(|v| v.as_str())
        {
            if !content.is_empty() {
                content_parts.push(ContentPart::Text {
                    text: content.to_string(),
                });
            }
        }

        // Tool calls
        if let Some(tool_calls) = message
            .and_then(|m| m.get("tool_calls"))
            .and_then(|v| v.as_array())
        {
            has_tool_calls = true;
            for tc in tool_calls {
                let call_id = tc
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let function = tc.get("function");
                let name = function
                    .and_then(|f| f.get("name"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let arguments_str = function
                    .and_then(|f| f.get("arguments"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("{}");
                let arguments = serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(
                    arguments_str,
                )
                .map(ArgumentValue::Dict)
                .unwrap_or_else(|_| ArgumentValue::Raw(arguments_str.to_string()));

                content_parts.push(ContentPart::ToolCall {
                    tool_call: ToolCallData {
                        id: call_id,
                        name,
                        arguments,
                        r#type: "function".to_string(),
                    },
                });
            }
        }
    }

    // Finish reason — direct mapping from Chat Completions string
    let finish_reason = choice
        .and_then(|c| c.get("finish_reason"))
        .and_then(|v| v.as_str())
        .map(|fr| match fr {
            "stop" => FinishReason {
                reason: "stop".into(),
                raw: Some("stop".into()),
            },
            "length" => FinishReason {
                reason: "length".into(),
                raw: Some("length".into()),
            },
            "tool_calls" => FinishReason {
                reason: "tool_calls".into(),
                raw: Some("tool_calls".into()),
            },
            "content_filter" => FinishReason {
                reason: "content_filter".into(),
                raw: Some("content_filter".into()),
            },
            other => FinishReason {
                reason: "other".into(),
                raw: Some(other.into()),
            },
        })
        .unwrap_or(FinishReason {
            reason: if has_tool_calls { "tool_calls" } else { "stop" }.into(),
            raw: None,
        });

    // Usage — Chat Completions uses prompt_tokens/completion_tokens
    let usage_obj = raw.get("usage");
    let input_tokens = usage_obj
        .and_then(|u| u.get("prompt_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    let output_tokens = usage_obj
        .and_then(|u| u.get("completion_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    let usage = Usage {
        input_tokens,
        output_tokens,
        // F-10: always recompute total_tokens instead of trusting provider
        total_tokens: input_tokens + output_tokens,
        reasoning_tokens: usage_obj
            .and_then(|u| u.get("completion_tokens_details"))
            .and_then(|d| d.get("reasoning_tokens"))
            .and_then(|v| v.as_u64())
            .map(|v| v as u32),
        cache_read_tokens: usage_obj
            .and_then(|u| u.get("prompt_tokens_details"))
            .and_then(|d| d.get("cached_tokens"))
            .and_then(|v| v.as_u64())
            .map(|v| v as u32),
        cache_write_tokens: None,
        raw: usage_obj.cloned(),
    };

    Ok(Response {
        id,
        model,
        provider: "openai-compatible".to_string(),
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

/// Parse an error response from a Chat Completions API.
pub(crate) fn parse_error(
    status: u16,
    headers: &reqwest::header::HeaderMap,
    body: serde_json::Value,
) -> Error {
    let (error_message, error_code) = crate::util::http::parse_provider_error_message(
        &body,
        &["error", "message"],
        &["error", "code"],
    );

    let retry_after = crate::util::http::parse_retry_after(headers);

    let mut err = Error::from_http_status(
        status,
        error_message,
        "openai-compatible",
        Some(body),
        retry_after,
    );
    err.error_code = error_code;
    err
}

// === Stream Translation ===

/// Tracks state for an in-progress tool call during streaming.
struct ActiveToolCall {
    id: String,
    name: String,
    accumulated_arguments: String,
    started: bool,
}

/// Stateful translator for Chat Completions streaming delta format.
struct ChatCompletionsStreamTranslator {
    response_id: String,
    model: String,
    text_started: bool,
    /// S-1: text_id generated on TextStart, propagated to TextDelta/TextEnd.
    current_text_id: Option<String>,
    reasoning_started: bool,
    active_tool_calls: Vec<ActiveToolCall>,
    stream_started: bool,
}

impl ChatCompletionsStreamTranslator {
    fn new() -> Self {
        Self {
            response_id: String::new(),
            model: String::new(),
            text_started: false,
            current_text_id: None,
            reasoning_started: false,
            active_tool_calls: Vec::new(),
            stream_started: false,
        }
    }

    /// Process a single SSE data line (already parsed from JSON).
    fn process(&mut self, data: &serde_json::Value) -> Vec<StreamEvent> {
        let mut events = Vec::new();

        // Extract response metadata from first chunk
        if !self.stream_started {
            self.stream_started = true;
            self.response_id = data
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            self.model = data
                .get("model")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            events.push(StreamEvent {
                event_type: StreamEventType::StreamStart,
                id: Some(self.response_id.clone()),
                ..Default::default()
            });
        }

        let choice = data
            .get("choices")
            .and_then(|v| v.as_array())
            .and_then(|a| a.first());

        if let Some(choice) = choice {
            let delta = choice.get("delta");
            let finish_reason = choice.get("finish_reason").and_then(|v| v.as_str());

            // Text delta
            if let Some(content) = delta
                .and_then(|d| d.get("content"))
                .and_then(|v| v.as_str())
            {
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
                    delta: Some(content.to_string()),
                    text_id: self.current_text_id.clone(),
                    ..Default::default()
                });
            }

            // Reasoning content delta (o-series models via Chat Completions)
            if let Some(reasoning) = delta
                .and_then(|d| d.get("reasoning_content"))
                .and_then(|v| v.as_str())
            {
                if !self.reasoning_started {
                    self.reasoning_started = true;
                    events.push(StreamEvent {
                        event_type: StreamEventType::ReasoningStart,
                        ..Default::default()
                    });
                }
                events.push(StreamEvent {
                    event_type: StreamEventType::ReasoningDelta,
                    delta: Some(reasoning.to_string()),
                    reasoning_delta: Some(reasoning.to_string()),
                    ..Default::default()
                });
            }

            // Tool call deltas
            if let Some(tool_calls) = delta
                .and_then(|d| d.get("tool_calls"))
                .and_then(|v| v.as_array())
            {
                for tc_delta in tool_calls {
                    let index =
                        tc_delta.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;

                    // Ensure we have enough slots
                    while self.active_tool_calls.len() <= index {
                        self.active_tool_calls.push(ActiveToolCall {
                            id: String::new(),
                            name: String::new(),
                            accumulated_arguments: String::new(),
                            started: false,
                        });
                    }

                    let active = &mut self.active_tool_calls[index];

                    // First chunk for this tool call has id and function.name
                    if let Some(id) = tc_delta.get("id").and_then(|v| v.as_str()) {
                        active.id = id.to_string();
                    }
                    if let Some(name) = tc_delta
                        .get("function")
                        .and_then(|f| f.get("name"))
                        .and_then(|v| v.as_str())
                    {
                        active.name = name.to_string();
                    }

                    // Emit ToolCallStart on first sight of this tool call
                    if !active.started {
                        active.started = true;
                        events.push(StreamEvent {
                            event_type: StreamEventType::ToolCallStart,
                            tool_call: Some(ToolCall {
                                id: active.id.clone(),
                                name: active.name.clone(),
                                arguments: serde_json::Map::new(),
                                raw_arguments: None,
                            }),
                            ..Default::default()
                        });
                    }

                    // Argument delta
                    if let Some(args_delta) = tc_delta
                        .get("function")
                        .and_then(|f| f.get("arguments"))
                        .and_then(|v| v.as_str())
                    {
                        if !args_delta.is_empty() {
                            active.accumulated_arguments.push_str(args_delta);
                            events.push(StreamEvent {
                                event_type: StreamEventType::ToolCallDelta,
                                delta: Some(args_delta.to_string()),
                                ..Default::default()
                            });
                        }
                    }
                }
            }

            // Finish
            if let Some(reason) = finish_reason {
                // Close any open reasoning
                if self.reasoning_started {
                    events.push(StreamEvent {
                        event_type: StreamEventType::ReasoningEnd,
                        ..Default::default()
                    });
                    self.reasoning_started = false;
                }

                // Close any open text
                if self.text_started {
                    events.push(StreamEvent {
                        event_type: StreamEventType::TextEnd,
                        text_id: self.current_text_id.take(),
                        ..Default::default()
                    });
                    self.text_started = false;
                }

                // Close any open tool calls
                for active in self.active_tool_calls.drain(..) {
                    if active.started {
                        let arguments: serde_json::Map<String, serde_json::Value> =
                            serde_json::from_str(&active.accumulated_arguments).unwrap_or_default();
                        events.push(StreamEvent {
                            event_type: StreamEventType::ToolCallEnd,
                            tool_call: Some(ToolCall {
                                id: active.id,
                                name: active.name,
                                arguments,
                                raw_arguments: Some(active.accumulated_arguments),
                            }),
                            ..Default::default()
                        });
                    }
                }

                // Map finish reason
                let finish = match reason {
                    "stop" => FinishReason {
                        reason: "stop".into(),
                        raw: Some("stop".into()),
                    },
                    "length" => FinishReason {
                        reason: "length".into(),
                        raw: Some("length".into()),
                    },
                    "tool_calls" => FinishReason {
                        reason: "tool_calls".into(),
                        raw: Some("tool_calls".into()),
                    },
                    "content_filter" => FinishReason {
                        reason: "content_filter".into(),
                        raw: Some("content_filter".into()),
                    },
                    other => FinishReason {
                        reason: "other".into(),
                        raw: Some(other.into()),
                    },
                };

                // Usage — may be on the same chunk or a subsequent one
                let usage = data.get("usage").map(|u| {
                    let input_tokens =
                        u.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                    let output_tokens = u
                        .get("completion_tokens")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0) as u32;
                    Usage {
                        input_tokens,
                        output_tokens,
                        // F-10: always recompute total_tokens
                        total_tokens: input_tokens + output_tokens,
                        reasoning_tokens: u
                            .get("completion_tokens_details")
                            .and_then(|d| d.get("reasoning_tokens"))
                            .and_then(|v| v.as_u64())
                            .map(|v| v as u32),
                        // BUG-1: Extract cached_tokens from streaming usage (was hardcoded None)
                        cache_read_tokens: u
                            .get("prompt_tokens_details")
                            .and_then(|d| d.get("cached_tokens"))
                            .and_then(|v| v.as_u64())
                            .map(|v| v as u32),
                        cache_write_tokens: None,
                        raw: Some(u.clone()),
                    }
                });

                events.push(StreamEvent {
                    event_type: StreamEventType::Finish,
                    finish_reason: Some(finish),
                    usage,
                    ..Default::default()
                });
            }
        }

        // Handle standalone usage chunk (some providers send usage separately)
        if choice.is_none() {
            if let Some(u) = data.get("usage") {
                let input_tokens =
                    u.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                let output_tokens = u
                    .get("completion_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as u32;
                let usage = Usage {
                    input_tokens,
                    output_tokens,
                    // F-10: always recompute total_tokens
                    total_tokens: input_tokens + output_tokens,
                    reasoning_tokens: u
                        .get("completion_tokens_details")
                        .and_then(|d| d.get("reasoning_tokens"))
                        .and_then(|v| v.as_u64())
                        .map(|v| v as u32),
                    // BUG-1: Extract cached_tokens from standalone usage chunk (was hardcoded None)
                    cache_read_tokens: u
                        .get("prompt_tokens_details")
                        .and_then(|d| d.get("cached_tokens"))
                        .and_then(|v| v.as_u64())
                        .map(|v| v as u32),
                    cache_write_tokens: None,
                    raw: Some(u.clone()),
                };
                events.push(StreamEvent {
                    event_type: StreamEventType::ProviderEvent,
                    usage: Some(usage),
                    ..Default::default()
                });
            }
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
    fn test_compat_adapter_builder_defaults() {
        let adapter = OpenAICompatibleAdapterBuilder::new(
            SecretString::from("key".to_string()),
            "https://example.com",
        )
        .build();
        assert_eq!(adapter.base_url, "https://example.com");
        assert_eq!(adapter.name(), "openai-compatible");
    }

    #[test]
    fn test_compat_adapter_builder_with_all_options() {
        let adapter = OpenAICompatibleAdapterBuilder::new(
            SecretString::from("key".to_string()),
            "https://custom.vllm.com",
        )
        .timeout(AdapterTimeout {
            connect: 5.0,
            request: 60.0,
            stream_read: 15.0,
        })
        .default_headers(reqwest::header::HeaderMap::new())
        .build();
        assert_eq!(adapter.base_url, "https://custom.vllm.com");
    }

    #[test]
    fn test_compat_adapter_builder_shortcut() {
        let adapter = OpenAICompatibleAdapter::builder(
            SecretString::from("key".to_string()),
            "https://test.com",
        )
        .build();
        assert_eq!(adapter.base_url, "https://test.com");
    }

    // === Core Tests ===

    #[test]
    fn test_adapter_name() {
        let adapter = OpenAICompatibleAdapter::new(
            SecretString::from("sk-test".to_string()),
            "https://example.com",
        );
        assert_eq!(adapter.name(), "openai-compatible");
    }

    #[test]
    fn test_new_sets_custom_base_url() {
        let adapter = OpenAICompatibleAdapter::new(
            SecretString::from("sk-test".to_string()),
            "https://my-vllm.example.com",
        );
        assert_eq!(adapter.base_url, "https://my-vllm.example.com");
    }

    #[test]
    fn test_build_headers_valid_key() {
        let adapter = OpenAICompatibleAdapter::new(
            SecretString::from("sk-test-key".to_string()),
            "https://example.com",
        );
        let headers = adapter.build_headers().unwrap();
        assert_eq!(
            headers.get("authorization").unwrap().to_str().unwrap(),
            "Bearer sk-test-key"
        );
        assert_eq!(
            headers.get("content-type").unwrap().to_str().unwrap(),
            "application/json"
        );
    }

    #[test]
    fn test_build_headers_invalid_key() {
        let adapter = OpenAICompatibleAdapter::new(
            SecretString::from("invalid\x00key".to_string()),
            "https://example.com",
        );
        let result = adapter.build_headers();
        assert!(result.is_err());
    }

    // ===== OC-02: Request Translation — Messages, System, Roles, Params =====

    #[test]
    fn test_system_message_in_messages_array() {
        let request = Request::default().model("gpt-4o").messages(vec![
            Message::system("You are helpful."),
            Message::user("Hi"),
        ]);
        let (body, _) = translate_request(&request);
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "You are helpful.");
        assert_eq!(messages[1]["role"], "user");
        // NOT in instructions
        assert!(body.get("instructions").is_none());
    }

    #[test]
    fn test_developer_message_in_messages_array() {
        let request = Request::default().model("gpt-4o").messages(vec![
            Message {
                role: Role::Developer,
                content: vec![ContentPart::text("Dev instructions")],
                name: None,
                tool_call_id: None,
            },
            Message::user("Hi"),
        ]);
        let (body, _) = translate_request(&request);
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages[0]["role"], "developer");
        assert_eq!(messages[0]["content"], "Dev instructions");
    }

    #[test]
    fn test_user_text_as_string() {
        let request = Request::default()
            .model("gpt-4o")
            .messages(vec![Message::user("Hello world")]);
        let (body, _) = translate_request(&request);
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages[0]["role"], "user");
        // Simple text-only → plain string (NOT array)
        assert_eq!(messages[0]["content"], "Hello world");
    }

    #[test]
    fn test_assistant_text() {
        let request = Request::default().model("gpt-4o").messages(vec![
            Message::user("Hi"),
            Message::assistant("Hello!"),
            Message::user("How are you?"),
        ]);
        let (body, _) = translate_request(&request);
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages[1]["role"], "assistant");
        assert_eq!(messages[1]["content"], "Hello!");
    }

    #[test]
    fn test_max_tokens_direct() {
        let request = Request::default()
            .model("gpt-4o")
            .messages(vec![Message::user("hi")])
            .max_tokens(500);
        let (body, _) = translate_request(&request);
        assert_eq!(body["max_tokens"], 500);
        // NOT max_output_tokens (that's Responses API)
        assert!(body.get("max_output_tokens").is_none());
    }

    #[test]
    fn test_reasoning_effort_ignored() {
        let request = Request::default()
            .model("o3")
            .messages(vec![Message::user("think hard")])
            .reasoning_effort("high");
        let (body, _) = translate_request(&request);
        assert!(body.get("reasoning").is_none());
        assert!(body.get("reasoning_effort").is_none());
    }

    #[test]
    fn test_temperature_and_stop() {
        let request = Request::default()
            .model("gpt-4o")
            .messages(vec![Message::user("hi")])
            .temperature(0.7)
            .stop_sequences(vec!["END".into()]);
        let (body, _) = translate_request(&request);
        assert_eq!(body["temperature"], 0.7);
        assert_eq!(body["stop"][0], "END");
    }

    // ===== OC-03: Request Translation — Tools, Tool Choice, Structured Output =====

    fn make_test_tool() -> ToolDefinition {
        ToolDefinition {
            name: "get_weather".into(),
            description: "Get weather".into(),
            parameters: serde_json::json!({"type": "object", "properties": {"city": {"type": "string"}}}),
            strict: None,
        }
    }

    #[test]
    fn test_tool_definitions_nested_format() {
        let request = Request::default()
            .model("gpt-4o")
            .messages(vec![Message::user("weather?")])
            .tools(vec![make_test_tool()]);
        let (body, _) = translate_request(&request);
        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["type"], "function");
        // Nested: function.name (NOT flat name)
        assert_eq!(tools[0]["function"]["name"], "get_weather");
        assert_eq!(tools[0]["function"]["description"], "Get weather");
        assert!(tools[0]["function"]["parameters"].is_object());
        assert_eq!(tools[0]["function"]["strict"], false); // None defaults to false for compat
    }

    #[test]
    fn test_assistant_tool_calls_nested_format() {
        let request = Request::default().model("gpt-4o").messages(vec![
            Message::user("weather?"),
            Message {
                role: Role::Assistant,
                content: vec![ContentPart::ToolCall {
                    tool_call: ToolCallData {
                        id: "call_123".into(),
                        name: "get_weather".into(),
                        arguments: ArgumentValue::Dict({
                            let mut m = serde_json::Map::new();
                            m.insert("city".into(), serde_json::json!("SF"));
                            m
                        }),
                        r#type: "function".into(),
                    },
                }],
                name: None,
                tool_call_id: None,
            },
        ]);
        let (body, _) = translate_request(&request);
        let messages = body["messages"].as_array().unwrap();
        let tc = &messages[1]["tool_calls"][0];
        assert_eq!(tc["id"], "call_123");
        assert_eq!(tc["type"], "function");
        assert_eq!(tc["function"]["name"], "get_weather");
        // arguments is a JSON string, not object
        let args_str = tc["function"]["arguments"].as_str().unwrap();
        let args: serde_json::Value = serde_json::from_str(args_str).unwrap();
        assert_eq!(args["city"], "SF");
    }

    #[test]
    fn test_tool_result_as_tool_role_message() {
        let request = Request::default().model("gpt-4o").messages(vec![
            Message::user("weather?"),
            Message {
                role: Role::Assistant,
                content: vec![ContentPart::ToolCall {
                    tool_call: ToolCallData {
                        id: "call_123".into(),
                        name: "get_weather".into(),
                        arguments: ArgumentValue::Dict(serde_json::Map::new()),
                        r#type: "function".into(),
                    },
                }],
                name: None,
                tool_call_id: None,
            },
            Message::tool_result("call_123", "72F sunny", false),
        ]);
        let (body, _) = translate_request(&request);
        let messages = body["messages"].as_array().unwrap();
        let tool_msg = &messages[2];
        assert_eq!(tool_msg["role"], "tool");
        assert_eq!(tool_msg["tool_call_id"], "call_123");
        assert_eq!(tool_msg["content"], "72F sunny");
    }

    #[test]
    fn test_tool_choice_auto_none_required() {
        for mode in &["auto", "none", "required"] {
            let request = Request::default()
                .model("gpt-4o")
                .messages(vec![Message::user("hi")])
                .tools(vec![make_test_tool()])
                .tool_choice(ToolChoice {
                    mode: mode.to_string(),
                    tool_name: None,
                });
            let (body, _) = translate_request(&request);
            assert_eq!(body["tool_choice"], *mode);
        }
    }

    #[test]
    fn test_tool_choice_named_nested_format() {
        let request = Request::default()
            .model("gpt-4o")
            .messages(vec![Message::user("hi")])
            .tools(vec![make_test_tool()])
            .tool_choice(ToolChoice {
                mode: "named".into(),
                tool_name: Some("get_weather".into()),
            });
        let (body, _) = translate_request(&request);
        // Nested format: function.name (NOT flat name like Responses API)
        assert_eq!(body["tool_choice"]["type"], "function");
        assert_eq!(body["tool_choice"]["function"]["name"], "get_weather");
    }

    #[test]
    fn test_structured_output_response_format() {
        let request = Request::default()
            .model("gpt-4o")
            .messages(vec![Message::user("extract")])
            .response_format(ResponseFormat {
                r#type: "json_schema".into(),
                json_schema: Some(serde_json::json!({"type": "object", "properties": {"name": {"type": "string"}}})),
                strict: true,
            });
        let (body, _) = translate_request(&request);
        // response_format (NOT text.format like Responses API)
        assert_eq!(body["response_format"]["type"], "json_schema");
        assert!(body["response_format"]["json_schema"]["schema"].is_object());
        assert!(body.get("text").is_none());
    }

    #[test]
    fn test_json_object_response_format() {
        let request = Request::default()
            .model("gpt-4o")
            .messages(vec![Message::user("give me JSON")])
            .response_format(ResponseFormat {
                r#type: "json_object".into(),
                json_schema: None,
                strict: false,
            });
        let (body, _) = translate_request(&request);
        assert_eq!(body["response_format"]["type"], "json_object");
    }

    #[test]
    fn test_provider_options_passthrough() {
        let request = Request::default()
            .model("gpt-4o")
            .messages(vec![Message::user("hi")])
            .provider_options(Some(serde_json::json!({
                "openai-compatible": {"repetition_penalty": 1.2, "top_k": 50}
            })));
        let (body, _) = translate_request(&request);
        assert_eq!(body["repetition_penalty"], 1.2);
        assert_eq!(body["top_k"], 50);
    }

    // === F-9: ImageData.detail passthrough ===

    #[test]
    fn test_translate_user_content_image_inline_includes_detail() {
        let parts = vec![ContentPart::Image {
            image: ImageData {
                url: None,
                data: Some(vec![0xFF, 0xD8]),
                media_type: Some("image/jpeg".into()),
                detail: Some("low".into()),
            },
        }];
        let (result, _) = translate_user_content(&parts);
        let json_str = serde_json::to_string(&result).unwrap();
        assert!(
            json_str.contains("\"low\""),
            "detail field 'low' should be in translated inline image: {json_str}"
        );
    }

    #[test]
    fn test_translate_user_content_image_url_includes_detail() {
        let parts = vec![
            ContentPart::Text {
                text: "Look".into(),
            },
            ContentPart::Image {
                image: ImageData {
                    url: Some("https://example.com/img.png".into()),
                    data: None,
                    media_type: None,
                    detail: Some("high".into()),
                },
            },
        ];
        let (result, _) = translate_user_content(&parts);
        let json_str = serde_json::to_string(&result).unwrap();
        assert!(
            json_str.contains("\"high\""),
            "detail field 'high' should be in translated output: {json_str}"
        );
    }

    // ===== OC-04: Response Translation =====

    #[test]
    fn test_parse_text_response() {
        let raw = serde_json::json!({
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello world!"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        });
        let headers = reqwest::header::HeaderMap::new();
        let resp = parse_response(raw, &headers).unwrap();
        assert_eq!(resp.text(), "Hello world!");
        assert_eq!(resp.id, "chatcmpl-123");
        assert_eq!(resp.model, "gpt-4o");
        assert_eq!(resp.provider, "openai-compatible");
        assert_eq!(resp.finish_reason.reason, "stop");
    }

    #[test]
    fn test_parse_tool_call_response() {
        let raw = serde_json::json!({
            "id": "chatcmpl-456",
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"city\":\"SF\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 15, "completion_tokens": 10, "total_tokens": 25}
        });
        let headers = reqwest::header::HeaderMap::new();
        let resp = parse_response(raw, &headers).unwrap();
        let tool_calls = resp.tool_calls();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "call_abc");
        assert_eq!(tool_calls[0].name, "get_weather");
        assert_eq!(resp.finish_reason.reason, "tool_calls");
    }

    #[test]
    fn test_parse_finish_reason_mapping() {
        for (raw_reason, expected) in &[
            ("stop", "stop"),
            ("length", "length"),
            ("tool_calls", "tool_calls"),
            ("content_filter", "content_filter"),
            ("unknown_reason", "other"),
        ] {
            let raw = serde_json::json!({
                "id": "chatcmpl-test",
                "model": "test",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": raw_reason
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            });
            let headers = reqwest::header::HeaderMap::new();
            let resp = parse_response(raw, &headers).unwrap();
            assert_eq!(
                resp.finish_reason.reason, *expected,
                "Failed for {raw_reason}"
            );
        }
    }

    #[test]
    fn test_parse_usage_prompt_completion_tokens() {
        let raw = serde_json::json!({
            "id": "chatcmpl-usage",
            "model": "test",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 42, "completion_tokens": 18, "total_tokens": 60}
        });
        let headers = reqwest::header::HeaderMap::new();
        let resp = parse_response(raw, &headers).unwrap();
        assert_eq!(resp.usage.input_tokens, 42);
        assert_eq!(resp.usage.output_tokens, 18);
        assert_eq!(resp.usage.total_tokens, 60);
        // Not available in Chat Completions
        assert!(resp.usage.reasoning_tokens.is_none());
        assert!(resp.usage.cache_read_tokens.is_none());
    }

    // ===== OC-06: complete() Implementation =====

    #[tokio::test]
    async fn test_complete_roundtrip() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/v1/chat/completions"))
            .and(wiremock::matchers::header(
                "authorization",
                "Bearer sk-test-key",
            ))
            .and(wiremock::matchers::header(
                "content-type",
                "application/json",
            ))
            .respond_with(
                wiremock::ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "id": "chatcmpl-test",
                    "model": "gpt-4o",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello from compat!"},
                        "finish_reason": "stop"
                    }],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
                })),
            )
            .mount(&server)
            .await;

        let adapter = OpenAICompatibleAdapter::new(
            SecretString::from("sk-test-key".to_string()),
            server.uri(),
        );
        let req = Request::default()
            .model("gpt-4o")
            .messages(vec![Message::user("Hi")]);
        let resp = adapter.complete(req).await.unwrap();
        assert_eq!(resp.text(), "Hello from compat!");
        assert_eq!(resp.provider, "openai-compatible");
        assert_eq!(resp.usage.input_tokens, 10);
    }

    #[tokio::test]
    async fn test_complete_error_roundtrip() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/v1/chat/completions"))
            .respond_with(
                wiremock::ResponseTemplate::new(429).set_body_json(serde_json::json!({
                    "error": {"message": "Rate limited"}
                })),
            )
            .mount(&server)
            .await;

        let adapter =
            OpenAICompatibleAdapter::new(SecretString::from("bad-key".to_string()), server.uri());
        let req = Request::default()
            .model("gpt-4o")
            .messages(vec![Message::user("Hi")]);
        let err = adapter.complete(req).await.unwrap_err();
        assert!(matches!(err.kind, unified_llm_types::ErrorKind::RateLimit));
    }

    #[tokio::test]
    async fn test_complete_request_body_structure() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/v1/chat/completions"))
            .respond_with(
                wiremock::ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "id": "chatcmpl-verify",
                    "model": "test-model",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop"
                    }],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
                })),
            )
            .mount(&server)
            .await;

        let adapter =
            OpenAICompatibleAdapter::new(SecretString::from("sk-test".to_string()), server.uri());
        let req = Request::default()
            .model("test-model")
            .messages(vec![Message::system("Be helpful."), Message::user("Hello")]);
        let resp = adapter.complete(req).await.unwrap();
        assert_eq!(resp.model, "test-model");
    }

    // ===== OC-08: stream() Implementation =====

    /// Build a Chat Completions SSE body from data-only lines.
    fn build_chat_sse_body(data_lines: &[&str]) -> String {
        data_lines
            .iter()
            .map(|d| format!("data: {d}\n\n"))
            .collect()
    }

    #[tokio::test]
    async fn test_stream_text_roundtrip() {
        let sse_body = build_chat_sse_body(&[
            r#"{"id":"chatcmpl-s1","model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}"#,
            r#"{"id":"chatcmpl-s1","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}"#,
            r#"{"id":"chatcmpl-s1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}"#,
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
        let req = Request::default()
            .model("gpt-4o")
            .messages(vec![Message::user("Hi")]);

        let stream = adapter.stream(req);
        let events: Vec<StreamEvent> = futures::StreamExt::collect::<Vec<_>>(stream)
            .await
            .into_iter()
            .filter_map(|r| r.ok())
            .collect();

        // Verify we got expected event sequence
        assert!(events
            .iter()
            .any(|e| e.event_type == StreamEventType::StreamStart));
        assert!(events
            .iter()
            .any(|e| e.event_type == StreamEventType::TextStart));
        let deltas: Vec<&str> = events
            .iter()
            .filter(|e| e.event_type == StreamEventType::TextDelta)
            .filter_map(|e| e.delta.as_deref())
            .collect();
        assert_eq!(deltas, vec!["Hello", " world"]);
        assert!(events
            .iter()
            .any(|e| e.event_type == StreamEventType::TextEnd));
        assert!(events
            .iter()
            .any(|e| e.event_type == StreamEventType::Finish));
    }

    #[tokio::test]
    async fn test_stream_error_response() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/v1/chat/completions"))
            .respond_with(
                wiremock::ResponseTemplate::new(429).set_body_json(serde_json::json!({
                    "error": {"message": "Rate limited"}
                })),
            )
            .mount(&server)
            .await;

        let adapter =
            OpenAICompatibleAdapter::new(SecretString::from("sk-test".to_string()), server.uri());
        let req = Request::default()
            .model("gpt-4o")
            .messages(vec![Message::user("Hi")]);

        let stream = adapter.stream(req);
        let results: Vec<Result<StreamEvent, Error>> =
            futures::StreamExt::collect::<Vec<_>>(stream).await;
        assert_eq!(results.len(), 1);
        let err = results[0].as_ref().unwrap_err();
        assert!(matches!(err.kind, unified_llm_types::ErrorKind::RateLimit));
    }

    // ===== OC-07: Streaming — ChatCompletionsStreamTranslator =====

    #[test]
    fn test_stream_translator_text_deltas() {
        let mut translator = ChatCompletionsStreamTranslator::new();

        // First chunk: stream start + text start + delta
        let data1 = serde_json::json!({
            "id": "chatcmpl-stream",
            "model": "gpt-4o",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hello"}, "finish_reason": null}]
        });
        let events1 = translator.process(&data1);
        assert!(events1
            .iter()
            .any(|e| e.event_type == StreamEventType::StreamStart));
        assert!(events1
            .iter()
            .any(|e| e.event_type == StreamEventType::TextStart));
        assert!(events1
            .iter()
            .any(|e| e.event_type == StreamEventType::TextDelta
                && e.delta.as_deref() == Some("Hello")));

        // Second chunk: just text delta
        let data2 = serde_json::json!({
            "id": "chatcmpl-stream",
            "choices": [{"index": 0, "delta": {"content": " world"}, "finish_reason": null}]
        });
        let events2 = translator.process(&data2);
        assert_eq!(events2.len(), 1);
        assert_eq!(events2[0].event_type, StreamEventType::TextDelta);
        assert_eq!(events2[0].delta.as_deref(), Some(" world"));

        // Finish chunk
        let data3 = serde_json::json!({
            "id": "chatcmpl-stream",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
        });
        let events3 = translator.process(&data3);
        assert!(events3
            .iter()
            .any(|e| e.event_type == StreamEventType::TextEnd));
        assert!(events3
            .iter()
            .any(|e| e.event_type == StreamEventType::Finish));
        let finish = events3
            .iter()
            .find(|e| e.event_type == StreamEventType::Finish)
            .unwrap();
        assert_eq!(finish.finish_reason.as_ref().unwrap().reason, "stop");
        assert_eq!(finish.usage.as_ref().unwrap().input_tokens, 5);
    }

    #[test]
    fn test_stream_translator_tool_call() {
        let mut translator = ChatCompletionsStreamTranslator::new();

        // First chunk: stream start
        let data0 = serde_json::json!({
            "id": "chatcmpl-tc",
            "model": "gpt-4o",
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]
        });
        translator.process(&data0);

        // Tool call start: id + function.name
        let data1 = serde_json::json!({
            "choices": [{"index": 0, "delta": {
                "tool_calls": [{"index": 0, "id": "call_abc", "type": "function", "function": {"name": "get_weather", "arguments": ""}}]
            }, "finish_reason": null}]
        });
        let events1 = translator.process(&data1);
        assert!(events1
            .iter()
            .any(|e| e.event_type == StreamEventType::ToolCallStart));
        let start = events1
            .iter()
            .find(|e| e.event_type == StreamEventType::ToolCallStart)
            .unwrap();
        assert_eq!(start.tool_call.as_ref().unwrap().name, "get_weather");

        // Tool call argument deltas
        let data2 = serde_json::json!({
            "choices": [{"index": 0, "delta": {
                "tool_calls": [{"index": 0, "function": {"arguments": "{\"city\":"}}]
            }, "finish_reason": null}]
        });
        let events2 = translator.process(&data2);
        assert!(events2
            .iter()
            .any(|e| e.event_type == StreamEventType::ToolCallDelta));

        let data3 = serde_json::json!({
            "choices": [{"index": 0, "delta": {
                "tool_calls": [{"index": 0, "function": {"arguments": "\"SF\"}"}}]
            }, "finish_reason": null}]
        });
        translator.process(&data3);

        // Finish
        let data4 = serde_json::json!({
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18}
        });
        let events4 = translator.process(&data4);
        assert!(events4
            .iter()
            .any(|e| e.event_type == StreamEventType::ToolCallEnd));
        let end = events4
            .iter()
            .find(|e| e.event_type == StreamEventType::ToolCallEnd)
            .unwrap();
        assert_eq!(end.tool_call.as_ref().unwrap().name, "get_weather");
        assert_eq!(end.tool_call.as_ref().unwrap().arguments["city"], "SF");
        assert!(events4
            .iter()
            .any(|e| e.event_type == StreamEventType::Finish));
    }

    #[test]
    fn test_stream_translator_finish_with_usage() {
        let mut translator = ChatCompletionsStreamTranslator::new();

        // Start
        let data1 = serde_json::json!({
            "id": "chatcmpl-u",
            "model": "gpt-4o",
            "choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": null}]
        });
        translator.process(&data1);

        // Finish with usage
        let data2 = serde_json::json!({
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
        });
        let events = translator.process(&data2);
        let finish = events
            .iter()
            .find(|e| e.event_type == StreamEventType::Finish)
            .unwrap();
        let usage = finish.usage.as_ref().unwrap();
        assert_eq!(usage.input_tokens, 20);
        assert_eq!(usage.output_tokens, 10);
        assert_eq!(usage.total_tokens, 30);
        assert!(usage.reasoning_tokens.is_none());
    }

    // ===== OC-05: Error Translation =====

    #[test]
    fn test_parse_error_401() {
        let body =
            serde_json::json!({"error": {"message": "Invalid API key", "code": "invalid_api_key"}});
        let headers = reqwest::header::HeaderMap::new();
        let err = parse_error(401, &headers, body);
        assert!(matches!(
            err.kind,
            unified_llm_types::ErrorKind::Authentication
        ));
        assert!(err.message.contains("Invalid API key"));
        assert_eq!(err.error_code, Some("invalid_api_key".into()));
    }

    #[test]
    fn test_parse_error_404() {
        let body = serde_json::json!({"error": {"message": "Model not found"}});
        let headers = reqwest::header::HeaderMap::new();
        let err = parse_error(404, &headers, body);
        assert!(matches!(err.kind, unified_llm_types::ErrorKind::NotFound));
    }

    #[test]
    fn test_parse_error_429_with_retry_after() {
        let body = serde_json::json!({"error": {"message": "Rate limited"}});
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("retry-after", "30".parse().unwrap());
        let err = parse_error(429, &headers, body);
        assert!(matches!(err.kind, unified_llm_types::ErrorKind::RateLimit));
        assert!(err.retry_after.is_some());
        assert_eq!(err.retry_after.unwrap().as_secs(), 30);
    }

    #[test]
    fn test_parse_error_500() {
        let body = serde_json::json!({"error": {"message": "Internal server error"}});
        let headers = reqwest::header::HeaderMap::new();
        let err = parse_error(500, &headers, body);
        assert!(matches!(err.kind, unified_llm_types::ErrorKind::Server));
    }

    // === H-3: Reasoning token support ===

    #[test]
    fn test_parse_response_with_reasoning_tokens() {
        let raw = serde_json::json!({
            "id": "chatcmpl-reasoning",
            "model": "o4-mini",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "The answer is 42."},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 30,
                "total_tokens": 50,
                "completion_tokens_details": {
                    "reasoning_tokens": 15
                }
            }
        });
        let headers = reqwest::header::HeaderMap::new();
        let resp = parse_response(raw, &headers).unwrap();
        assert_eq!(resp.usage.reasoning_tokens, Some(15));
        assert_eq!(resp.usage.input_tokens, 20);
        assert_eq!(resp.usage.output_tokens, 30);
    }

    #[test]
    fn test_stream_translator_reasoning_content() {
        let mut translator = ChatCompletionsStreamTranslator::new();

        // First chunk: stream start
        let data1 = serde_json::json!({
            "id": "chatcmpl-reason",
            "model": "o4-mini",
            "choices": [{"index": 0, "delta": {"role": "assistant", "reasoning_content": "Let me think..."}, "finish_reason": null}]
        });
        let events1 = translator.process(&data1);
        assert!(events1
            .iter()
            .any(|e| e.event_type == StreamEventType::StreamStart));
        assert!(events1
            .iter()
            .any(|e| e.event_type == StreamEventType::ReasoningStart));
        assert!(events1
            .iter()
            .any(|e| e.event_type == StreamEventType::ReasoningDelta
                && e.delta.as_deref() == Some("Let me think...")));

        // Second chunk: more reasoning
        let data2 = serde_json::json!({
            "choices": [{"index": 0, "delta": {"reasoning_content": " Step 1..."}, "finish_reason": null}]
        });
        let events2 = translator.process(&data2);
        assert_eq!(events2.len(), 1);
        assert_eq!(events2[0].event_type, StreamEventType::ReasoningDelta);

        // Third chunk: text content (reasoning done, text begins)
        let data3 = serde_json::json!({
            "choices": [{"index": 0, "delta": {"content": "42"}, "finish_reason": null}]
        });
        let events3 = translator.process(&data3);
        assert!(events3
            .iter()
            .any(|e| e.event_type == StreamEventType::TextStart));
        assert!(events3
            .iter()
            .any(|e| e.event_type == StreamEventType::TextDelta));

        // Finish
        let data4 = serde_json::json!({
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "completion_tokens_details": {"reasoning_tokens": 12}
            }
        });
        let events4 = translator.process(&data4);
        assert!(events4
            .iter()
            .any(|e| e.event_type == StreamEventType::ReasoningEnd));
        assert!(events4
            .iter()
            .any(|e| e.event_type == StreamEventType::TextEnd));
        let finish = events4
            .iter()
            .find(|e| e.event_type == StreamEventType::Finish)
            .unwrap();
        assert_eq!(finish.usage.as_ref().unwrap().reasoning_tokens, Some(12));
    }

    // === S-1: text_id propagation to TextDelta/TextEnd ===

    #[test]
    fn test_compat_stream_text_id_consistent_across_start_delta_end() {
        let mut translator = ChatCompletionsStreamTranslator::new();

        // First chunk: stream start + text start + delta
        let data1 = serde_json::json!({
            "id": "chatcmpl-tid",
            "model": "gpt-4o",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hello"}, "finish_reason": null}]
        });
        let events1 = translator.process(&data1);
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

        // Second chunk: just text delta
        let data2 = serde_json::json!({
            "id": "chatcmpl-tid",
            "choices": [{"index": 0, "delta": {"content": " world"}, "finish_reason": null}]
        });
        let events2 = translator.process(&data2);
        let text_delta2 = events2
            .iter()
            .find(|e| e.event_type == StreamEventType::TextDelta)
            .expect("Should emit TextDelta");
        assert_eq!(
            text_delta2.text_id.as_ref(),
            Some(&text_id),
            "Second TextDelta must carry same text_id"
        );

        // Finish chunk → TextEnd with text_id
        let data3 = serde_json::json!({
            "id": "chatcmpl-tid",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
        });
        let events3 = translator.process(&data3);
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
        let adapter = OpenAICompatibleAdapter::new_with_timeout(
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
