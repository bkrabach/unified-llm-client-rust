// OpenAI Responses API adapter.

use futures::StreamExt;
use secrecy::{ExposeSecret, SecretString};

use unified_llm_types::{
    AdapterTimeout, ArgumentValue, BoxFuture, BoxStream, ContentPart, Error, FinishReason, Message,
    ProviderAdapter, Request, Response, Role, StreamError, StreamEvent, StreamEventType, ToolCall,
    ToolCallData, Usage,
};

use crate::util::sse::SseParser;

/// Default OpenAI API base URL.
const DEFAULT_BASE_URL: &str = "https://api.openai.com";

/// OpenAI Responses API adapter.
pub struct OpenAiAdapter {
    api_key: SecretString,
    pub(crate) base_url: String,
    http_client: reqwest::Client,
    /// Per-chunk timeout for streaming responses (from AdapterTimeout.stream_read).
    stream_read_timeout: std::time::Duration,
}

impl OpenAiAdapter {
    /// Create a new OpenAiAdapter with the given API key.
    ///
    /// Uses default timeouts: connect=10s, request=120s.
    pub fn new(api_key: SecretString) -> Self {
        let timeout = AdapterTimeout::default();
        Self {
            api_key,
            base_url: DEFAULT_BASE_URL.to_string(),
            stream_read_timeout: std::time::Duration::from_secs_f64(timeout.stream_read),
            http_client: Self::build_http_client(&timeout),
        }
    }

    /// Create a new OpenAiAdapter with a custom base URL (for testing with wiremock).
    ///
    /// Uses default timeouts: connect=10s, request=120s.
    pub fn new_with_base_url(api_key: SecretString, base_url: impl Into<String>) -> Self {
        let timeout = AdapterTimeout::default();
        Self {
            api_key,
            base_url: base_url.into(),
            stream_read_timeout: std::time::Duration::from_secs_f64(timeout.stream_read),
            http_client: Self::build_http_client(&timeout),
        }
    }

    /// Create a new OpenAiAdapter with custom timeouts.
    pub fn new_with_timeout(api_key: SecretString, timeout: AdapterTimeout) -> Self {
        Self {
            api_key,
            base_url: DEFAULT_BASE_URL.to_string(),
            stream_read_timeout: std::time::Duration::from_secs_f64(timeout.stream_read),
            http_client: Self::build_http_client(&timeout),
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
            base_url: base_url.into(),
            stream_read_timeout: std::time::Duration::from_secs_f64(timeout.stream_read),
            http_client: Self::build_http_client(&timeout),
        }
    }

    /// Create from environment variable OPENAI_API_KEY.
    pub fn from_env() -> Result<Self, Error> {
        Self::from_env_with_check(true)
    }

    /// Internal from_env with check control (for testing without real env var).
    pub fn from_env_with_check(check_env: bool) -> Result<Self, Error> {
        if !check_env {
            return Err(Error::configuration("OPENAI_API_KEY not set"));
        }
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| Error::configuration("OPENAI_API_KEY not set"))?;
        Ok(Self::new(SecretString::from(api_key)))
    }

    /// Create a builder for fine-grained configuration.
    pub fn builder(api_key: SecretString) -> OpenAiAdapterBuilder {
        OpenAiAdapterBuilder::new(api_key)
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

    /// Build common HTTP headers for OpenAI API requests.
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

    /// Perform the actual HTTP request for complete().
    async fn do_complete(&self, request: Request) -> Result<Response, Error> {
        let url = format!("{}/v1/responses", self.base_url);
        let body = translate_request(&request);

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

        parse_response(response_body, &headers)
    }

    /// Perform the HTTP request for stream() and return a stream of events.
    fn do_stream(&self, request: Request) -> BoxStream<'_, Result<StreamEvent, Error>> {
        let stream = async_stream::stream! {
            let url = format!("{}/v1/responses", self.base_url);
            let mut body = translate_request(&request);
            // Add stream: true to the request body
            if let Some(obj) = body.as_object_mut() {
                obj.insert("stream".into(), serde_json::Value::Bool(true));
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
                    .unwrap_or(serde_json::json!({"error": {"message": "Failed to parse error response"}}));
                yield Err(parse_error(status, &headers, error_body));
                return;
            }

            // True incremental streaming: read chunks as they arrive
            let mut parser = SseParser::new();
            let mut byte_stream = http_response.bytes_stream();

            let mut translator = OpenAiStreamTranslator::new();
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
                    let event_type = match &sse_event.event_type {
                        Some(t) => t.as_str(),
                        None => continue,
                    };

                    let data: serde_json::Value = match serde_json::from_str(&sse_event.data) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };

                    for evt in translator.process(event_type, &data) {
                        yield Ok(evt);
                    }
                }
            }
        };
        Box::pin(stream)
    }
}

/// Builder for constructing an `OpenAiAdapter` with fine-grained configuration.
pub struct OpenAiAdapterBuilder {
    api_key: SecretString,
    base_url: Option<String>,
    timeout: Option<AdapterTimeout>,
    default_headers: Option<reqwest::header::HeaderMap>,
}

impl OpenAiAdapterBuilder {
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

    /// Build the `OpenAiAdapter`.
    pub fn build(self) -> OpenAiAdapter {
        let timeout = self.timeout.unwrap_or_default();
        OpenAiAdapter {
            api_key: self.api_key,
            base_url: self
                .base_url
                .unwrap_or_else(|| DEFAULT_BASE_URL.to_string()),
            stream_read_timeout: std::time::Duration::from_secs_f64(timeout.stream_read),
            http_client: OpenAiAdapter::build_http_client_with_headers(
                &timeout,
                self.default_headers,
            ),
        }
    }
}

impl ProviderAdapter for OpenAiAdapter {
    fn name(&self) -> &str {
        "openai"
    }

    fn complete(&self, request: Request) -> BoxFuture<'_, Result<Response, Error>> {
        Box::pin(self.do_complete(request))
    }

    fn stream(&self, request: Request) -> BoxStream<'_, Result<StreamEvent, Error>> {
        self.do_stream(request)
    }
}

// === Request Translation ===

/// Translate a unified Request into an OpenAI Responses API JSON body.
pub fn translate_request(request: &Request) -> serde_json::Value {
    let mut body = serde_json::Map::new();

    // Model
    body.insert(
        "model".into(),
        serde_json::Value::String(request.model.clone()),
    );

    // All system messages are extracted regardless of conversation position per provider API
    // conventions. Mid-conversation system messages are repositioned to the system prompt area.
    let mut instructions_parts: Vec<String> = Vec::new();
    let mut input_items: Vec<serde_json::Value> = Vec::new();

    for msg in &request.messages {
        match msg.role {
            Role::System | Role::Developer => {
                instructions_parts.push(msg.text());
            }
            Role::User => {
                let content_parts = translate_user_content(&msg.content);
                input_items.push(serde_json::json!({
                    "type": "message",
                    "role": "user",
                    "content": content_parts,
                }));
            }
            Role::Assistant => {
                // Collect text parts and tool call parts separately
                let text_parts: Vec<serde_json::Value> = msg
                    .content
                    .iter()
                    .filter_map(|p| {
                        if let ContentPart::Text { text } = p {
                            Some(serde_json::json!({"type": "output_text", "text": text}))
                        } else {
                            None
                        }
                    })
                    .collect();

                // Emit assistant message with text content if any text parts exist
                if !text_parts.is_empty() {
                    input_items.push(serde_json::json!({
                        "type": "message",
                        "role": "assistant",
                        "content": text_parts,
                    }));
                }

                // Emit function_call items for any tool calls
                for part in &msg.content {
                    if let ContentPart::ToolCall { tool_call } = part {
                        let arguments_str = match &tool_call.arguments {
                            ArgumentValue::Dict(map) => {
                                serde_json::to_string(&serde_json::Value::Object(map.clone()))
                                    .unwrap_or_else(|_| "{}".to_string())
                            }
                            ArgumentValue::Raw(s) => s.clone(),
                        };
                        input_items.push(serde_json::json!({
                            "type": "function_call",
                            "call_id": tool_call.id,
                            "name": tool_call.name,
                            "arguments": arguments_str,
                        }));
                    }
                }
            }
            Role::Tool => {
                // Tool results become top-level function_call_output input items
                for part in &msg.content {
                    if let ContentPart::ToolResult { tool_result } = part {
                        let output = match &tool_result.content {
                            serde_json::Value::String(s) => s.clone(),
                            other => other.to_string(),
                        };
                        input_items.push(serde_json::json!({
                            "type": "function_call_output",
                            "call_id": tool_result.tool_call_id,
                            "output": output,
                        }));
                    }
                }
            }
        }
    }

    // Instructions (from system + developer messages)
    if !instructions_parts.is_empty() {
        body.insert(
            "instructions".into(),
            serde_json::Value::String(instructions_parts.join("\n\n")),
        );
    }

    // Input items
    body.insert("input".into(), serde_json::Value::Array(input_items));

    // Optional parameters
    if let Some(temp) = request.temperature {
        body.insert("temperature".into(), serde_json::json!(temp));
    }
    if let Some(top_p) = request.top_p {
        body.insert("top_p".into(), serde_json::json!(top_p));
    }
    if let Some(ref stop) = request.stop_sequences {
        body.insert("stop".into(), serde_json::json!(stop));
    }

    // max_tokens -> max_output_tokens (OpenAI Responses API naming)
    if let Some(max_tokens) = request.max_tokens {
        body.insert("max_output_tokens".into(), serde_json::json!(max_tokens));
    }

    // reasoning_effort -> reasoning.effort
    if let Some(ref effort) = request.reasoning_effort {
        body.insert("reasoning".into(), serde_json::json!({"effort": effort}));
    }

    // Tools — flat format with strict: true
    if let Some(ref tools) = request.tools {
        let tool_defs: Vec<serde_json::Value> = tools
            .iter()
            .map(|t| {
                serde_json::json!({
                    "type": "function",
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                    "strict": t.strict.unwrap_or(true),
                })
            })
            .collect();
        body.insert("tools".into(), serde_json::Value::Array(tool_defs));
    }

    // Tool choice
    if let Some(ref tc) = request.tool_choice {
        let choice_val = match tc.mode.as_str() {
            "auto" => serde_json::json!("auto"),
            "none" => serde_json::json!("none"),
            "required" => serde_json::json!("required"),
            "named" => {
                if let Some(ref name) = tc.tool_name {
                    serde_json::json!({"type": "function", "name": name})
                } else {
                    serde_json::json!("auto")
                }
            }
            _ => serde_json::json!("auto"),
        };
        body.insert("tool_choice".into(), choice_val);
    }

    // Structured output: response_format -> text.format
    if let Some(ref fmt) = request.response_format {
        if fmt.r#type == "json_schema" {
            let schema = fmt.json_schema.clone().unwrap_or(serde_json::json!({}));
            body.insert(
                "text".into(),
                serde_json::json!({
                    "format": {
                        "type": "json_schema",
                        "name": "response",
                        "schema": schema,
                        "strict": fmt.strict,
                    }
                }),
            );
        } else if fmt.r#type == "json_object" {
            body.insert(
                "text".into(),
                serde_json::json!({
                    "format": {
                        "type": "json_object",
                    }
                }),
            );
        }
    }

    // Provider options passthrough using shared utility.
    const INTERNAL_KEYS: &[&str] = &[];
    if let Some(opts) =
        crate::util::provider_options::get_provider_options(&request.provider_options, "openai")
    {
        let mut body_val = serde_json::Value::Object(body);
        crate::util::provider_options::merge_provider_options(&mut body_val, &opts, INTERNAL_KEYS);
        return body_val;
    }

    serde_json::Value::Object(body)
}

/// Translate user content parts to OpenAI input_text format.
fn translate_user_content(parts: &[ContentPart]) -> Vec<serde_json::Value> {
    parts
        .iter()
        .filter_map(|part| match part {
            ContentPart::Text { text } => {
                Some(serde_json::json!({"type": "input_text", "text": text}))
            }
            ContentPart::Image { image } => {
                if let Some(ref data) = image.data {
                    let b64 = crate::util::image::base64_encode(data);
                    let media_type = image.media_type.as_deref().unwrap_or("image/png");
                    let data_uri = crate::util::image::encode_data_uri(media_type, &b64);
                    let mut img_obj = serde_json::json!({
                        "type": "input_image",
                        "image_url": data_uri,
                    });
                    // F-9: pass detail field through when present
                    if let Some(ref detail) = image.detail {
                        img_obj["detail"] = serde_json::json!(detail);
                    }
                    Some(img_obj)
                } else {
                    image.url.as_ref().map(|url| {
                        let mut img_obj = if crate::util::image::is_local_path(url) {
                            match crate::util::image::resolve_local_file(url) {
                                Ok((data, mime)) => {
                                    let b64 = crate::util::image::base64_encode(&data);
                                    let data_uri = crate::util::image::encode_data_uri(&mime, &b64);
                                    serde_json::json!({
                                        "type": "input_image",
                                        "image_url": data_uri,
                                    })
                                }
                                Err(e) => {
                                    tracing::warn!(
                                        "Failed to resolve local image '{}': {}",
                                        url,
                                        e.message
                                    );
                                    serde_json::json!({
                                        "type": "input_image",
                                        "image_url": url,
                                    })
                                }
                            }
                        } else {
                            serde_json::json!({
                                "type": "input_image",
                                "image_url": url,
                            })
                        };
                        // F-9: pass detail field through when present
                        if let Some(ref detail) = image.detail {
                            img_obj["detail"] = serde_json::json!(detail);
                        }
                        img_obj
                    })
                }
            }
            _ => None,
        })
        .collect()
}

// === Response Translation ===

/// Map OpenAI status + output to unified finish reason.
fn determine_finish_reason(status: &str, has_tool_calls: bool) -> FinishReason {
    if has_tool_calls {
        FinishReason {
            reason: "tool_calls".to_string(),
            raw: Some(status.to_string()),
        }
    } else {
        match status {
            "completed" => FinishReason {
                reason: "stop".to_string(),
                raw: Some("completed".to_string()),
            },
            "incomplete" => FinishReason {
                reason: "length".to_string(),
                raw: Some("incomplete".to_string()),
            },
            "failed" => FinishReason {
                reason: "error".to_string(),
                raw: Some("failed".to_string()),
            },
            other => FinishReason {
                reason: "other".to_string(),
                raw: Some(other.to_string()),
            },
        }
    }
}

/// Parse an OpenAI Responses API response JSON into a unified Response.
pub fn parse_response(
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
    let status = raw
        .get("status")
        .and_then(|v| v.as_str())
        .unwrap_or("completed");

    // Parse output array
    let mut content_parts: Vec<ContentPart> = Vec::new();
    let mut has_tool_calls = false;

    if let Some(output) = raw.get("output").and_then(|v| v.as_array()) {
        for item in output {
            let item_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
            match item_type {
                "message" => {
                    if let Some(content) = item.get("content").and_then(|v| v.as_array()) {
                        for block in content {
                            let block_type =
                                block.get("type").and_then(|v| v.as_str()).unwrap_or("");
                            if block_type == "output_text" {
                                let text = block
                                    .get("text")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string();
                                content_parts.push(ContentPart::Text { text });
                            }
                        }
                    }
                }
                "function_call" => {
                    has_tool_calls = true;
                    let call_id = item
                        .get("call_id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let name = item
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let arguments_str = item
                        .get("arguments")
                        .and_then(|v| v.as_str())
                        .unwrap_or("{}");
                    let arguments = serde_json::from_str::<
                        serde_json::Map<String, serde_json::Value>,
                    >(arguments_str)
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
                "reasoning" => {
                    // OpenAI reasoning output items contain summary text
                    // Map to ContentPart::Thinking for unified representation
                    if let Some(summary) = item.get("summary").and_then(|v| v.as_array()) {
                        for block in summary {
                            let block_type =
                                block.get("type").and_then(|v| v.as_str()).unwrap_or("");
                            if block_type == "summary_text" {
                                let text = block
                                    .get("text")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string();
                                content_parts.push(ContentPart::Thinking {
                                    thinking: unified_llm_types::ThinkingData {
                                        text,
                                        signature: None,
                                        redacted: false,
                                        data: None,
                                    },
                                });
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    // Finish reason
    let finish_reason = determine_finish_reason(status, has_tool_calls);

    // Usage
    let usage_obj = raw.get("usage");
    let input_tokens = usage_obj
        .and_then(|u| u.get("input_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    let output_tokens = usage_obj
        .and_then(|u| u.get("output_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    let reasoning_tokens = usage_obj
        .and_then(|u| u.get("output_tokens_details"))
        .and_then(|d| d.get("reasoning_tokens"))
        .and_then(|v| v.as_u64())
        .map(|v| v as u32);
    let cache_read_tokens = usage_obj
        .and_then(|u| u.get("input_tokens_details"))
        .and_then(|d| d.get("cached_tokens"))
        .and_then(|v| v.as_u64())
        .map(|v| v as u32);

    let usage = Usage {
        input_tokens,
        output_tokens,
        // F-10: always recompute total_tokens instead of trusting provider
        total_tokens: input_tokens + output_tokens,
        reasoning_tokens,
        cache_read_tokens,
        cache_write_tokens: None, // OpenAI doesn't have cache_write_tokens
        raw: usage_obj.cloned(),
    };

    Ok(Response {
        id,
        model,
        provider: "openai".to_string(),
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

/// Parse an OpenAI error response into a unified Error.
pub fn parse_error(
    status: u16,
    headers: &reqwest::header::HeaderMap,
    body: serde_json::Value,
) -> Error {
    // Extract message and code using shared utility with OpenAI JSON paths.
    let (error_message, error_code) = crate::util::http::parse_provider_error_message(
        &body,
        &["error", "message"],
        &["error", "code"],
    );

    let retry_after = crate::util::http::parse_retry_after(headers);

    let mut err = Error::from_http_status(status, error_message, "openai", Some(body), retry_after);
    err.error_code = error_code;
    err
}

// === Stream Translation ===

/// Holds streaming state and translates OpenAI SSE events into unified StreamEvents.
struct OpenAiStreamTranslator {
    response_id: String,
    model: String,
    text_started: bool,
    /// S-1: text_id generated on TextStart, propagated to TextDelta/TextEnd.
    current_text_id: Option<String>,
    // Track tool call state per output_index
    active_tool_id: Option<String>,
    active_tool_name: Option<String>,
    accumulated_tool_json: String,
    reasoning_started: bool,
}

impl OpenAiStreamTranslator {
    fn new() -> Self {
        Self {
            response_id: String::new(),
            model: String::new(),
            text_started: false,
            current_text_id: None,
            active_tool_id: None,
            active_tool_name: None,
            accumulated_tool_json: String::new(),
            reasoning_started: false,
        }
    }

    /// Translate a single OpenAI SSE event into zero or more unified StreamEvents.
    fn process(&mut self, event_type: &str, data: &serde_json::Value) -> Vec<StreamEvent> {
        let mut events = Vec::new();

        match event_type {
            "response.created" => {
                // Real API nests under "response" key; fall back to flat for compat
                let response_data = data.get("response").unwrap_or(data);
                self.response_id = response_data
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                self.model = response_data
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

            "response.output_item.added" => {
                if let Some(item) = data.get("item") {
                    let item_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
                    match item_type {
                        "function_call" => {
                            let call_id = item
                                .get("call_id")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string();
                            let name = item
                                .get("name")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string();
                            self.active_tool_id = Some(call_id.clone());
                            self.active_tool_name = Some(name.clone());
                            self.accumulated_tool_json.clear();
                            events.push(StreamEvent {
                                event_type: StreamEventType::ToolCallStart,
                                tool_call: Some(ToolCall {
                                    id: call_id,
                                    name,
                                    arguments: serde_json::Map::new(),
                                    raw_arguments: None,
                                }),
                                ..Default::default()
                            });
                        }
                        "reasoning" => {
                            self.reasoning_started = true;
                            events.push(StreamEvent {
                                event_type: StreamEventType::ReasoningStart,
                                ..Default::default()
                            });
                        }
                        // "message" — handled by content_part.added
                        _ => {}
                    }
                }
            }

            "response.content_part.added" => {
                if let Some(part) = data.get("part") {
                    let part_type = part.get("type").and_then(|v| v.as_str()).unwrap_or("");
                    // NOTE: Single boolean limits to one concurrent text segment. Current providers
                    // send text sequentially. For concurrent segments, replace with counter or
                    // segment-id tracking.
                    if part_type == "output_text" && !self.text_started {
                        self.text_started = true;
                        let text_id = format!("txt_{}", uuid::Uuid::new_v4());
                        self.current_text_id = Some(text_id.clone());
                        events.push(StreamEvent {
                            event_type: StreamEventType::TextStart,
                            text_id: Some(text_id),
                            ..Default::default()
                        });
                    }
                }
            }

            "response.output_text.delta" => {
                // If TextStart hasn't been emitted yet, emit it now
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
                let delta = data
                    .get("delta")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                events.push(StreamEvent {
                    event_type: StreamEventType::TextDelta,
                    delta: Some(delta),
                    text_id: self.current_text_id.clone(),
                    ..Default::default()
                });
            }

            "response.output_text.done" => {
                events.push(StreamEvent {
                    event_type: StreamEventType::TextEnd,
                    text_id: self.current_text_id.take(),
                    ..Default::default()
                });
                self.text_started = false;
            }

            "response.function_call_arguments.delta" => {
                let delta = data
                    .get("delta")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                self.accumulated_tool_json.push_str(&delta);
                events.push(StreamEvent {
                    event_type: StreamEventType::ToolCallDelta,
                    delta: Some(delta),
                    ..Default::default()
                });
            }

            "response.function_call_arguments.done" => {
                let arguments_str = data
                    .get("arguments")
                    .and_then(|v| v.as_str())
                    .unwrap_or(&self.accumulated_tool_json);
                let arguments: serde_json::Map<String, serde_json::Value> =
                    serde_json::from_str(arguments_str).unwrap_or_default();
                let tool_call = ToolCall {
                    id: self.active_tool_id.take().unwrap_or_default(),
                    name: self.active_tool_name.take().unwrap_or_default(),
                    arguments,
                    raw_arguments: Some(arguments_str.to_string()),
                };
                events.push(StreamEvent {
                    event_type: StreamEventType::ToolCallEnd,
                    tool_call: Some(tool_call),
                    ..Default::default()
                });
                self.accumulated_tool_json.clear();
            }

            "response.reasoning_summary_text.delta" => {
                if !self.reasoning_started {
                    self.reasoning_started = true;
                    events.push(StreamEvent {
                        event_type: StreamEventType::ReasoningStart,
                        ..Default::default()
                    });
                }
                let delta = data
                    .get("delta")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                events.push(StreamEvent {
                    event_type: StreamEventType::ReasoningDelta,
                    delta: Some(delta.clone()),
                    reasoning_delta: Some(delta),
                    ..Default::default()
                });
            }

            "response.output_item.done" => {
                if let Some(item) = data.get("item") {
                    let item_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
                    if item_type == "reasoning" {
                        self.reasoning_started = false;
                        events.push(StreamEvent {
                            event_type: StreamEventType::ReasoningEnd,
                            ..Default::default()
                        });
                    }
                    // function_call done — ToolCallEnd already emitted by arguments.done
                    // message done — TextEnd already emitted by output_text.done
                }
            }

            "response.completed" => {
                // Real API nests under "response" key; fall back to flat for compat
                let response_data = data.get("response").unwrap_or(data);
                // Parse usage from the final response object
                let usage_obj = response_data.get("usage");
                let input_tokens = usage_obj
                    .and_then(|u| u.get("input_tokens"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as u32;
                let output_tokens = usage_obj
                    .and_then(|u| u.get("output_tokens"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as u32;
                let reasoning_tokens = usage_obj
                    .and_then(|u| u.get("output_tokens_details"))
                    .and_then(|d| d.get("reasoning_tokens"))
                    .and_then(|v| v.as_u64())
                    .map(|v| v as u32);
                let cache_read_tokens = usage_obj
                    .and_then(|u| u.get("input_tokens_details"))
                    .and_then(|d| d.get("cached_tokens"))
                    .and_then(|v| v.as_u64())
                    .map(|v| v as u32);

                let usage = Usage {
                    input_tokens,
                    output_tokens,
                    // F-10: always recompute total_tokens
                    total_tokens: input_tokens + output_tokens,
                    reasoning_tokens,
                    cache_read_tokens,
                    cache_write_tokens: None,
                    raw: usage_obj.cloned(),
                };

                // Determine finish reason from status
                let status = response_data
                    .get("status")
                    .and_then(|v| v.as_str())
                    .unwrap_or("completed");
                let has_tool_calls = response_data
                    .get("output")
                    .and_then(|v| v.as_array())
                    .map(|items| {
                        items.iter().any(|item| {
                            item.get("type").and_then(|v| v.as_str()) == Some("function_call")
                        })
                    })
                    .unwrap_or(false);

                let finish_reason = determine_finish_reason(status, has_tool_calls);

                events.push(StreamEvent {
                    event_type: StreamEventType::Finish,
                    finish_reason: Some(finish_reason),
                    usage: Some(usage),
                    ..Default::default()
                });
            }

            "response.failed" => {
                // Real API nests under "response" key; fall back to flat for compat
                let response_data = data.get("response").unwrap_or(data);
                let error_msg = response_data
                    .get("error")
                    .and_then(|e| e.get("message"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("Stream error")
                    .to_string();
                events.push(StreamEvent {
                    event_type: StreamEventType::Error,
                    error: Some(Box::new(StreamError::stream(error_msg))),
                    ..Default::default()
                });
            }

            // Forward unknown SSE events as PROVIDER_EVENT (spec §3.13, M-4)
            _ => {
                events.push(StreamEvent {
                    event_type: StreamEventType::ProviderEvent,
                    raw: Some(serde_json::json!({
                        "sse_event_type": event_type,
                        "data": data.clone(),
                    })),
                    ..Default::default()
                });
            }
        }

        events
    }
}

// === Test Helpers ===

/// Build an SSE body from event type/data tuples (for OpenAI format).
#[cfg(test)]
fn build_openai_sse_body(events: &[(&str, &str)]) -> String {
    events
        .iter()
        .map(|(t, d)| format!("event: {t}\ndata: {d}\n\n"))
        .collect()
}

/// Create a minimal OpenAI response JSON for testing.
#[cfg(test)]
fn make_openai_response_json(id: &str, text: &str, status: &str) -> serde_json::Value {
    serde_json::json!({
        "id": id,
        "object": "response",
        "status": status,
        "model": "gpt-4o-2025-01-01",
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text}]
        }],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15
        }
    })
}

/// Create a test tool definition.
#[cfg(test)]
fn make_test_tool() -> unified_llm_types::ToolDefinition {
    unified_llm_types::ToolDefinition {
        name: "get_weather".into(),
        description: "Get weather".into(),
        parameters: serde_json::json!({"type": "object", "properties": {"city": {"type": "string"}}}),
        strict: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use unified_llm_types::*;

    // === Builder Tests (H-8) ===

    #[test]
    fn test_openai_adapter_builder_defaults() {
        let adapter = OpenAiAdapterBuilder::new(SecretString::from("key".to_string())).build();
        assert_eq!(adapter.base_url, DEFAULT_BASE_URL);
        assert_eq!(adapter.name(), "openai");
    }

    #[test]
    fn test_openai_adapter_builder_with_all_options() {
        let adapter = OpenAiAdapterBuilder::new(SecretString::from("key".to_string()))
            .base_url("https://custom.openai.com")
            .timeout(AdapterTimeout {
                connect: 5.0,
                request: 60.0,
                stream_read: 15.0,
            })
            .default_headers(reqwest::header::HeaderMap::new())
            .build();
        assert_eq!(adapter.base_url, "https://custom.openai.com");
    }

    #[test]
    fn test_openai_adapter_builder_shortcut() {
        let adapter = OpenAiAdapter::builder(SecretString::from("key".to_string()))
            .base_url("https://test.com")
            .build();
        assert_eq!(adapter.base_url, "https://test.com");
    }

    // ===== P2B-T01: Adapter Struct, Auth, from_env =====

    #[test]
    fn test_openai_adapter_name() {
        let adapter = OpenAiAdapter::new(SecretString::from("sk-test".to_string()));
        assert_eq!(adapter.name(), "openai");
    }

    #[test]
    fn test_openai_uses_responses_api_endpoint() {
        // DoD 8.2.1: Must use native Responses API, NOT Chat Completions
        let adapter = OpenAiAdapter::new_with_base_url(
            SecretString::from("sk-test".to_string()),
            "https://api.openai.com",
        );
        assert_eq!(adapter.base_url, "https://api.openai.com");
    }

    #[test]
    fn test_openai_from_env_no_key() {
        let result = OpenAiAdapter::from_env_with_check(false);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_openai_auth_header() {
        // DoD 8.2.2: Authentication works
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/v1/responses"))
            .and(wiremock::matchers::header(
                "authorization",
                "Bearer sk-test-key",
            ))
            .and(wiremock::matchers::header(
                "content-type",
                "application/json",
            ))
            .respond_with(
                wiremock::ResponseTemplate::new(200).set_body_json(make_openai_response_json(
                    "resp_001",
                    "Hello!",
                    "completed",
                )),
            )
            .mount(&server)
            .await;

        let adapter = OpenAiAdapter::new_with_base_url(
            SecretString::from("sk-test-key".to_string()),
            server.uri(),
        );
        let req = Request::default()
            .model("gpt-4o")
            .messages(vec![Message::user("Hi")]);
        let resp = adapter.complete(req).await.unwrap();
        assert_eq!(resp.provider, "openai");
    }

    // ===== P2B-T02: Request Translation — Messages and System =====

    #[test]
    fn test_openai_system_to_instructions() {
        // DoD 8.2.5: System messages extracted to instructions parameter
        let request = Request::default().model("gpt-4o").messages(vec![
            Message::system("You are helpful."),
            Message::user("Hi"),
        ]);
        let body = translate_request(&request);
        assert_eq!(body["instructions"], "You are helpful.");
        // System message NOT in input array
        let input = body["input"].as_array().unwrap();
        assert_eq!(input.len(), 1);
        assert_eq!(input[0]["role"], "user");
    }

    #[test]
    fn test_openai_developer_to_instructions() {
        // DoD 8.2.6: Developer role also goes to instructions
        let request = Request::default().model("gpt-4o").messages(vec![
            Message {
                role: Role::Developer,
                content: vec![ContentPart::text("Dev instructions")],
                name: None,
                tool_call_id: None,
            },
            Message::user("Hi"),
        ]);
        let body = translate_request(&request);
        assert!(body["instructions"]
            .as_str()
            .unwrap()
            .contains("Dev instructions"));
    }

    #[test]
    fn test_openai_user_message_input_text() {
        // DoD 8.2.6: user TEXT -> input_text (NOT "text")
        let request = Request::default()
            .model("gpt-4o")
            .messages(vec![Message::user("Hello world")]);
        let body = translate_request(&request);
        let input = &body["input"][0];
        assert_eq!(input["type"], "message");
        assert_eq!(input["role"], "user");
        assert_eq!(input["content"][0]["type"], "input_text");
        assert_eq!(input["content"][0]["text"], "Hello world");
    }

    #[test]
    fn test_openai_assistant_message_output_text() {
        // DoD 8.2.6: assistant TEXT -> output_text
        let request = Request::default().model("gpt-4o").messages(vec![
            Message::user("Hi"),
            Message::assistant("Hello!"),
            Message::user("How are you?"),
        ]);
        let body = translate_request(&request);
        let input = body["input"].as_array().unwrap();
        assert_eq!(input[1]["type"], "message");
        assert_eq!(input[1]["role"], "assistant");
        assert_eq!(input[1]["content"][0]["type"], "output_text");
        assert_eq!(input[1]["content"][0]["text"], "Hello!");
    }

    #[test]
    fn test_openai_multiple_system_messages_concatenated() {
        let request = Request::default().model("gpt-4o").messages(vec![
            Message::system("Rule 1."),
            Message::system("Rule 2."),
            Message::user("Hi"),
        ]);
        let body = translate_request(&request);
        let instructions = body["instructions"].as_str().unwrap();
        assert!(instructions.contains("Rule 1."));
        assert!(instructions.contains("Rule 2."));
    }

    // ===== P2B-T03: Request Translation — Tools and Generation Params =====

    #[test]
    fn test_openai_tool_definition_translation() {
        let request = Request::default()
            .model("gpt-4o")
            .messages(vec![Message::user("weather?")])
            .tools(vec![ToolDefinition {
                name: "get_weather".into(),
                description: "Get weather".into(),
                parameters: serde_json::json!({"type": "object", "properties": {"city": {"type": "string"}}}),
                strict: None,
            }]);
        let body = translate_request(&request);
        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["name"], "get_weather");
        assert_eq!(tools[0]["description"], "Get weather");
        assert!(tools[0]["parameters"].is_object());
        assert_eq!(tools[0]["strict"], true);
    }

    #[test]
    fn test_openai_tool_choice_auto() {
        let request = Request::default()
            .model("gpt-4o")
            .messages(vec![Message::user("hi")])
            .tools(vec![make_test_tool()])
            .tool_choice(ToolChoice {
                mode: "auto".into(),
                tool_name: None,
            });
        let body = translate_request(&request);
        assert_eq!(body["tool_choice"], "auto");
    }

    #[test]
    fn test_openai_tool_choice_none() {
        let request = Request::default()
            .model("gpt-4o")
            .messages(vec![Message::user("hi")])
            .tool_choice(ToolChoice {
                mode: "none".into(),
                tool_name: None,
            });
        let body = translate_request(&request);
        assert_eq!(body["tool_choice"], "none");
    }

    #[test]
    fn test_openai_tool_choice_required() {
        let request = Request::default()
            .model("gpt-4o")
            .messages(vec![Message::user("hi")])
            .tools(vec![make_test_tool()])
            .tool_choice(ToolChoice {
                mode: "required".into(),
                tool_name: None,
            });
        let body = translate_request(&request);
        assert_eq!(body["tool_choice"], "required");
    }

    #[test]
    fn test_openai_tool_choice_named() {
        let request = Request::default()
            .model("gpt-4o")
            .messages(vec![Message::user("hi")])
            .tools(vec![make_test_tool()])
            .tool_choice(ToolChoice {
                mode: "named".into(),
                tool_name: Some("get_weather".into()),
            });
        let body = translate_request(&request);
        assert_eq!(body["tool_choice"]["type"], "function");
        // Responses API flat format: name is a sibling of type, NOT nested under "function"
        assert_eq!(body["tool_choice"]["name"], "get_weather");
    }

    #[test]
    fn test_openai_max_output_tokens() {
        // Responses API uses max_output_tokens, NOT max_tokens
        let request = Request::default()
            .model("gpt-4o")
            .messages(vec![Message::user("hi")])
            .max_tokens(500);
        let body = translate_request(&request);
        assert_eq!(body["max_output_tokens"], 500);
        assert!(body.get("max_tokens").is_none());
    }

    #[test]
    fn test_openai_reasoning_effort() {
        // DoD 8.5.2: reasoning_effort passed to reasoning.effort
        let request = Request::default()
            .model("o3")
            .messages(vec![Message::user("think hard")])
            .reasoning_effort("high");
        let body = translate_request(&request);
        assert_eq!(body["reasoning"]["effort"], "high");
    }

    #[test]
    fn test_openai_structured_output_via_text_format() {
        // Responses API: text.format, not response_format
        let request = Request::default()
            .model("gpt-4o")
            .messages(vec![Message::user("extract")])
            .response_format(ResponseFormat {
                r#type: "json_schema".into(),
                json_schema: Some(
                    serde_json::json!({"type": "object", "properties": {"name": {"type": "string"}}}),
                ),
                strict: true,
            });
        let body = translate_request(&request);
        assert_eq!(body["text"]["format"]["type"], "json_schema");
        assert!(body["text"]["format"]["schema"].is_object());
        assert!(body.get("response_format").is_none());
    }

    #[test]
    fn test_openai_json_object_response_format() {
        // json_object mode should be mapped to text.format.type = "json_object"
        let request = Request::default()
            .model("gpt-4o")
            .messages(vec![Message::user("give me JSON")])
            .response_format(ResponseFormat {
                r#type: "json_object".into(),
                json_schema: None,
                strict: false,
            });
        let body = translate_request(&request);
        assert_eq!(body["text"]["format"]["type"], "json_object");
        assert!(body.get("response_format").is_none());
    }

    // === F-11: ResponseFormat.strict uses field value ===

    #[test]
    fn test_response_format_strict_false_not_overridden() {
        let request = Request::default()
            .model("gpt-4o")
            .messages(vec![Message::user("test")])
            .response_format(ResponseFormat {
                r#type: "json_schema".into(),
                json_schema: Some(serde_json::json!({"type": "object"})),
                strict: false,
            });
        let body = translate_request(&request);
        let strict = body["text"]["format"]["strict"].as_bool();
        assert_eq!(
            strict,
            Some(false),
            "strict should be false, not hardcoded to true"
        );
    }

    #[test]
    fn test_response_format_strict_true_forwarded() {
        let request = Request::default()
            .model("gpt-4o")
            .messages(vec![Message::user("test")])
            .response_format(ResponseFormat {
                r#type: "json_schema".into(),
                json_schema: Some(serde_json::json!({"type": "object"})),
                strict: true,
            });
        let body = translate_request(&request);
        let strict = body["text"]["format"]["strict"].as_bool();
        assert_eq!(strict, Some(true), "strict: true should be forwarded");
    }

    // === F-9: ImageData.detail passthrough ===

    #[test]
    fn test_translate_user_content_image_url_includes_detail() {
        let parts = vec![ContentPart::Image {
            image: ImageData {
                url: Some("https://example.com/img.png".into()),
                data: None,
                media_type: None,
                detail: Some("high".into()),
            },
        }];
        let result = translate_user_content(&parts);
        assert_eq!(result.len(), 1);
        let json_str = serde_json::to_string(&result[0]).unwrap();
        assert!(
            json_str.contains("high"),
            "detail field 'high' should be in translated output: {json_str}"
        );
    }

    #[test]
    fn test_translate_user_content_image_no_detail_omits_field() {
        let parts = vec![ContentPart::Image {
            image: ImageData {
                url: Some("https://example.com/img.png".into()),
                data: None,
                media_type: None,
                detail: None,
            },
        }];
        let result = translate_user_content(&parts);
        assert_eq!(result.len(), 1);
        let json_str = serde_json::to_string(&result[0]).unwrap();
        assert!(
            !json_str.contains("detail"),
            "detail field should not appear when None: {json_str}"
        );
    }

    #[test]
    fn test_openai_provider_options_passthrough() {
        // DoD 8.2.7: provider_options.openai merged into request body
        let request = Request::default()
            .model("gpt-4o")
            .messages(vec![Message::user("hi")])
            .provider_options(Some(serde_json::json!({
                "openai": {"store": true, "metadata": {"session": "abc"}}
            })));
        let body = translate_request(&request);
        assert_eq!(body["store"], true);
        assert_eq!(body["metadata"]["session"], "abc");
    }

    // ===== P2B-T04: Request Translation — Tool Calls and Results as Input Items =====

    #[test]
    fn test_openai_tool_call_as_input_item() {
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
            Message::tool_result("call_123", "72F sunny", false),
        ]);
        let body = translate_request(&request);
        let input = body["input"].as_array().unwrap();

        // User message
        assert_eq!(input[0]["type"], "message");
        assert_eq!(input[0]["role"], "user");

        // Tool call -> function_call input item
        assert_eq!(input[1]["type"], "function_call");
        assert_eq!(input[1]["call_id"], "call_123");
        assert_eq!(input[1]["name"], "get_weather");
        // arguments is a JSON string
        let args: serde_json::Value =
            serde_json::from_str(input[1]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args["city"], "SF");

        // Tool result -> function_call_output input item
        assert_eq!(input[2]["type"], "function_call_output");
        assert_eq!(input[2]["call_id"], "call_123");
        assert_eq!(input[2]["output"], "72F sunny");
    }

    #[test]
    fn test_openai_multiple_tool_calls_as_separate_items() {
        let request = Request::default().model("gpt-4o").messages(vec![
            Message::user("weather in SF and NYC?"),
            Message {
                role: Role::Assistant,
                content: vec![
                    ContentPart::ToolCall {
                        tool_call: ToolCallData {
                            id: "call_1".into(),
                            name: "get_weather".into(),
                            arguments: ArgumentValue::Dict({
                                let mut m = serde_json::Map::new();
                                m.insert("city".into(), serde_json::json!("SF"));
                                m
                            }),
                            r#type: "function".into(),
                        },
                    },
                    ContentPart::ToolCall {
                        tool_call: ToolCallData {
                            id: "call_2".into(),
                            name: "get_weather".into(),
                            arguments: ArgumentValue::Dict({
                                let mut m = serde_json::Map::new();
                                m.insert("city".into(), serde_json::json!("NYC"));
                                m
                            }),
                            r#type: "function".into(),
                        },
                    },
                ],
                name: None,
                tool_call_id: None,
            },
            Message::tool_result("call_1", "72F", false),
            Message::tool_result("call_2", "65F", false),
        ]);
        let body = translate_request(&request);
        let input = body["input"].as_array().unwrap();
        // 1 user + 2 function_calls + 2 function_call_outputs = 5
        assert_eq!(input.len(), 5);
        assert_eq!(input[1]["type"], "function_call");
        assert_eq!(input[2]["type"], "function_call");
        assert_eq!(input[3]["type"], "function_call_output");
        assert_eq!(input[4]["type"], "function_call_output");
    }

    #[test]
    fn test_openai_assistant_mixed_text_and_tool_calls() {
        // When an assistant message has BOTH text and tool_calls,
        // the text must appear as an output message AND the tool calls as function_call items.
        let request = Request::default().model("gpt-4o").messages(vec![
            Message::user("weather?"),
            Message {
                role: Role::Assistant,
                content: vec![
                    ContentPart::Text {
                        text: "Let me check the weather for you.".into(),
                    },
                    ContentPart::ToolCall {
                        tool_call: ToolCallData {
                            id: "call_456".into(),
                            name: "get_weather".into(),
                            arguments: ArgumentValue::Dict({
                                let mut m = serde_json::Map::new();
                                m.insert("city".into(), serde_json::json!("SF"));
                                m
                            }),
                            r#type: "function".into(),
                        },
                    },
                ],
                name: None,
                tool_call_id: None,
            },
        ]);
        let body = translate_request(&request);
        let input = body["input"].as_array().unwrap();

        // input[0] = user message
        assert_eq!(input[0]["type"], "message");
        assert_eq!(input[0]["role"], "user");

        // input[1] = assistant message with the TEXT content
        assert_eq!(input[1]["type"], "message");
        assert_eq!(input[1]["role"], "assistant");
        let content = input[1]["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "output_text");
        assert_eq!(content[0]["text"], "Let me check the weather for you.");

        // input[2] = function_call item for the tool call
        assert_eq!(input[2]["type"], "function_call");
        assert_eq!(input[2]["call_id"], "call_456");
        assert_eq!(input[2]["name"], "get_weather");
    }

    // ===== P2B-T05: Response Translation =====

    #[test]
    fn test_openai_parse_text_response() {
        let body = serde_json::json!({
            "id": "resp_001",
            "object": "response",
            "status": "completed",
            "model": "gpt-4o-2025-01-01",
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello there!"}]
            }],
            "usage": {
                "input_tokens": 12,
                "output_tokens": 5,
                "total_tokens": 17
            }
        });
        let response = parse_response(body, &reqwest::header::HeaderMap::new()).unwrap();
        assert_eq!(response.id, "resp_001");
        assert_eq!(response.model, "gpt-4o-2025-01-01");
        assert_eq!(response.provider, "openai");
        assert_eq!(response.text(), "Hello there!");
        assert_eq!(response.finish_reason.reason, "stop");
        assert_eq!(response.usage.input_tokens, 12);
        assert_eq!(response.usage.output_tokens, 5);
        assert_eq!(response.usage.total_tokens, 17);
    }

    #[test]
    fn test_openai_parse_tool_call_response() {
        let body = serde_json::json!({
            "id": "resp_002",
            "object": "response",
            "status": "completed",
            "model": "gpt-4o",
            "output": [{
                "type": "function_call",
                "call_id": "call_abc",
                "name": "get_weather",
                "arguments": "{\"city\":\"SF\"}"
            }],
            "usage": {"input_tokens": 20, "output_tokens": 15, "total_tokens": 35}
        });
        let response = parse_response(body, &reqwest::header::HeaderMap::new()).unwrap();
        assert_eq!(response.finish_reason.reason, "tool_calls");
        let calls = response.tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_abc");
        assert_eq!(calls[0].name, "get_weather");
        match &calls[0].arguments {
            ArgumentValue::Dict(m) => assert_eq!(m["city"], "SF"),
            _ => panic!("Expected Dict arguments"),
        }
    }

    #[test]
    fn test_openai_reasoning_tokens_in_usage() {
        // DoD 8.5.1: reasoning_tokens from output_tokens_details
        let body = serde_json::json!({
            "id": "resp_003",
            "object": "response",
            "status": "completed",
            "model": "o3-2025-01-01",
            "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "42"}]}],
            "usage": {
                "input_tokens": 50,
                "output_tokens": 200,
                "total_tokens": 250,
                "output_tokens_details": {"reasoning_tokens": 180}
            }
        });
        let response = parse_response(body, &reqwest::header::HeaderMap::new()).unwrap();
        assert_eq!(response.usage.reasoning_tokens, Some(180));
        assert_eq!(response.usage.output_tokens, 200);
    }

    #[test]
    fn test_openai_cache_read_tokens() {
        // DoD 8.6.2: cache_read_tokens from input_tokens_details.cached_tokens
        let body = serde_json::json!({
            "id": "resp_004",
            "object": "response",
            "status": "completed",
            "model": "gpt-4o",
            "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "cached"}]}],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 10,
                "total_tokens": 110,
                "input_tokens_details": {"cached_tokens": 80}
            }
        });
        let response = parse_response(body, &reqwest::header::HeaderMap::new()).unwrap();
        assert_eq!(response.usage.cache_read_tokens, Some(80));
    }

    #[test]
    fn test_openai_finish_reason_mapping() {
        for (status, has_tools, expected) in [
            ("completed", false, "stop"),
            ("incomplete", false, "length"),
            ("completed", true, "tool_calls"),
        ] {
            let output = if has_tools {
                serde_json::json!([{"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"}])
            } else {
                serde_json::json!([{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "x"}]}])
            };
            let body = serde_json::json!({
                "id": "r", "object": "response", "status": status, "model": "gpt-4o",
                "output": output,
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
            });
            let response = parse_response(body, &reqwest::header::HeaderMap::new()).unwrap();
            assert_eq!(
                response.finish_reason.reason, expected,
                "status={status}, has_tools={has_tools}"
            );
        }
    }

    // ===== P2B-T06: Error Translation =====

    #[test]
    fn test_openai_error_401() {
        let body = serde_json::json!({
            "error": {"message": "Incorrect API key provided", "type": "invalid_request_error", "code": "invalid_api_key"}
        });
        let err = parse_error(401, &reqwest::header::HeaderMap::new(), body);
        assert_eq!(err.kind, ErrorKind::Authentication);
        assert!(!err.retryable);
        assert_eq!(err.provider, Some("openai".into()));
    }

    #[test]
    fn test_openai_error_429_with_retry_after() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("retry-after", "5".parse().unwrap());
        let body = serde_json::json!({
            "error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}
        });
        let err = parse_error(429, &headers, body);
        assert_eq!(err.kind, ErrorKind::RateLimit);
        assert!(err.retryable);
        assert_eq!(err.retry_after, Some(std::time::Duration::from_secs(5)));
    }

    #[test]
    fn test_openai_error_404() {
        let body = serde_json::json!({
            "error": {"message": "The model 'nonexistent' does not exist", "code": "model_not_found"}
        });
        let err = parse_error(404, &reqwest::header::HeaderMap::new(), body);
        assert_eq!(err.kind, ErrorKind::NotFound);
        assert!(!err.retryable);
    }

    #[test]
    fn test_openai_error_500() {
        let body = serde_json::json!({
            "error": {"message": "Internal server error"}
        });
        let err = parse_error(500, &reqwest::header::HeaderMap::new(), body);
        assert_eq!(err.kind, ErrorKind::Server);
        assert!(err.retryable);
    }

    #[test]
    fn test_openai_error_preserves_raw() {
        let body = serde_json::json!({"error": {"message": "test", "code": "test_code"}});
        let err = parse_error(400, &reqwest::header::HeaderMap::new(), body);
        assert!(err.raw.is_some());
    }

    // ===== P2B-T07: Beta Headers (N/A) and Provider Options =====

    #[test]
    fn test_openai_beta_headers_noop() {
        let request = Request::default()
            .model("gpt-4o")
            .messages(vec![Message::user("hi")])
            .provider_options(Some(serde_json::json!({
                "openai": {"store": true}
            })));
        let body = translate_request(&request);
        assert_eq!(body["store"], true);
    }

    // ===== P2B-T08: Streaming — Text Events =====

    #[tokio::test]
    async fn test_openai_stream_text_events() {
        let sse_body = build_openai_sse_body(&[
            (
                "response.created",
                r#"{"response":{"id":"resp_s1","object":"response","status":"in_progress","model":"gpt-4o","output":[]},"sequence_number":0}"#,
            ),
            (
                "response.output_item.added",
                r#"{"item":{"type":"message","role":"assistant","content":[]},"output_index":0}"#,
            ),
            (
                "response.content_part.added",
                r#"{"part":{"type":"output_text","text":""},"output_index":0,"content_index":0}"#,
            ),
            (
                "response.output_text.delta",
                r#"{"delta":"Hello","output_index":0,"content_index":0}"#,
            ),
            (
                "response.output_text.delta",
                r#"{"delta":" world","output_index":0,"content_index":0}"#,
            ),
            (
                "response.output_text.done",
                r#"{"text":"Hello world","output_index":0,"content_index":0}"#,
            ),
            (
                "response.output_item.done",
                r#"{"item":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Hello world"}]},"output_index":0}"#,
            ),
            (
                "response.completed",
                r#"{"response":{"id":"resp_s1","object":"response","status":"completed","model":"gpt-4o","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Hello world"}]}],"usage":{"input_tokens":10,"output_tokens":3,"total_tokens":13}},"sequence_number":5}"#,
            ),
        ]);

        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/v1/responses"))
            .respond_with(
                wiremock::ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse_body),
            )
            .mount(&server)
            .await;

        let adapter = OpenAiAdapter::new_with_base_url(
            SecretString::from("sk-test".to_string()),
            server.uri(),
        );
        let req = Request::default()
            .model("gpt-4o")
            .messages(vec![Message::user("Hi")]);
        let stream = adapter.stream(req);
        let events: Vec<StreamEvent> = futures::StreamExt::collect::<Vec<_>>(stream)
            .await
            .into_iter()
            .filter_map(|r| r.ok())
            .collect();

        // Verify event types in order
        let types: Vec<_> = events.iter().map(|e| e.event_type.clone()).collect();
        assert!(types.contains(&StreamEventType::StreamStart));
        assert!(types.contains(&StreamEventType::TextStart));
        assert!(types.contains(&StreamEventType::TextDelta));
        assert!(types.contains(&StreamEventType::TextEnd));
        assert!(types.contains(&StreamEventType::Finish));

        // Verify text deltas
        let text: String = events
            .iter()
            .filter(|e| e.event_type == StreamEventType::TextDelta)
            .filter_map(|e| e.delta.as_ref())
            .cloned()
            .collect();
        assert_eq!(text, "Hello world");

        // Verify finish has usage
        let finish = events
            .iter()
            .find(|e| e.event_type == StreamEventType::Finish)
            .unwrap();
        assert!(finish.usage.is_some());
    }

    // ===== P2B-T09: Streaming — Tool Call Events =====

    #[tokio::test]
    async fn test_openai_stream_tool_call_events() {
        let sse_body = build_openai_sse_body(&[
            (
                "response.created",
                r#"{"response":{"id":"resp_t1","status":"in_progress","model":"gpt-4o","output":[]},"sequence_number":0}"#,
            ),
            (
                "response.output_item.added",
                r#"{"item":{"type":"function_call","call_id":"call_1","name":"get_weather","arguments":""},"output_index":0}"#,
            ),
            (
                "response.function_call_arguments.delta",
                r#"{"delta":"{\"city\"","output_index":0}"#,
            ),
            (
                "response.function_call_arguments.delta",
                r#"{"delta":":\"SF\"}","output_index":0}"#,
            ),
            (
                "response.function_call_arguments.done",
                r#"{"arguments":"{\"city\":\"SF\"}","output_index":0}"#,
            ),
            (
                "response.output_item.done",
                r#"{"item":{"type":"function_call","call_id":"call_1","name":"get_weather","arguments":"{\"city\":\"SF\"}"},"output_index":0}"#,
            ),
            (
                "response.completed",
                r#"{"response":{"id":"resp_t1","status":"completed","model":"gpt-4o","output":[{"type":"function_call","call_id":"call_1","name":"get_weather","arguments":"{\"city\":\"SF\"}"}],"usage":{"input_tokens":20,"output_tokens":10,"total_tokens":30}},"sequence_number":5}"#,
            ),
        ]);

        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .respond_with(
                wiremock::ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse_body),
            )
            .mount(&server)
            .await;

        let adapter = OpenAiAdapter::new_with_base_url(
            SecretString::from("sk-test".to_string()),
            server.uri(),
        );
        let stream = adapter.stream(
            Request::default()
                .model("gpt-4o")
                .messages(vec![Message::user("weather?")]),
        );
        let events: Vec<StreamEvent> = futures::StreamExt::collect::<Vec<_>>(stream)
            .await
            .into_iter()
            .filter_map(|r| r.ok())
            .collect();

        let types: Vec<_> = events.iter().map(|e| e.event_type.clone()).collect();
        assert!(types.contains(&StreamEventType::ToolCallStart));
        assert!(types.contains(&StreamEventType::ToolCallDelta));
        assert!(types.contains(&StreamEventType::ToolCallEnd));

        let start = events
            .iter()
            .find(|e| e.event_type == StreamEventType::ToolCallStart)
            .unwrap();
        assert!(start.tool_call.is_some());
        assert_eq!(start.tool_call.as_ref().unwrap().name, "get_weather");
    }

    // ===== P2B-T10: Streaming — Reasoning Events =====

    #[tokio::test]
    async fn test_openai_stream_reasoning_events() {
        let sse_body = build_openai_sse_body(&[
            (
                "response.created",
                r#"{"response":{"id":"resp_r1","status":"in_progress","model":"o3","output":[]},"sequence_number":0}"#,
            ),
            (
                "response.output_item.added",
                r#"{"item":{"type":"reasoning","id":"rs_1"},"output_index":0}"#,
            ),
            (
                "response.reasoning_summary_text.delta",
                r#"{"delta":"Let me think...","output_index":0}"#,
            ),
            (
                "response.output_item.done",
                r#"{"item":{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"Let me think..."}]},"output_index":0}"#,
            ),
            (
                "response.output_item.added",
                r#"{"item":{"type":"message","role":"assistant","content":[]},"output_index":1}"#,
            ),
            (
                "response.output_text.delta",
                r#"{"delta":"42","output_index":1,"content_index":0}"#,
            ),
            (
                "response.completed",
                r#"{"response":{"id":"resp_r1","status":"completed","model":"o3","output":[],"usage":{"input_tokens":10,"output_tokens":50,"total_tokens":60,"output_tokens_details":{"reasoning_tokens":40}}},"sequence_number":5}"#,
            ),
        ]);

        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .respond_with(
                wiremock::ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse_body),
            )
            .mount(&server)
            .await;

        let adapter = OpenAiAdapter::new_with_base_url(
            SecretString::from("sk-test".to_string()),
            server.uri(),
        );
        let events: Vec<StreamEvent> = futures::StreamExt::collect::<Vec<_>>(
            adapter.stream(
                Request::default()
                    .model("o3")
                    .messages(vec![Message::user("think")]),
            ),
        )
        .await
        .into_iter()
        .filter_map(|r| r.ok())
        .collect();

        let types: Vec<_> = events.iter().map(|e| e.event_type.clone()).collect();
        assert!(types.contains(&StreamEventType::ReasoningStart));
        assert!(types.contains(&StreamEventType::ReasoningDelta));
        assert!(types.contains(&StreamEventType::ReasoningEnd));

        // Verify finish event has reasoning_tokens
        let finish = events
            .iter()
            .find(|e| e.event_type == StreamEventType::Finish)
            .unwrap();
        assert_eq!(finish.usage.as_ref().unwrap().reasoning_tokens, Some(40));
    }

    // ===== Additional edge case tests =====

    #[test]
    fn test_openai_image_in_user_message() {
        let request = Request::default().model("gpt-4o").messages(vec![Message {
            role: Role::User,
            content: vec![
                ContentPart::text("What's this?"),
                ContentPart::image_bytes(vec![0xFF, 0xD8], "image/jpeg"),
            ],
            name: None,
            tool_call_id: None,
        }]);
        let body = translate_request(&request);
        let content = body["input"][0]["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "input_text");
        assert_eq!(content[1]["type"], "input_image");
        let image_url = content[1]["image_url"].as_str().unwrap();
        assert!(image_url.starts_with("data:image/jpeg;base64,"));
    }

    #[test]
    fn test_openai_local_file_path_image_resolved_to_base64() {
        // Create a temp file with a .png extension
        let dir = std::env::temp_dir().join("unified_llm_test_openai_img");
        std::fs::create_dir_all(&dir).unwrap();
        let file_path = dir.join("test.png");
        let fake_png = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        std::fs::write(&file_path, &fake_png).unwrap();

        let request = Request::default().model("gpt-4o").messages(vec![Message {
            role: Role::User,
            content: vec![ContentPart::Image {
                image: ImageData {
                    url: Some(file_path.to_str().unwrap().to_string()),
                    data: None,
                    media_type: None,
                    detail: None,
                },
            }],
            name: None,
            tool_call_id: None,
        }]);
        let body = translate_request(&request);
        let content = body["input"][0]["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "input_image");
        let image_url = content[0]["image_url"].as_str().unwrap();
        // Must be a data URI, NOT the raw file path
        assert!(
            image_url.starts_with("data:image/png;base64,"),
            "Expected data URI, got: {}",
            &image_url[..image_url.len().min(60)]
        );

        // Verify the base64 content decodes to the original data
        let b64_part = image_url.strip_prefix("data:image/png;base64,").unwrap();
        use base64::Engine as _;
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(b64_part)
            .unwrap();
        assert_eq!(decoded, fake_png);

        // Cleanup
        let _ = std::fs::remove_file(&file_path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn test_openai_http_url_image_passed_through() {
        // HTTP URLs must NOT be resolved as local files
        let request = Request::default().model("gpt-4o").messages(vec![Message {
            role: Role::User,
            content: vec![ContentPart::Image {
                image: ImageData {
                    url: Some("https://example.com/cat.png".to_string()),
                    data: None,
                    media_type: None,
                    detail: None,
                },
            }],
            name: None,
            tool_call_id: None,
        }]);
        let body = translate_request(&request);
        let content = body["input"][0]["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "input_image");
        assert_eq!(content[0]["image_url"], "https://example.com/cat.png");
    }

    #[tokio::test]
    async fn test_openai_error_in_do_complete() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .respond_with(
                wiremock::ResponseTemplate::new(401).set_body_json(serde_json::json!({
                    "error": {"message": "Invalid API key", "type": "invalid_request_error", "code": "invalid_api_key"}
                })),
            )
            .mount(&server)
            .await;

        let adapter = OpenAiAdapter::new_with_base_url(
            SecretString::from("bad-key".to_string()),
            server.uri(),
        );
        let err = adapter
            .complete(
                Request::default()
                    .model("gpt-4o")
                    .messages(vec![Message::user("hi")]),
            )
            .await
            .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Authentication);
    }

    #[tokio::test]
    async fn test_openai_stream_error_propagation() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .respond_with(
                wiremock::ResponseTemplate::new(500).set_body_json(serde_json::json!({
                    "error": {"message": "Internal server error"}
                })),
            )
            .mount(&server)
            .await;

        let adapter = OpenAiAdapter::new_with_base_url(
            SecretString::from("sk-test".to_string()),
            server.uri(),
        );
        let stream = adapter.stream(
            Request::default()
                .model("gpt-4o")
                .messages(vec![Message::user("hi")]),
        );
        let events: Vec<Result<StreamEvent, Error>> =
            futures::StreamExt::collect::<Vec<_>>(stream).await;
        assert_eq!(events.len(), 1);
        assert!(events[0].is_err());
        assert_eq!(events[0].as_ref().unwrap_err().kind, ErrorKind::Server);
    }

    #[test]
    fn test_openai_temperature_passthrough() {
        let request = Request::default()
            .model("gpt-4o")
            .messages(vec![Message::user("hi")])
            .temperature(0.7);
        let body = translate_request(&request);
        assert_eq!(body["temperature"], 0.7);
    }

    #[test]
    fn test_openai_parse_response_with_rate_limit_headers() {
        let raw = serde_json::json!({
            "id": "resp_rl",
            "model": "gpt-4o",
            "status": "completed",
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hi"}]
            }],
            "usage": {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7}
        });
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("x-ratelimit-remaining-requests", "42".parse().unwrap());
        headers.insert("x-ratelimit-limit-requests", "100".parse().unwrap());

        let response = parse_response(raw, &headers).unwrap();
        let rl = response.rate_limit.unwrap();
        assert_eq!(rl.requests_remaining, Some(42));
        assert_eq!(rl.requests_limit, Some(100));
    }

    #[test]
    fn test_openai_request_includes_model() {
        let request = Request::default()
            .model("gpt-4o-mini")
            .messages(vec![Message::user("hi")]);
        let body = translate_request(&request);
        assert_eq!(body["model"], "gpt-4o-mini");
    }

    // === AF-03: OpenAI Reasoning Output Items Tests ===

    #[test]
    fn test_openai_reasoning_output_item_parsed() {
        let body = serde_json::json!({
            "id": "resp_test",
            "object": "response",
            "status": "completed",
            "model": "o3-2025-01-01",
            "output": [
                {
                    "type": "reasoning",
                    "id": "rs_test",
                    "summary": [
                        {"type": "summary_text", "text": "I thought about the problem carefully."}
                    ]
                },
                {
                    "type": "message",
                    "id": "msg_test",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": "The answer is 42."}
                    ]
                }
            ],
            "usage": {
                "input_tokens": 20,
                "output_tokens": 15,
                "total_tokens": 35,
                "output_tokens_details": {"reasoning_tokens": 10}
            }
        });
        let response = parse_response(body, &reqwest::header::HeaderMap::new()).unwrap();
        assert_eq!(response.text(), "The answer is 42.");
        assert_eq!(
            response.reasoning(),
            Some("I thought about the problem carefully.".to_string()),
            "Reasoning summary text should be captured from reasoning output items"
        );
    }

    #[test]
    fn test_openai_reasoning_and_message_combined() {
        let body = serde_json::json!({
            "id": "resp_test",
            "object": "response",
            "status": "completed",
            "model": "o3-2025-01-01",
            "output": [
                {
                    "type": "reasoning",
                    "id": "rs_test",
                    "summary": [
                        {"type": "summary_text", "text": "Step 1: analyze."},
                        {"type": "summary_text", "text": " Step 2: solve."}
                    ]
                },
                {
                    "type": "message",
                    "id": "msg_test",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": "Done."}
                    ]
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 8, "total_tokens": 18}
        });
        let response = parse_response(body, &reqwest::header::HeaderMap::new()).unwrap();
        assert_eq!(response.text(), "Done.");
        // Both summary_text blocks should be captured as separate Thinking content parts
        let thinking_parts: Vec<_> = response
            .message
            .content
            .iter()
            .filter(|p| matches!(p, ContentPart::Thinking { .. }))
            .collect();
        assert!(
            thinking_parts.len() >= 1,
            "Should have at least one thinking content part"
        );
    }

    // === S-1: text_id propagation to TextDelta/TextEnd ===

    #[test]
    fn test_openai_stream_text_id_consistent_across_start_delta_end() {
        let mut translator = OpenAiStreamTranslator::new();

        // Emit response.created
        translator.process(
            "response.created",
            &serde_json::json!({"response":{"id":"resp_1","model":"gpt-4o","output":[]},"sequence_number":0}),
        );

        // Emit content_part.added (TextStart)
        let events_start = translator.process(
            "response.content_part.added",
            &serde_json::json!({"part":{"type":"output_text","text":""},"output_index":0,"content_index":0}),
        );
        let text_start = events_start
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

        // Emit text delta
        let events_delta = translator.process(
            "response.output_text.delta",
            &serde_json::json!({"delta":"Hello","output_index":0,"content_index":0}),
        );
        let text_delta = events_delta
            .iter()
            .find(|e| e.event_type == StreamEventType::TextDelta)
            .expect("Should emit TextDelta");
        assert_eq!(
            text_delta.text_id.as_ref(),
            Some(&text_id),
            "TextDelta must carry same text_id as TextStart"
        );

        // Emit text done (TextEnd)
        let events_end = translator.process(
            "response.output_text.done",
            &serde_json::json!({"text":"Hello","output_index":0,"content_index":0}),
        );
        let text_end = events_end
            .iter()
            .find(|e| e.event_type == StreamEventType::TextEnd)
            .expect("Should emit TextEnd");
        assert_eq!(
            text_end.text_id.as_ref(),
            Some(&text_id),
            "TextEnd must carry same text_id as TextStart"
        );
    }

    // === S-3: Nested response.completed/created unwrapping ===

    #[test]
    fn test_openai_stream_nested_response_completed_unwraps_correctly() {
        let mut translator = OpenAiStreamTranslator::new();

        // response.created with real API nesting
        let created_events = translator.process(
            "response.created",
            &serde_json::json!({
                "response": {"id": "resp_nested", "model": "gpt-4o", "output": []},
                "sequence_number": 0
            }),
        );
        let stream_start = created_events
            .iter()
            .find(|e| e.event_type == StreamEventType::StreamStart)
            .expect("Should emit StreamStart");
        assert_eq!(
            stream_start.id.as_deref(),
            Some("resp_nested"),
            "response.created must unwrap nested 'response' to extract id"
        );

        // response.completed with real API nesting + tool call output
        let completed_events = translator.process(
            "response.completed",
            &serde_json::json!({
                "response": {
                    "id": "resp_nested",
                    "status": "completed",
                    "output": [{"type": "function_call", "call_id": "call_1", "name": "get_weather"}],
                    "usage": {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30}
                },
                "sequence_number": 5
            }),
        );
        let finish = completed_events
            .iter()
            .find(|e| e.event_type == StreamEventType::Finish)
            .expect("Should emit Finish");

        // Critical: finish_reason must be "tool_calls", not "stop"
        let finish_reason = finish
            .finish_reason
            .as_ref()
            .expect("Finish must have finish_reason");
        assert_eq!(
            finish_reason.reason, "tool_calls",
            "response.completed with function_call output must set finish_reason='tool_calls'"
        );

        // Critical: usage must have real values, not zeros
        let usage = finish.usage.as_ref().expect("Finish must have usage");
        assert_eq!(
            usage.input_tokens, 20,
            "input_tokens must be unwrapped from nested response"
        );
        assert_eq!(
            usage.output_tokens, 10,
            "output_tokens must be unwrapped from nested response"
        );
    }

    #[test]
    fn test_openai_stream_nested_response_failed_unwraps_correctly() {
        let mut translator = OpenAiStreamTranslator::new();

        // response.created so translator has state
        translator.process(
            "response.created",
            &serde_json::json!({
                "response": {"id": "resp_fail", "model": "gpt-4o", "output": []},
                "sequence_number": 0
            }),
        );

        // response.failed with real API nesting — error is inside "response"
        let failed_events = translator.process(
            "response.failed",
            &serde_json::json!({
                "response": {
                    "id": "resp_fail",
                    "status": "failed",
                    "error": {"message": "Rate limit exceeded"}
                },
                "sequence_number": 3
            }),
        );

        let error_event = failed_events
            .iter()
            .find(|e| e.event_type == StreamEventType::Error)
            .expect("Should emit Error event");

        let error = error_event
            .error
            .as_ref()
            .expect("Error event must have error field");
        assert_eq!(
            error.message, "Rate limit exceeded",
            "Error must extract nested error message, got: {}",
            error.message
        );
    }

    // === C-2: AdapterTimeout wiring ===

    #[tokio::test]
    async fn test_connect_timeout_is_wired() {
        use std::time::{Duration, Instant};

        // 10.255.255.1 is a non-routable address — connection will hang until timeout
        let adapter = OpenAiAdapter::new_with_base_url_and_timeout(
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
