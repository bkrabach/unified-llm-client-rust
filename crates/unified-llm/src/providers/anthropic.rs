// Anthropic Messages API adapter.

use futures::StreamExt;
use secrecy::{ExposeSecret, SecretString};

use unified_llm_types::{
    AdapterTimeout, BoxFuture, BoxStream, Error, FinishReason, ProviderAdapter, Request, Response,
    StreamError, StreamEvent, StreamEventType, ToolCall, Usage,
};

use crate::util::sse::SseParser;

/// Default Anthropic API base URL.
const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";

/// Required Anthropic API version header value.
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Default max_tokens when not specified (Anthropic requires this field).
const DEFAULT_MAX_TOKENS: u32 = 4096;

/// Synthetic tool name used for tool-based structured output extraction.
/// When `response_format.type == "json_schema"` and no user-defined tools are present,
/// the adapter defines this synthetic tool whose `input_schema` IS the desired output schema,
/// then forces `tool_choice` to it. The model's tool_use input is the structured output.
const STRUCTURED_OUTPUT_TOOL_NAME: &str = "structured_output";

/// Anthropic Messages API adapter.
pub struct AnthropicAdapter {
    api_key: SecretString,
    base_url: String,
    http_client: reqwest::Client,
    /// Per-chunk timeout for streaming responses (from AdapterTimeout.stream_read).
    stream_read_timeout: std::time::Duration,
}

impl AnthropicAdapter {
    /// Create a new AnthropicAdapter with the given API key.
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

    /// Create a new AnthropicAdapter with a custom base URL (for testing with wiremock).
    ///
    /// Uses default timeouts: connect=10s, request=120s.
    pub fn new_with_base_url(api_key: SecretString, base_url: impl Into<String>) -> Self {
        let timeout = AdapterTimeout::default();
        Self {
            api_key,
            base_url: crate::util::normalize_base_url(&base_url.into()),
            stream_read_timeout: std::time::Duration::from_secs_f64(timeout.stream_read),
            http_client: Self::build_http_client(&timeout),
        }
    }

    /// Create a new AnthropicAdapter with custom timeouts.
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
            base_url: crate::util::normalize_base_url(&base_url.into()),
            stream_read_timeout: std::time::Duration::from_secs_f64(timeout.stream_read),
            http_client: Self::build_http_client(&timeout),
        }
    }

    /// Create from environment variable ANTHROPIC_API_KEY.
    pub fn from_env() -> Result<Self, Error> {
        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .map_err(|_| Error::configuration("ANTHROPIC_API_KEY not set"))?;
        Ok(Self::new(SecretString::from(api_key)))
    }

    /// Create a builder for fine-grained configuration.
    pub fn builder(api_key: SecretString) -> AnthropicAdapterBuilder {
        AnthropicAdapterBuilder::new(api_key)
    }

    /// Build an HTTP client with the given timeout configuration and optional default headers.
    ///
    /// Wires `connect` → `connect_timeout()` and `request` → `timeout()`.
    /// Note: `stream_read` requires a custom per-chunk timeout implementation
    /// (e.g., tokio timeout on individual chunk reads) and is not wired here.
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
        // L-8: Return Result instead of panicking on client build failure
        builder.build().unwrap_or_else(|e| {
            tracing::error!("Failed to build HTTP client: {}", e);
            // Fallback to default client — better than panicking in a library
            reqwest::Client::new()
        })
    }
}

/// Builder for constructing an `AnthropicAdapter` with fine-grained configuration.
pub struct AnthropicAdapterBuilder {
    api_key: SecretString,
    base_url: Option<String>,
    timeout: Option<AdapterTimeout>,
    default_headers: Option<reqwest::header::HeaderMap>,
}

impl AnthropicAdapterBuilder {
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

    /// Build the `AnthropicAdapter`.
    pub fn build(self) -> AnthropicAdapter {
        let timeout = self.timeout.unwrap_or_default();
        let base_url = self
            .base_url
            .map(|u| crate::util::normalize_base_url(&u))
            .unwrap_or_else(|| DEFAULT_BASE_URL.to_string());
        AnthropicAdapter {
            api_key: self.api_key,
            base_url,
            stream_read_timeout: std::time::Duration::from_secs_f64(timeout.stream_read),
            http_client: AnthropicAdapter::build_http_client_with_headers(
                &timeout,
                self.default_headers,
            ),
        }
    }
}

impl AnthropicAdapter {
    /// Build common HTTP headers for Anthropic API requests.
    fn build_headers(&self, beta_headers: &[String]) -> Result<reqwest::header::HeaderMap, Error> {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            "x-api-key",
            self.api_key.expose_secret().parse().map_err(|_| {
                Error::configuration("Invalid API key: contains non-ASCII or control characters")
            })?,
        );
        headers.insert("anthropic-version", ANTHROPIC_VERSION.parse().unwrap());
        headers.insert("content-type", "application/json".parse().unwrap());
        if !beta_headers.is_empty() {
            let beta_value = beta_headers.join(",");
            headers.insert(
                "anthropic-beta",
                beta_value.parse().map_err(|_| {
                    Error::configuration(
                        "Invalid beta header value: contains non-ASCII or control characters",
                    )
                })?,
            );
        }
        Ok(headers)
    }

    /// Perform the actual HTTP request for complete().
    async fn do_complete(&self, mut request: Request) -> Result<Response, Error> {
        // H-4: Pre-resolve local file images to avoid blocking I/O in translate_request.
        crate::util::image::pre_resolve_local_images(&mut request.messages).await?;

        let url = format!("{}/v1/messages", self.base_url);
        let beta_headers = collect_beta_headers(&request);
        let (body, translation_warnings) = translate_request_with_cache(&request);

        let request_headers = self.build_headers(&beta_headers)?;

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

        // GAP-2: Post-process tool-based structured output extraction.
        // If we injected a synthetic tool for json_schema and the response contains
        // our synthetic tool call, convert the tool arguments to text content so that
        // response.text() returns the JSON for generate_object() to parse.
        let used_tool_extraction = request
            .response_format
            .as_ref()
            .map(|rf| rf.r#type == "json_schema" && rf.json_schema.is_some())
            .unwrap_or(false)
            && request.tools.is_none()
            && request.tool_choice.is_none();
        if used_tool_extraction {
            postprocess_structured_output_tool(&mut response);
        }

        Ok(response)
    }

    /// Perform the HTTP request for stream() and return a stream of events.
    fn do_stream(&self, mut request: Request) -> BoxStream<'_, Result<StreamEvent, Error>> {
        let stream = async_stream::stream! {
            // H-4: Pre-resolve local file images to avoid blocking I/O in translate_request.
            if let Err(e) = crate::util::image::pre_resolve_local_images(&mut request.messages).await {
                yield Err(e);
                return;
            }

            let url = format!("{}/v1/messages", self.base_url);
            let beta_headers = collect_beta_headers(&request);
            let (mut body, translation_warnings) = translate_request_with_cache(&request);
            // L-4: Log warnings that can't be attached to streaming responses
            for w in &translation_warnings {
                tracing::warn!("Translation warning (streaming): {}", w.message);
            }
            // Add stream: true to the request body
            if let Some(obj) = body.as_object_mut() {
                obj.insert("stream".into(), serde_json::Value::Bool(true));
            }

            let request_headers = match self.build_headers(&beta_headers) {
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

            let mut translator = StreamTranslator::new();
            let stream_read_timeout = self.stream_read_timeout;
            // L-6: Buffer for incomplete UTF-8 sequences across chunk boundaries
            let mut utf8_remainder: Vec<u8> = Vec::new();

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

                // L-6: Buffer partial UTF-8 sequences across chunks instead of dropping them.
                // Prepend any leftover bytes from the previous chunk.
                let full_chunk = if utf8_remainder.is_empty() {
                    chunk.to_vec()
                } else {
                    let mut combined = std::mem::take(&mut utf8_remainder);
                    combined.extend_from_slice(&chunk);
                    combined
                };
                let chunk_str = match std::str::from_utf8(&full_chunk) {
                    Ok(s) => s.to_string(),
                    Err(e) => {
                        // Valid prefix up to the error point
                        let valid_up_to = e.valid_up_to();
                        if valid_up_to == 0 && full_chunk.len() < 4 {
                            // Entire chunk is an incomplete multi-byte sequence — buffer it
                            utf8_remainder = full_chunk;
                            continue;
                        }
                        // Save the incomplete tail for the next chunk
                        utf8_remainder = full_chunk[valid_up_to..].to_vec();
                        // Process the valid prefix
                        std::str::from_utf8(&full_chunk[..valid_up_to])
                            .unwrap_or("")
                            .to_string()
                    }
                };

                let sse_events = parser.feed(&chunk_str);

                for sse_event in sse_events {
                    let event_type = match &sse_event.event_type {
                        Some(t) => t.as_str(),
                        None => continue,
                    };

                    // Silently ignore ping events
                    if event_type == "ping" {
                        continue;
                    }

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

/// Holds streaming state and translates Anthropic SSE events into unified StreamEvents.
struct StreamTranslator {
    message_id: String,
    model: String,
    input_tokens: u32,
    output_tokens: u32,
    stop_reason: Option<String>,
    active_block_type: Option<String>,
    active_block_index: Option<u64>,
    active_tool_id: Option<String>,
    active_tool_name: Option<String>,
    accumulated_tool_json: String,
    cache_read_tokens: Option<u32>,
    cache_write_tokens: Option<u32>,
    thinking_text_length: usize,
    active_thinking_signature: Option<String>,
}

impl StreamTranslator {
    fn new() -> Self {
        Self {
            message_id: String::new(),
            model: String::new(),
            input_tokens: 0,
            output_tokens: 0,
            stop_reason: None,
            active_block_type: None,
            active_block_index: None,
            active_tool_id: None,
            active_tool_name: None,
            accumulated_tool_json: String::new(),
            cache_read_tokens: None,
            cache_write_tokens: None,
            thinking_text_length: 0,
            active_thinking_signature: None,
        }
    }

    /// Translate a single Anthropic SSE event into zero or more unified StreamEvents.
    fn process(&mut self, event_type: &str, data: &serde_json::Value) -> Vec<StreamEvent> {
        let mut events = Vec::new();

        match event_type {
            "message_start" => {
                if let Some(msg) = data.get("message") {
                    self.message_id = msg
                        .get("id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    self.model = msg
                        .get("model")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    if let Some(usage) = msg.get("usage") {
                        self.input_tokens = usage
                            .get("input_tokens")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0) as u32;
                        self.cache_read_tokens = usage
                            .get("cache_read_input_tokens")
                            .and_then(|v| v.as_u64())
                            .map(|v| v as u32);
                        self.cache_write_tokens = usage
                            .get("cache_creation_input_tokens")
                            .and_then(|v| v.as_u64())
                            .map(|v| v as u32);
                    }
                }
                events.push(StreamEvent {
                    event_type: StreamEventType::StreamStart,
                    id: Some(self.message_id.clone()),
                    ..Default::default()
                });
            }
            "content_block_start" => {
                // Extract block index from the event data (Anthropic sends "index": N)
                let block_idx = data.get("index").and_then(|v| v.as_u64());
                self.active_block_index = block_idx;

                if let Some(block) = data.get("content_block") {
                    let block_type = block.get("type").and_then(|v| v.as_str()).unwrap_or("text");
                    self.active_block_type = Some(block_type.to_string());
                    match block_type {
                        "text" => {
                            events.push(StreamEvent {
                                event_type: StreamEventType::TextStart,
                                text_id: block_idx.map(|i| i.to_string()),
                                ..Default::default()
                            });
                        }
                        "tool_use" => {
                            let id = block
                                .get("id")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string();
                            let name = block
                                .get("name")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string();
                            // Track tool state for ToolCallEnd
                            self.active_tool_id = Some(id.clone());
                            self.active_tool_name = Some(name.clone());
                            self.accumulated_tool_json.clear();
                            events.push(StreamEvent {
                                event_type: StreamEventType::ToolCallStart,
                                tool_call: Some(ToolCall {
                                    id: id.clone(),
                                    name: name.clone(),
                                    arguments: serde_json::Map::new(),
                                    raw_arguments: None,
                                }),
                                ..Default::default()
                            });
                        }
                        "thinking" => {
                            self.active_thinking_signature = None; // reset for new block
                            events.push(StreamEvent {
                                event_type: StreamEventType::ReasoningStart,
                                ..Default::default()
                            });
                        }
                        "redacted_thinking" => {
                            // Emit ReasoningStart + immediate ReasoningEnd with data payload
                            let payload_data = block
                                .get("data")
                                .and_then(|v| v.as_str())
                                .map(|s| s.to_string());
                            events.push(StreamEvent {
                                event_type: StreamEventType::ReasoningStart,
                                ..Default::default()
                            });
                            events.push(StreamEvent {
                                event_type: StreamEventType::ReasoningEnd,
                                raw: Some(serde_json::json!({
                                    "redacted": true,
                                    "data": payload_data,
                                })),
                                ..Default::default()
                            });
                        }
                        _ => {}
                    }
                }
            }
            "content_block_delta" => {
                if let Some(delta) = data.get("delta") {
                    let delta_type = delta.get("type").and_then(|v| v.as_str()).unwrap_or("");
                    match delta_type {
                        "text_delta" => {
                            let text = delta
                                .get("text")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string();
                            events.push(StreamEvent {
                                event_type: StreamEventType::TextDelta,
                                delta: Some(text),
                                text_id: self.active_block_index.map(|i| i.to_string()),
                                ..Default::default()
                            });
                        }
                        "input_json_delta" => {
                            let partial_json = delta
                                .get("partial_json")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string();
                            // Accumulate JSON fragments for ToolCallEnd
                            self.accumulated_tool_json.push_str(&partial_json);
                            events.push(StreamEvent {
                                event_type: StreamEventType::ToolCallDelta,
                                delta: Some(partial_json),
                                ..Default::default()
                            });
                        }
                        "thinking_delta" => {
                            let thinking = delta
                                .get("thinking")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string();
                            self.thinking_text_length += thinking.len();
                            events.push(StreamEvent {
                                event_type: StreamEventType::ReasoningDelta,
                                delta: Some(thinking.clone()),
                                reasoning_delta: Some(thinking),
                                ..Default::default()
                            });
                        }
                        "signature_delta" => {
                            let sig_fragment = delta
                                .get("signature")
                                .and_then(|v| v.as_str())
                                .unwrap_or("");
                            match self.active_thinking_signature {
                                Some(ref mut existing) => existing.push_str(sig_fragment),
                                None => {
                                    self.active_thinking_signature = Some(sig_fragment.to_string())
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            "content_block_stop" => {
                // Emit the appropriate End event based on active block type
                match self.active_block_type.as_deref() {
                    Some("text") => {
                        events.push(StreamEvent {
                            event_type: StreamEventType::TextEnd,
                            text_id: self.active_block_index.map(|i| i.to_string()),
                            ..Default::default()
                        });
                    }
                    Some("tool_use") => {
                        // Build complete tool_call from accumulated state
                        let tool_call = if self.active_tool_id.is_some()
                            || self.active_tool_name.is_some()
                        {
                            let id = self.active_tool_id.take().unwrap_or_default();
                            let name = self.active_tool_name.take().unwrap_or_default();
                            let arguments: serde_json::Map<String, serde_json::Value> =
                                match serde_json::from_str(self.accumulated_tool_json.as_str()) {
                                    Ok(v) => v,
                                    Err(e) => {
                                        // L-7: Warn on tool JSON deserialization failure
                                        tracing::warn!(
                                                "Failed to parse tool call arguments as JSON: {}. Raw: '{}'",
                                                e,
                                                &self.accumulated_tool_json[..self.accumulated_tool_json.len().min(200)]
                                            );
                                        serde_json::Map::new()
                                    }
                                };
                            let raw_arguments = if self.accumulated_tool_json.is_empty() {
                                None
                            } else {
                                Some(self.accumulated_tool_json.clone())
                            };
                            Some(ToolCall {
                                id,
                                name,
                                arguments,
                                raw_arguments,
                            })
                        } else {
                            None
                        };
                        events.push(StreamEvent {
                            event_type: StreamEventType::ToolCallEnd,
                            tool_call,
                            ..Default::default()
                        });
                        self.accumulated_tool_json.clear();
                    }
                    Some("thinking") => {
                        // Signature is accumulated from signature_delta events
                        // during content_block_delta processing (not from content_block_stop).
                        events.push(StreamEvent {
                            event_type: StreamEventType::ReasoningEnd,
                            raw: self
                                .active_thinking_signature
                                .take()
                                .map(|sig| serde_json::json!({"signature": sig})),
                            ..Default::default()
                        });
                    }
                    _ => {}
                }
                self.active_block_type = None;
                self.active_block_index = None;
            }
            "message_delta" => {
                if let Some(delta) = data.get("delta") {
                    self.stop_reason = delta
                        .get("stop_reason")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                }
                if let Some(usage) = data.get("usage") {
                    self.output_tokens = usage
                        .get("output_tokens")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0) as u32;
                }
            }
            "message_stop" => {
                let finish_reason = self
                    .stop_reason
                    .as_deref()
                    .map(map_finish_reason)
                    .unwrap_or_else(FinishReason::stop);
                // M-2: Anthropic does not provide a separate reasoning token count.
                // This is an APPROXIMATION based on thinking text character count / 4.
                // Actual token counts may differ. Use for directional cost estimates only.
                let reasoning_tokens = if self.thinking_text_length > 0 {
                    Some((self.thinking_text_length / 4) as u32)
                } else {
                    None
                };
                let usage = Usage {
                    input_tokens: self.input_tokens,
                    output_tokens: self.output_tokens,
                    total_tokens: self.input_tokens + self.output_tokens,
                    reasoning_tokens,
                    cache_read_tokens: self.cache_read_tokens,
                    cache_write_tokens: self.cache_write_tokens,
                    raw: None,
                };
                events.push(StreamEvent {
                    event_type: StreamEventType::Finish,
                    finish_reason: Some(finish_reason),
                    usage: Some(usage),
                    ..Default::default()
                });
            }
            "error" => {
                // CQ-15: Explicitly handle Anthropic SSE "error" events
                let error_msg = data
                    .get("error")
                    .and_then(|e| e.get("message"))
                    .and_then(|m| m.as_str())
                    .unwrap_or("Unknown Anthropic streaming error");
                let error_type = data
                    .get("error")
                    .and_then(|e| e.get("type"))
                    .and_then(|t| t.as_str())
                    .unwrap_or("unknown");
                tracing::error!(
                    "Anthropic SSE error event: type={}, message={}",
                    error_type,
                    error_msg
                );
                // L-5: Map Anthropic SSE error.type to proper HTTP status
                let status_code = match error_type {
                    "overloaded_error" => 529,
                    "rate_limit_error" => 429,
                    "authentication_error" => 401,
                    "permission_error" => 403,
                    "not_found_error" => 404,
                    "invalid_request_error" => 400,
                    _ => 500, // api_error and unknown types
                };
                let provider_error = Error::from_http_status(
                    status_code,
                    format!("Anthropic streaming error: {}", error_msg),
                    "anthropic",
                    Some(data.clone()),
                    None,
                );
                events.push(StreamEvent {
                    event_type: StreamEventType::Error,
                    error: Some(Box::new(StreamError::from_error(&provider_error))),
                    ..Default::default()
                });
            }
            _ => {
                // Forward unknown SSE events as PROVIDER_EVENT (spec §3.13, M-4)
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

/// Collect beta headers needed for this request.
fn collect_beta_headers(request: &Request) -> Vec<String> {
    let mut betas = Vec::new();

    // User-specified betas from provider_options.anthropic.beta_headers (spec name)
    // or provider_options.anthropic.betas (legacy name). Check beta_headers first.
    if let Some(ref opts) = request.provider_options {
        if let Some(anthropic_opts) = opts.get("anthropic") {
            let user_betas = anthropic_opts
                .get("beta_headers")
                .or_else(|| anthropic_opts.get("betas"));
            if let Some(user_betas) = user_betas {
                if let Some(arr) = user_betas.as_array() {
                    for b in arr {
                        if let Some(s) = b.as_str() {
                            betas.push(s.to_string());
                        }
                    }
                }
            }
        }
    }

    // Auto-add prompt-caching beta if cache_control will be injected
    if should_auto_cache(request.provider_options.as_ref()) {
        let cache_beta = "prompt-caching-2024-07-31".to_string();
        if !betas.contains(&cache_beta) {
            betas.push(cache_beta);
        }
    }

    // Deduplicate
    betas.sort();
    betas.dedup();
    betas
}

/// Check if auto-caching should be enabled (default true unless explicitly disabled).
fn should_auto_cache(provider_options: Option<&serde_json::Value>) -> bool {
    if let Some(opts) = provider_options {
        if let Some(anthropic_opts) = opts.get("anthropic") {
            if let Some(auto_cache) = anthropic_opts.get("auto_cache") {
                return auto_cache.as_bool().unwrap_or(true);
            }
        }
    }
    // Default: auto-cache is enabled
    true
}

impl ProviderAdapter for AnthropicAdapter {
    fn name(&self) -> &str {
        "anthropic"
    }

    fn complete(&self, request: Request) -> BoxFuture<'_, Result<Response, Error>> {
        Box::pin(self.do_complete(request))
    }

    fn stream(&self, request: Request) -> BoxStream<'_, Result<StreamEvent, Error>> {
        self.do_stream(request)
    }
}

// === Request Translation ===

/// Translate a unified Request into an Anthropic Messages API JSON body.
pub(crate) fn translate_request(
    request: &Request,
) -> (serde_json::Value, Vec<unified_llm_types::Warning>) {
    let mut body = serde_json::Map::new();
    let mut warnings: Vec<unified_llm_types::Warning> = Vec::new();

    // Model
    body.insert(
        "model".into(),
        serde_json::Value::String(request.model.clone()),
    );

    // max_tokens — Anthropic REQUIRES this field, default to 4096
    let max_tokens = request.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
    body.insert(
        "max_tokens".into(),
        serde_json::Value::Number(max_tokens.into()),
    );

    // All system messages are extracted regardless of conversation position per provider API
    // conventions. Mid-conversation system messages are repositioned to the system prompt area.
    let mut system_parts: Vec<String> = Vec::new();
    let mut api_messages: Vec<serde_json::Value> = Vec::new();

    for msg in &request.messages {
        match msg.role {
            unified_llm_types::Role::System | unified_llm_types::Role::Developer => {
                system_parts.push(msg.text());
            }
            unified_llm_types::Role::Tool => {
                // Tool results become user messages with tool_result content blocks
                let mut content_blocks = Vec::new();
                for part in &msg.content {
                    if let unified_llm_types::ContentPart::ToolResult { tool_result } = part {
                        let mut block = serde_json::json!({
                            "type": "tool_result",
                            "tool_use_id": tool_result.tool_call_id,
                        });
                        // Content can be string or structured
                        match &tool_result.content {
                            serde_json::Value::String(s) => {
                                block["content"] = serde_json::Value::String(s.clone());
                            }
                            other => {
                                block["content"] = other.clone();
                            }
                        }
                        if tool_result.is_error {
                            block["is_error"] = serde_json::Value::Bool(true);
                        }
                        content_blocks.push(block);
                    }
                }
                api_messages.push(serde_json::json!({
                    "role": "user",
                    "content": content_blocks,
                }));
            }
            unified_llm_types::Role::User | unified_llm_types::Role::Assistant => {
                let role_str = match msg.role {
                    unified_llm_types::Role::User => "user",
                    unified_llm_types::Role::Assistant => "assistant",
                    other => {
                        // CQ-1: Graceful handling instead of unreachable!()
                        tracing::warn!(
                            "Unexpected role {:?} in Anthropic message translation, skipping",
                            other
                        );
                        continue;
                    }
                };
                let (content_blocks, mut part_warnings) = translate_content_parts(&msg.content);
                warnings.append(&mut part_warnings);
                api_messages.push(serde_json::json!({
                    "role": role_str,
                    "content": content_blocks,
                }));
            }
        }
    }

    // Merge consecutive same-role messages (strict alternation requirement)
    let merged_messages = merge_consecutive_messages(api_messages);
    body.insert("messages".into(), serde_json::Value::Array(merged_messages));

    // System parameter — build the system text, then inject structured output instructions if needed
    let mut system_text = system_parts.join("\n\n");

    // Determine if we can use tool-based extraction for structured output (GAP-2).
    // Primary strategy: define a synthetic tool whose input_schema IS the desired output schema,
    // force tool_choice to it, and extract the tool_use input as structured output.
    // Fall back to system-prompt injection if tools or tool_choice are already set (to avoid conflicts).
    let use_tool_based_extraction = request
        .response_format
        .as_ref()
        .map(|rf| rf.r#type == "json_schema" && rf.json_schema.is_some())
        .unwrap_or(false)
        && request.tools.is_none()
        && request.tool_choice.is_none();

    // Structured output handling for Anthropic:
    // - json_schema with tool-based extraction: handled below after tools section
    // - json_schema without tool-based extraction (fallback): system prompt injection
    // - json_object: always system prompt injection
    if let Some(ref response_format) = request.response_format {
        if response_format.r#type == "json_schema" && !use_tool_based_extraction {
            // Fallback: inject schema into system prompt when tools/tool_choice conflict
            if let Some(ref schema) = response_format.json_schema {
                let schema_instruction = format!(
                    "\n\nYou MUST respond with valid JSON that conforms to this JSON Schema:\n\
                    ```json\n{}\n```\n\
                    Do not include any text outside the JSON object. \
                    Do not wrap the response in markdown code fences.",
                    serde_json::to_string_pretty(schema).unwrap_or_default()
                );
                system_text.push_str(&schema_instruction);
            }
        } else if response_format.r#type == "json_object" {
            system_text.push_str(
                "\n\nYou MUST respond with valid JSON. \
                Do not include any text outside the JSON object. \
                Do not wrap the response in markdown code fences.",
            );
        }
    }

    if !system_text.is_empty() {
        body.insert("system".into(), serde_json::Value::String(system_text));
    }

    // Optional parameters
    if let Some(temp) = request.temperature {
        body.insert("temperature".into(), serde_json::json!(temp));
    }
    if let Some(top_p) = request.top_p {
        body.insert("top_p".into(), serde_json::json!(top_p));
    }
    if let Some(ref stop) = request.stop_sequences {
        body.insert("stop_sequences".into(), serde_json::json!(stop));
    }

    // Tools
    if let Some(ref tools) = request.tools {
        let tool_defs: Vec<serde_json::Value> = tools
            .iter()
            .map(|t| {
                serde_json::json!({
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                })
            })
            .collect();
        body.insert("tools".into(), serde_json::Value::Array(tool_defs));
    }

    // Tool choice (spec §5.3: translate unified tool_choice to Anthropic format)
    if let Some(ref tc) = request.tool_choice {
        match tc.mode.as_str() {
            "none" => {
                // Anthropic doesn't support tool_choice=none with tools present;
                // omit the tools array entirely to prevent tool use.
                body.remove("tools");
            }
            "auto" => {
                body.insert("tool_choice".into(), serde_json::json!({"type": "auto"}));
            }
            "required" => {
                body.insert("tool_choice".into(), serde_json::json!({"type": "any"}));
            }
            "named" => {
                if let Some(ref name) = tc.tool_name {
                    body.insert(
                        "tool_choice".into(),
                        serde_json::json!({"type": "tool", "name": name}),
                    );
                }
            }
            _ => {}
        }
    }

    // GAP-2: Tool-based structured output extraction (primary strategy for json_schema).
    // Injects a synthetic tool with the user's schema as input_schema, then forces
    // tool_choice to it. The response post-processing in do_complete() converts the
    // tool_use input back into text content.
    if use_tool_based_extraction {
        if let Some(ref response_format) = request.response_format {
            if let Some(ref schema) = response_format.json_schema {
                let tool_def = serde_json::json!({
                    "name": STRUCTURED_OUTPUT_TOOL_NAME,
                    "description": "Generate structured output matching the specified JSON schema. \
                        All output MUST be provided as the input to this tool.",
                    "input_schema": schema,
                });
                body.insert("tools".into(), serde_json::json!([tool_def]));
                body.insert(
                    "tool_choice".into(),
                    serde_json::json!({"type": "tool", "name": STRUCTURED_OUTPUT_TOOL_NAME}),
                );
            }
        }
    }

    // Map reasoning_effort to Anthropic's thinking config.
    // Only inject if provider_options.anthropic.thinking is NOT already set.
    if let Some(ref effort) = request.reasoning_effort {
        let has_explicit_thinking = request
            .provider_options
            .as_ref()
            .and_then(|opts| opts.get("anthropic"))
            .and_then(|a| a.get("thinking"))
            .is_some();

        if !has_explicit_thinking && effort.as_str() != "none" {
            let budget = match effort.as_str() {
                "low" => 1024,
                "medium" => 4096,
                "high" => 10000,
                _ => 4096, // default to medium for unknown values
            };
            // C-1: Anthropic requires budget_tokens < max_tokens.
            // Auto-adjust max_tokens upward if budget would equal or exceed it.
            let current_max = body
                .get("max_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(DEFAULT_MAX_TOKENS as u64);
            if budget as u64 >= current_max {
                let new_max = budget as u64 + 1024; // leave room for visible output
                body.insert("max_tokens".into(), serde_json::json!(new_max));
            }
            body.insert(
                "thinking".into(),
                serde_json::json!({
                    "type": "enabled",
                    "budget_tokens": budget
                }),
            );
        }
    }

    // Provider options passthrough using shared utility.
    const INTERNAL_KEYS: &[&str] = &["betas", "beta_headers", "auto_cache"];
    if let Some(opts) =
        crate::util::provider_options::get_provider_options(&request.provider_options, "anthropic")
    {
        let mut body_val = serde_json::Value::Object(body);
        crate::util::provider_options::merge_provider_options(&mut body_val, &opts, INTERNAL_KEYS);
        return (body_val, warnings);
    }

    (serde_json::Value::Object(body), warnings)
}

/// Translate a request with automatic cache_control injection.
///
/// This is the main entry point for request translation. It calls `translate_request`
/// and then injects cache_control breakpoints if auto-caching is enabled:
/// 1. Last block of system prompt
/// 2. Last tool definition (if tools present)
/// 3. Last message in conversation prefix (messages before final user message)
pub(crate) fn translate_request_with_cache(
    request: &Request,
) -> (serde_json::Value, Vec<unified_llm_types::Warning>) {
    let (mut body, warnings) = translate_request(request);

    if !should_auto_cache(request.provider_options.as_ref()) {
        return (body, warnings);
    }

    let body_obj = match body.as_object_mut() {
        Some(obj) => obj,
        None => return (body, warnings),
    };

    // 1. Inject cache_control on system prompt (convert to array format if needed)
    if let Some(system) = body_obj.remove("system") {
        match system {
            serde_json::Value::String(s) => {
                // Convert plain string to array of content blocks with cache_control on last
                let block = serde_json::json!({
                    "type": "text",
                    "text": s,
                    "cache_control": {"type": "ephemeral"}
                });
                body_obj.insert("system".into(), serde_json::json!([block]));
            }
            serde_json::Value::Array(mut arr) => {
                // Already an array, add cache_control to last block
                if let Some(last) = arr.last_mut() {
                    if let Some(obj) = last.as_object_mut() {
                        obj.insert(
                            "cache_control".into(),
                            serde_json::json!({"type": "ephemeral"}),
                        );
                    }
                }
                body_obj.insert("system".into(), serde_json::Value::Array(arr));
            }
            other => {
                body_obj.insert("system".into(), other);
            }
        }
    }

    // 2. Inject cache_control on last tool definition
    if let Some(tools) = body_obj.get_mut("tools") {
        if let Some(tools_arr) = tools.as_array_mut() {
            if let Some(last_tool) = tools_arr.last_mut() {
                if let Some(obj) = last_tool.as_object_mut() {
                    obj.insert(
                        "cache_control".into(),
                        serde_json::json!({"type": "ephemeral"}),
                    );
                }
            }
        }
    }

    // 3. Inject cache_control on last message in conversation prefix
    // (the message just before the final user message)
    if let Some(messages) = body_obj.get_mut("messages") {
        if let Some(msgs_arr) = messages.as_array_mut() {
            if msgs_arr.len() >= 2 {
                // Find the index of the last user message
                let last_user_idx = msgs_arr
                    .iter()
                    .rposition(|m| m.get("role").and_then(|r| r.as_str()) == Some("user"));

                if let Some(last_user_idx) = last_user_idx {
                    if last_user_idx > 0 {
                        // The prefix is everything before the last user message
                        let prefix_idx = last_user_idx - 1;
                        if let Some(prefix_msg) = msgs_arr.get_mut(prefix_idx) {
                            if let Some(content) = prefix_msg.get_mut("content") {
                                if let Some(content_arr) = content.as_array_mut() {
                                    if let Some(last_block) = content_arr.last_mut() {
                                        if let Some(obj) = last_block.as_object_mut() {
                                            obj.insert(
                                                "cache_control".into(),
                                                serde_json::json!({"type": "ephemeral"}),
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    (body, warnings)
}

/// Translate unified ContentParts into Anthropic content blocks.
/// Returns the translated blocks and any warnings for dropped/unsupported parts.
fn translate_content_parts(
    parts: &[unified_llm_types::ContentPart],
) -> (Vec<serde_json::Value>, Vec<unified_llm_types::Warning>) {
    let mut warnings = Vec::new();
    let blocks = parts
        .iter()
        .filter_map(|part| match part {
            unified_llm_types::ContentPart::Text { text } => {
                Some(serde_json::json!({"type": "text", "text": text}))
            }
            unified_llm_types::ContentPart::Image { image } => {
                if let Some(ref data) = image.data {
                    let b64 = crate::util::image::base64_encode(data);
                    let media_type = image.media_type.as_deref().unwrap_or("image/png");
                    Some(serde_json::json!({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
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
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": mime,
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
                                    serde_json::json!({
                                        "type": "image",
                                        "source": {
                                            "type": "url",
                                            "url": url,
                                        }
                                    })
                                }
                            }
                        } else {
                            serde_json::json!({
                                "type": "image",
                                "source": {
                                    "type": "url",
                                    "url": url,
                                }
                            })
                        }
                    })
                }
            }
            unified_llm_types::ContentPart::ToolCall { tool_call } => {
                let input = match &tool_call.arguments {
                    unified_llm_types::ArgumentValue::Dict(map) => {
                        serde_json::Value::Object(map.clone())
                    }
                    unified_llm_types::ArgumentValue::Raw(s) => {
                        // L-7: Warn on tool JSON deserialization failure in outbound path
                        match serde_json::from_str(s) {
                            Ok(v) => v,
                            Err(e) => {
                                tracing::warn!(
                                    "Failed to parse outbound tool arguments as JSON: {}",
                                    e
                                );
                                serde_json::json!({})
                            }
                        }
                    }
                };
                Some(serde_json::json!({
                    "type": "tool_use",
                    "id": tool_call.id,
                    "name": tool_call.name,
                    "input": input,
                }))
            }
            unified_llm_types::ContentPart::ToolResult { tool_result } => {
                let mut block = serde_json::json!({
                    "type": "tool_result",
                    "tool_use_id": tool_result.tool_call_id,
                });
                match &tool_result.content {
                    serde_json::Value::String(s) => {
                        block["content"] = serde_json::Value::String(s.clone());
                    }
                    other => {
                        block["content"] = other.clone();
                    }
                }
                if tool_result.is_error {
                    block["is_error"] = serde_json::Value::Bool(true);
                }
                Some(block)
            }
            unified_llm_types::ContentPart::Thinking { thinking } => {
                let mut block = serde_json::json!({
                    "type": "thinking",
                    "thinking": thinking.text,
                });
                if let Some(ref sig) = thinking.signature {
                    block["signature"] = serde_json::Value::String(sig.clone());
                }
                Some(block)
            }
            unified_llm_types::ContentPart::RedactedThinking { thinking } => {
                thinking.data.as_ref().map(|data| {
                    serde_json::json!({
                        "type": "redacted_thinking",
                        "data": data,
                    })
                })
            }
            other => {
                let msg = format!(
                    "Dropped unsupported content part kind={:?} for provider=anthropic",
                    other.kind()
                );
                tracing::warn!("{}", msg);
                warnings.push(unified_llm_types::Warning {
                    message: msg,
                    code: Some("dropped_content_part".to_string()),
                });
                None
            }
        })
        .collect();
    (blocks, warnings)
}

/// Merge consecutive messages with the same role (Anthropic strict alternation).
fn merge_consecutive_messages(messages: Vec<serde_json::Value>) -> Vec<serde_json::Value> {
    let mut merged: Vec<serde_json::Value> = Vec::new();

    for msg in messages {
        let should_merge = if let Some(last) = merged.last() {
            last.get("role") == msg.get("role")
        } else {
            false
        };

        if should_merge {
            // Merge content arrays
            let last = merged.last_mut().unwrap();
            if let (Some(existing), Some(new)) = (
                last.get_mut("content").and_then(|c| c.as_array_mut()),
                msg.get("content").and_then(|c| c.as_array()),
            ) {
                existing.extend(new.iter().cloned());
            }
        } else {
            merged.push(msg);
        }
    }

    merged
}

// === Response Translation ===

/// Map Anthropic stop_reason to unified FinishReason.
pub(crate) fn map_finish_reason(reason: &str) -> unified_llm_types::FinishReason {
    let unified = match reason {
        "end_turn" | "stop_sequence" => "stop",
        "max_tokens" => "length",
        "tool_use" => "tool_calls",
        _ => "other",
    };
    unified_llm_types::FinishReason {
        reason: unified.to_string(),
        raw: Some(reason.to_string()),
    }
}

/// Parse an Anthropic Messages API response JSON into a unified Response.
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

    // Parse content blocks
    let content_parts = parse_content_blocks(
        raw.get("content")
            .and_then(|v| v.as_array())
            .map(|a| a.as_slice())
            .unwrap_or(&[]),
    );

    // Finish reason
    let finish_reason = raw
        .get("stop_reason")
        .and_then(|v| v.as_str())
        .map(map_finish_reason)
        .unwrap_or(unified_llm_types::FinishReason::stop());

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
    let cache_read_tokens = usage_obj
        .and_then(|u| u.get("cache_read_input_tokens"))
        .and_then(|v| v.as_u64())
        .map(|v| v as u32);
    let cache_write_tokens = usage_obj
        .and_then(|u| u.get("cache_creation_input_tokens"))
        .and_then(|v| v.as_u64())
        .map(|v| v as u32);

    // Estimate reasoning_tokens from thinking block text lengths (~4 chars per token)
    let reasoning_tokens: Option<u32> = {
        let thinking_chars: usize = content_parts
            .iter()
            .filter_map(|p| match p {
                unified_llm_types::ContentPart::Thinking { thinking } => Some(thinking.text.len()),
                _ => None,
            })
            .sum();
        if thinking_chars > 0 {
            Some((thinking_chars / 4) as u32)
        } else {
            None
        }
    };

    let usage = unified_llm_types::Usage {
        input_tokens,
        output_tokens,
        total_tokens: input_tokens + output_tokens,
        reasoning_tokens,
        cache_read_tokens,
        cache_write_tokens,
        raw: usage_obj.cloned(),
    };

    Ok(Response {
        id,
        model,
        provider: "anthropic".to_string(),
        message: unified_llm_types::Message {
            role: unified_llm_types::Role::Assistant,
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

/// Post-process a response that used tool-based structured output extraction (GAP-2).
///
/// When the adapter injects a synthetic `structured_output` tool to extract JSON,
/// the Anthropic response contains a `tool_use` block instead of text. This function
/// converts the tool's arguments into a `ContentPart::Text` so that `response.text()`
/// returns the JSON string, which `generate_object()` can then parse and validate.
///
/// Also changes `finish_reason` from `tool_calls` to `stop` since the tool call was
/// synthetic and should not trigger the tool execution loop.
fn postprocess_structured_output_tool(response: &mut Response) {
    // Find the synthetic tool call and extract its arguments as JSON text
    let json_text = response.message.content.iter().find_map(|part| {
        if let unified_llm_types::ContentPart::ToolCall { tool_call } = part {
            if tool_call.name == STRUCTURED_OUTPUT_TOOL_NAME {
                let text = match &tool_call.arguments {
                    unified_llm_types::ArgumentValue::Dict(map) => {
                        serde_json::to_string(&serde_json::Value::Object(map.clone()))
                            .unwrap_or_default()
                    }
                    unified_llm_types::ArgumentValue::Raw(s) => s.clone(),
                };
                return Some(text);
            }
        }
        None
    });

    if let Some(text) = json_text {
        // Remove the synthetic tool call from content parts
        response.message.content.retain(|p| {
            if let unified_llm_types::ContentPart::ToolCall { tool_call } = p {
                tool_call.name != STRUCTURED_OUTPUT_TOOL_NAME
            } else {
                true
            }
        });

        // Insert the JSON as text content
        response
            .message
            .content
            .insert(0, unified_llm_types::ContentPart::Text { text });

        // Change finish reason from tool_calls to stop since the tool was synthetic
        if response.finish_reason.reason == "tool_calls" {
            response.finish_reason = unified_llm_types::FinishReason::stop();
        }
    }
}

/// Parse Anthropic content blocks into unified ContentParts.
fn parse_content_blocks(blocks: &[serde_json::Value]) -> Vec<unified_llm_types::ContentPart> {
    blocks
        .iter()
        .filter_map(|block| {
            let block_type = block.get("type")?.as_str()?;
            match block_type {
                "text" => {
                    let text = block.get("text")?.as_str()?.to_string();
                    Some(unified_llm_types::ContentPart::Text { text })
                }
                "tool_use" => {
                    let id = block.get("id")?.as_str()?.to_string();
                    let name = block.get("name")?.as_str()?.to_string();
                    let input = block.get("input").cloned().unwrap_or(serde_json::json!({}));
                    let arguments = match input {
                        serde_json::Value::Object(map) => {
                            unified_llm_types::ArgumentValue::Dict(map)
                        }
                        other => unified_llm_types::ArgumentValue::Raw(other.to_string()),
                    };
                    Some(unified_llm_types::ContentPart::ToolCall {
                        tool_call: unified_llm_types::ToolCallData {
                            id,
                            name,
                            arguments,
                            r#type: "function".to_string(),
                        },
                    })
                }
                "thinking" => {
                    let text = block
                        .get("thinking")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let signature = block
                        .get("signature")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    Some(unified_llm_types::ContentPart::Thinking {
                        thinking: unified_llm_types::ThinkingData {
                            text,
                            signature,
                            redacted: false,
                            data: None,
                        },
                    })
                }
                "redacted_thinking" => {
                    let data = block
                        .get("data")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    Some(unified_llm_types::ContentPart::RedactedThinking {
                        thinking: unified_llm_types::ThinkingData {
                            text: String::new(),
                            signature: None,
                            redacted: true,
                            data,
                        },
                    })
                }
                _ => None,
            }
        })
        .collect()
}

// === Error Translation ===

/// Parse an Anthropic error response into a unified Error.
pub(crate) fn parse_error(
    status: u16,
    headers: &reqwest::header::HeaderMap,
    body: serde_json::Value,
) -> Error {
    // Extract message and code using shared utility with Anthropic JSON paths.
    let (error_message, error_code) = crate::util::http::parse_provider_error_message(
        &body,
        &["error", "message"],
        &["error", "type"],
    );

    let retry_after = crate::util::http::parse_retry_after(headers);

    let mut err =
        Error::from_http_status(status, error_message, "anthropic", Some(body), retry_after);
    err.error_code = error_code;
    err
}

#[cfg(test)]
mod tests {
    use super::*;
    use unified_llm_types::*;

    // === Builder Tests (H-8) ===

    #[test]
    fn test_anthropic_adapter_builder_defaults() {
        let adapter = AnthropicAdapterBuilder::new(SecretString::from("key".to_string())).build();
        assert_eq!(adapter.base_url, DEFAULT_BASE_URL);
        assert_eq!(adapter.name(), "anthropic");
    }

    #[test]
    fn test_anthropic_adapter_builder_with_all_options() {
        let adapter = AnthropicAdapterBuilder::new(SecretString::from("key".to_string()))
            .base_url("https://custom.api.com")
            .timeout(AdapterTimeout {
                connect: 5.0,
                request: 60.0,
                stream_read: 15.0,
            })
            .default_headers(reqwest::header::HeaderMap::new())
            .build();
        assert_eq!(adapter.base_url, "https://custom.api.com");
    }

    #[test]
    fn test_anthropic_adapter_builder_shortcut() {
        let adapter = AnthropicAdapter::builder(SecretString::from("key".to_string()))
            .base_url("https://test.com")
            .build();
        assert_eq!(adapter.base_url, "https://test.com");
    }

    // === Request Translation Tests ===

    #[test]
    fn test_anthropic_adapter_name() {
        let adapter = AnthropicAdapter::new(SecretString::from("test-key".to_string()));
        assert_eq!(adapter.name(), "anthropic");
    }

    #[test]
    fn test_system_message_extraction() {
        let messages = vec![Message::system("You are helpful"), Message::user("Hello")];
        let (body, _) = translate_request(
            &Request::default()
                .model("claude-opus-4-6")
                .messages(messages),
        );
        // System should be extracted to system parameter
        assert!(body.get("system").is_some());
        assert_eq!(body["system"], "You are helpful");
        // Messages array should NOT contain system messages
        let msgs = body["messages"].as_array().unwrap();
        assert!(msgs.iter().all(|m| m["role"] != "system"));
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "user");
    }

    #[test]
    fn test_developer_role_merged_with_system() {
        let messages = vec![
            Message::system("System instructions"),
            Message {
                role: Role::Developer,
                content: vec![ContentPart::text("Developer context")],
                name: None,
                tool_call_id: None,
            },
            Message::user("Hello"),
        ];
        let (body, _) = translate_request(&Request::default().model("test").messages(messages));
        let system = body["system"].as_str().unwrap();
        assert!(system.contains("System instructions"));
        assert!(system.contains("Developer context"));
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 1);
    }

    #[test]
    fn test_strict_alternation_merges_consecutive_same_role() {
        let messages = vec![Message::user("First"), Message::user("Second")];
        let (body, _) = translate_request(&Request::default().model("test").messages(messages));
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 1); // Merged into single user message
        let content = msgs[0]["content"].as_array().unwrap();
        assert_eq!(content.len(), 2); // Two text blocks
    }

    #[test]
    fn test_max_tokens_default_4096() {
        let (body, _) = translate_request(
            &Request::default()
                .model("test")
                .messages(vec![Message::user("Hi")]),
        );
        assert_eq!(body["max_tokens"], 4096);
    }

    #[test]
    fn test_max_tokens_explicit() {
        let (body, _) = translate_request(
            &Request::default()
                .model("test")
                .messages(vec![Message::user("Hi")])
                .max_tokens(1024),
        );
        assert_eq!(body["max_tokens"], 1024);
    }

    #[test]
    fn test_all_roles_translated() {
        let messages = vec![
            Message::system("sys"),
            Message::user("user msg"),
            Message::assistant("asst msg"),
            Message::tool_result("call_1", "result", false),
        ];
        let (body, _) = translate_request(&Request::default().model("test").messages(messages));
        assert!(body.get("system").is_some());
        let msgs = body["messages"].as_array().unwrap();
        // user, assistant, then tool_result as user => need to merge user+tool_result(user)
        // user msg (user), asst msg (assistant), tool result (user) - should not merge since they alternate
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[1]["role"], "assistant");
        assert_eq!(msgs[2]["role"], "user"); // Tool result becomes user
    }

    #[test]
    fn test_provider_options_passthrough() {
        let req = Request::default()
            .model("test")
            .messages(vec![Message::user("Hi")])
            .provider_options(Some(
                serde_json::json!({"anthropic": {"thinking": {"type": "enabled"}}}),
            ));
        let (body, _) = translate_request(&req);
        assert!(body.get("thinking").is_some());
        assert_eq!(body["thinking"]["type"], "enabled");
    }

    #[test]
    fn test_content_part_text_translation() {
        let messages = vec![Message::user("Hello world")];
        let (body, _) = translate_request(&Request::default().model("test").messages(messages));
        let content = body["messages"][0]["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "Hello world");
    }

    #[test]
    fn test_content_part_image_local_file_resolved_to_base64() {
        // Create a temp file with a .jpg extension
        let dir = std::env::temp_dir().join("unified_llm_test_anthropic_img");
        std::fs::create_dir_all(&dir).unwrap();
        let file_path = dir.join("photo.jpg");
        let fake_jpeg = vec![0xFF, 0xD8, 0xFF, 0xE0];
        std::fs::write(&file_path, &fake_jpeg).unwrap();

        let msg = Message {
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
        };
        let (body, _) = translate_request(&Request::default().model("test").messages(vec![msg]));
        let content = body["messages"][0]["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "image");
        // Must use base64 source, NOT url source
        assert_eq!(content[0]["source"]["type"], "base64");
        assert_eq!(content[0]["source"]["media_type"], "image/jpeg");
        assert!(content[0]["source"]["data"].as_str().unwrap().len() > 0);

        let _ = std::fs::remove_file(&file_path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn test_content_part_image_http_url_passed_through() {
        let msg = Message {
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
        };
        let (body, _) = translate_request(&Request::default().model("test").messages(vec![msg]));
        let content = body["messages"][0]["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "image");
        assert_eq!(content[0]["source"]["type"], "url");
        assert_eq!(content[0]["source"]["url"], "https://example.com/cat.png");
    }

    #[test]
    fn test_content_part_image_base64_translation() {
        let msg = Message {
            role: Role::User,
            content: vec![ContentPart::image_bytes(vec![0xFF, 0xD8], "image/jpeg")],
            name: None,
            tool_call_id: None,
        };
        let (body, _) = translate_request(&Request::default().model("test").messages(vec![msg]));
        let content = body["messages"][0]["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "image");
        assert_eq!(content[0]["source"]["type"], "base64");
        assert_eq!(content[0]["source"]["media_type"], "image/jpeg");
        assert!(content[0]["source"]["data"].as_str().is_some());
    }

    #[test]
    fn test_content_part_tool_call_translation() {
        let msg = Message {
            role: Role::Assistant,
            content: vec![ContentPart::ToolCall {
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
            }],
            name: None,
            tool_call_id: None,
        };
        let (body, _) = translate_request(&Request::default().model("test").messages(vec![msg]));
        let content = body["messages"][0]["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "tool_use");
        assert_eq!(content[0]["id"], "call_1");
        assert_eq!(content[0]["name"], "get_weather");
        assert_eq!(content[0]["input"]["city"], "SF");
    }

    // === Response Translation Tests ===

    #[test]
    fn test_parse_simple_text_response() {
        let raw = serde_json::json!({
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello!"}],
            "model": "claude-opus-4-6",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        });
        let response = parse_response(raw, &reqwest::header::HeaderMap::new()).unwrap();
        assert_eq!(response.text(), "Hello!");
        assert_eq!(response.finish_reason.reason, "stop");
        assert_eq!(response.finish_reason.raw, Some("end_turn".into()));
        assert_eq!(response.usage.input_tokens, 10);
        assert_eq!(response.usage.output_tokens, 5);
        assert_eq!(response.usage.total_tokens, 15);
        assert_eq!(response.provider, "anthropic");
        assert_eq!(response.id, "msg_123");
        assert_eq!(response.model, "claude-opus-4-6");
    }

    #[test]
    fn test_parse_tool_call_response() {
        let raw = serde_json::json!({
            "id": "msg_456",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me check."},
                {"type": "tool_use", "id": "call_1", "name": "get_weather", "input": {"city": "SF"}}
            ],
            "model": "claude-opus-4-6",
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 20, "output_tokens": 15}
        });
        let response = parse_response(raw, &reqwest::header::HeaderMap::new()).unwrap();
        assert_eq!(response.text(), "Let me check.");
        assert_eq!(response.finish_reason.reason, "tool_calls");
        let tool_calls = response.tool_calls();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].name, "get_weather");
        assert_eq!(tool_calls[0].id, "call_1");
    }

    #[test]
    fn test_finish_reason_mapping() {
        assert_eq!(map_finish_reason("end_turn").reason, "stop");
        assert_eq!(map_finish_reason("stop_sequence").reason, "stop");
        assert_eq!(map_finish_reason("max_tokens").reason, "length");
        assert_eq!(map_finish_reason("tool_use").reason, "tool_calls");
    }

    #[test]
    fn test_parse_usage_mapping() {
        let raw = serde_json::json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hi"}],
            "model": "claude-opus-4-6",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 100, "output_tokens": 50}
        });
        let response = parse_response(raw, &reqwest::header::HeaderMap::new()).unwrap();
        assert_eq!(response.usage.input_tokens, 100);
        assert_eq!(response.usage.output_tokens, 50);
        assert_eq!(response.usage.total_tokens, 150);
    }

    #[test]
    fn test_parse_cache_usage_tokens() {
        let raw = serde_json::json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hi"}],
            "model": "claude-opus-4-6",
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 10,
                "cache_read_input_tokens": 80,
                "cache_creation_input_tokens": 20
            }
        });
        let response = parse_response(raw, &reqwest::header::HeaderMap::new()).unwrap();
        assert_eq!(response.usage.cache_read_tokens, Some(80));
        assert_eq!(response.usage.cache_write_tokens, Some(20));
    }

    #[test]
    fn test_parse_multiple_content_blocks() {
        let raw = serde_json::json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Part 1"},
                {"type": "text", "text": "Part 2"},
            ],
            "model": "claude-opus-4-6",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        });
        let response = parse_response(raw, &reqwest::header::HeaderMap::new()).unwrap();
        assert_eq!(response.message.content.len(), 2);
        assert_eq!(response.text(), "Part 1Part 2");
    }

    // === Error Translation Tests ===

    #[test]
    fn test_parse_error_401() {
        let body = serde_json::json!({
            "type": "error",
            "error": {"type": "authentication_error", "message": "Invalid API key"}
        });
        let headers = reqwest::header::HeaderMap::new();
        let err = parse_error(401, &headers, body);
        assert_eq!(err.kind, ErrorKind::Authentication);
        assert!(!err.retryable);
        assert_eq!(err.provider, Some("anthropic".into()));
        assert_eq!(err.error_code, Some("authentication_error".into()));
    }

    #[test]
    fn test_parse_error_429_with_retry_after() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("retry-after", "5".parse().unwrap());
        let body = serde_json::json!({
            "type": "error",
            "error": {"type": "rate_limit_error", "message": "Rate limited"}
        });
        let err = parse_error(429, &headers, body);
        assert_eq!(err.kind, ErrorKind::RateLimit);
        assert!(err.retryable);
        assert_eq!(err.retry_after, Some(std::time::Duration::from_secs(5)));
        assert_eq!(err.error_code, Some("rate_limit_error".into()));
    }

    #[test]
    fn test_parse_error_500() {
        let body = serde_json::json!({
            "type": "error",
            "error": {"type": "api_error", "message": "Internal server error"}
        });
        let headers = reqwest::header::HeaderMap::new();
        let err = parse_error(500, &headers, body);
        assert_eq!(err.kind, ErrorKind::Server);
        assert!(err.retryable);
    }

    #[test]
    fn test_parse_error_body_message_extraction() {
        let body = serde_json::json!({
            "type": "error",
            "error": {"type": "invalid_request_error", "message": "context length exceeded"}
        });
        let headers = reqwest::header::HeaderMap::new();
        let err = parse_error(400, &headers, body);
        // Message classification should override to ContextLength
        assert_eq!(err.kind, ErrorKind::ContextLength);
        assert!(err.message.contains("context length exceeded"));
    }

    // === Wiremock Integration Tests (P1-T19) ===

    #[tokio::test]
    async fn test_anthropic_complete_full_roundtrip() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/v1/messages"))
            .and(wiremock::matchers::header("x-api-key", "test-key"))
            .and(wiremock::matchers::header_exists("anthropic-version"))
            .respond_with(
                wiremock::ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Hello from Claude!"}],
                    "model": "claude-opus-4-6",
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 12, "output_tokens": 6}
                })),
            )
            .mount(&server)
            .await;

        let adapter = AnthropicAdapter::new_with_base_url(
            SecretString::from("test-key".to_string()),
            server.uri(),
        );
        let req = Request::default()
            .model("claude-opus-4-6")
            .messages(vec![Message::user("Say hello")]);
        let resp = adapter.complete(req).await.unwrap();

        assert_eq!(resp.text(), "Hello from Claude!");
        assert_eq!(resp.finish_reason.reason, "stop");
        assert_eq!(resp.usage.input_tokens, 12);
        assert_eq!(resp.usage.output_tokens, 6);
        assert_eq!(resp.provider, "anthropic");
        assert_eq!(resp.id, "msg_test");
    }

    #[tokio::test]
    async fn test_anthropic_complete_error_roundtrip() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/v1/messages"))
            .respond_with(
                wiremock::ResponseTemplate::new(429)
                    .insert_header("retry-after", "5")
                    .set_body_json(serde_json::json!({
                        "type": "error",
                        "error": {"type": "rate_limit_error", "message": "Rate limited"}
                    })),
            )
            .mount(&server)
            .await;

        let adapter = AnthropicAdapter::new_with_base_url(
            SecretString::from("test-key".to_string()),
            server.uri(),
        );
        let req = Request::default()
            .model("test")
            .messages(vec![Message::user("Hi")]);
        let err = adapter.complete(req).await.unwrap_err();

        assert_eq!(err.kind, ErrorKind::RateLimit);
        assert!(err.retryable);
        assert_eq!(err.retry_after, Some(std::time::Duration::from_secs(5)));
    }

    #[tokio::test]
    async fn test_anthropic_complete_sends_correct_request_body() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/v1/messages"))
            .respond_with(
                wiremock::ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Ok"}],
                    "model": "claude-opus-4-6",
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 5, "output_tokens": 2}
                })),
            )
            .expect(1)
            .mount(&server)
            .await;

        let adapter = AnthropicAdapter::new_with_base_url(
            SecretString::from("test-key".to_string()),
            server.uri(),
        );
        let req = Request::default()
            .model("claude-opus-4-6")
            .messages(vec![
                Message::system("You are helpful"),
                Message::user("Hello"),
            ])
            .temperature(0.5);
        let resp = adapter.complete(req).await.unwrap();
        assert_eq!(resp.text(), "Ok");
    }

    #[tokio::test]
    async fn test_anthropic_complete_provider_options_passthrough() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/v1/messages"))
            .respond_with(
                wiremock::ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "thinking..."}],
                    "model": "claude-opus-4-6",
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 5}
                })),
            )
            .mount(&server)
            .await;

        let adapter = AnthropicAdapter::new_with_base_url(
            SecretString::from("test-key".to_string()),
            server.uri(),
        );
        let req = Request::default()
            .model("claude-opus-4-6")
            .messages(vec![Message::user("Think about this")])
            .provider_options(Some(
                serde_json::json!({"anthropic": {"thinking": {"type": "enabled", "budget_tokens": 1024}}}),
            ));
        let resp = adapter.complete(req).await.unwrap();
        assert_eq!(resp.text(), "thinking...");
    }

    // === Streaming Tests (P1-T20) ===

    // Use shared build_sse_body from testing module.
    use crate::testing::build_sse_body;

    #[tokio::test]
    async fn test_anthropic_stream_text_basic() {
        use futures::StreamExt;

        let sse_body = build_sse_body(&[
            (
                "message_start",
                r#"{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-20250514","usage":{"input_tokens":10,"output_tokens":0}}}"#,
            ),
            (
                "content_block_start",
                r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
            ),
            (
                "content_block_delta",
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#,
            ),
            (
                "content_block_delta",
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}"#,
            ),
            (
                "content_block_stop",
                r#"{"type":"content_block_stop","index":0}"#,
            ),
            (
                "message_delta",
                r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}"#,
            ),
            ("message_stop", r#"{"type":"message_stop"}"#),
        ]);

        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/v1/messages"))
            .respond_with(
                wiremock::ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse_body),
            )
            .mount(&server)
            .await;

        let adapter = AnthropicAdapter::new_with_base_url(
            SecretString::from("test-key".to_string()),
            server.uri(),
        );
        let req = Request::default()
            .model("claude-sonnet-4-20250514")
            .messages(vec![Message::user("Hi")]);

        let stream = adapter.stream(req);
        let events: Vec<Result<StreamEvent, Error>> = stream.collect().await;

        // Should have: StreamStart, TextStart, TextDelta, TextDelta, TextEnd, Finish
        let types: Vec<StreamEventType> = events
            .iter()
            .map(|e| e.as_ref().unwrap().event_type.clone())
            .collect();
        assert!(
            types.contains(&StreamEventType::StreamStart),
            "Missing StreamStart: {:?}",
            types
        );
        assert!(
            types.contains(&StreamEventType::TextStart),
            "Missing TextStart: {:?}",
            types
        );
        assert!(
            types.contains(&StreamEventType::TextDelta),
            "Missing TextDelta: {:?}",
            types
        );
        assert!(
            types.contains(&StreamEventType::TextEnd),
            "Missing TextEnd: {:?}",
            types
        );
        assert!(
            types.contains(&StreamEventType::Finish),
            "Missing Finish: {:?}",
            types
        );
    }

    #[tokio::test]
    async fn test_anthropic_stream_text_deltas_correct() {
        use futures::StreamExt;

        let sse_body = build_sse_body(&[
            (
                "message_start",
                r#"{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-20250514","usage":{"input_tokens":10,"output_tokens":0}}}"#,
            ),
            (
                "content_block_start",
                r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
            ),
            (
                "content_block_delta",
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#,
            ),
            (
                "content_block_delta",
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}"#,
            ),
            (
                "content_block_stop",
                r#"{"type":"content_block_stop","index":0}"#,
            ),
            (
                "message_delta",
                r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}"#,
            ),
            ("message_stop", r#"{"type":"message_stop"}"#),
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

        let adapter = AnthropicAdapter::new_with_base_url(
            SecretString::from("test-key".to_string()),
            server.uri(),
        );
        let req = Request::default()
            .model("test")
            .messages(vec![Message::user("Hi")]);

        let events: Vec<Result<StreamEvent, Error>> = adapter.stream(req).collect().await;

        // Collect text deltas
        let text: String = events
            .iter()
            .filter_map(|e| {
                let evt = e.as_ref().ok()?;
                if evt.event_type == StreamEventType::TextDelta {
                    evt.delta.clone()
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(text, "Hello world");
    }

    #[tokio::test]
    async fn test_anthropic_stream_finish_has_usage() {
        use futures::StreamExt;

        let sse_body = build_sse_body(&[
            (
                "message_start",
                r#"{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-20250514","usage":{"input_tokens":10,"output_tokens":0}}}"#,
            ),
            (
                "content_block_start",
                r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
            ),
            (
                "content_block_delta",
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}"#,
            ),
            (
                "content_block_stop",
                r#"{"type":"content_block_stop","index":0}"#,
            ),
            (
                "message_delta",
                r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}"#,
            ),
            ("message_stop", r#"{"type":"message_stop"}"#),
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

        let adapter = AnthropicAdapter::new_with_base_url(
            SecretString::from("test-key".to_string()),
            server.uri(),
        );
        let req = Request::default()
            .model("test")
            .messages(vec![Message::user("Hi")]);

        let events: Vec<Result<StreamEvent, Error>> = adapter.stream(req).collect().await;

        // Finish event should have usage and finish_reason
        let finish = events
            .iter()
            .filter_map(|e| e.as_ref().ok())
            .find(|e| e.event_type == StreamEventType::Finish)
            .expect("Should have Finish event");
        assert!(finish.usage.is_some(), "Finish should have usage");
        let usage = finish.usage.as_ref().unwrap();
        assert_eq!(usage.input_tokens, 10);
        assert_eq!(usage.output_tokens, 5);
        assert!(
            finish.finish_reason.is_some(),
            "Finish should have finish_reason"
        );
        assert_eq!(finish.finish_reason.as_ref().unwrap().reason, "stop");
    }

    #[tokio::test]
    async fn test_anthropic_stream_start_has_id() {
        use futures::StreamExt;

        let sse_body = build_sse_body(&[
            (
                "message_start",
                r#"{"type":"message_start","message":{"id":"msg_abc123","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-20250514","usage":{"input_tokens":10,"output_tokens":0}}}"#,
            ),
            (
                "content_block_start",
                r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
            ),
            (
                "content_block_delta",
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}"#,
            ),
            (
                "content_block_stop",
                r#"{"type":"content_block_stop","index":0}"#,
            ),
            (
                "message_delta",
                r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}"#,
            ),
            ("message_stop", r#"{"type":"message_stop"}"#),
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

        let adapter = AnthropicAdapter::new_with_base_url(
            SecretString::from("test-key".to_string()),
            server.uri(),
        );
        let req = Request::default()
            .model("test")
            .messages(vec![Message::user("Hi")]);

        let events: Vec<Result<StreamEvent, Error>> = adapter.stream(req).collect().await;

        let start = events
            .iter()
            .filter_map(|e| e.as_ref().ok())
            .find(|e| e.event_type == StreamEventType::StreamStart)
            .expect("Should have StreamStart");
        assert_eq!(start.id, Some("msg_abc123".to_string()));
    }

    #[tokio::test]
    async fn test_anthropic_stream_sends_stream_true() {
        use futures::StreamExt;

        let sse_body = build_sse_body(&[
            (
                "message_start",
                r#"{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"test","usage":{"input_tokens":5,"output_tokens":0}}}"#,
            ),
            (
                "message_delta",
                r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}"#,
            ),
            ("message_stop", r#"{"type":"message_stop"}"#),
        ]);

        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/v1/messages"))
            .and(wiremock::matchers::body_json(serde_json::json!({
                "model": "test",
                "max_tokens": 4096,
                "stream": true,
                "messages": [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}]
            })))
            .respond_with(
                wiremock::ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse_body),
            )
            .mount(&server)
            .await;

        let adapter = AnthropicAdapter::new_with_base_url(
            SecretString::from("test-key".to_string()),
            server.uri(),
        );
        let req = Request::default()
            .model("test")
            .messages(vec![Message::user("Hi")]);

        // If the body matcher fails, we get a 404 from wiremock
        let events: Vec<Result<StreamEvent, Error>> = adapter.stream(req).collect().await;
        // Should have events (not empty, which is what the current stub returns)
        assert!(!events.is_empty(), "Stream should not be empty");
    }

    #[tokio::test]
    async fn test_anthropic_stream_ping_ignored() {
        use futures::StreamExt;

        let sse_body = build_sse_body(&[
            (
                "message_start",
                r#"{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"test","usage":{"input_tokens":5,"output_tokens":0}}}"#,
            ),
            ("ping", r#"{"type":"ping"}"#),
            (
                "content_block_start",
                r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
            ),
            (
                "content_block_delta",
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}"#,
            ),
            (
                "content_block_stop",
                r#"{"type":"content_block_stop","index":0}"#,
            ),
            (
                "message_delta",
                r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}"#,
            ),
            ("message_stop", r#"{"type":"message_stop"}"#),
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

        let adapter = AnthropicAdapter::new_with_base_url(
            SecretString::from("test-key".to_string()),
            server.uri(),
        );
        let req = Request::default()
            .model("test")
            .messages(vec![Message::user("Hi")]);

        let events: Vec<Result<StreamEvent, Error>> = adapter.stream(req).collect().await;
        // No event should be a ping-related type; all should be real events
        let types: Vec<StreamEventType> = events
            .iter()
            .filter_map(|e| e.as_ref().ok())
            .map(|e| e.event_type.clone())
            .collect();
        // Should NOT contain any ProviderEvent for ping
        for t in &types {
            assert_ne!(
                *t,
                StreamEventType::ProviderEvent,
                "ping should be silently ignored"
            );
        }
    }

    #[tokio::test]
    async fn test_anthropic_stream_error_response() {
        use futures::StreamExt;

        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .respond_with(
                wiremock::ResponseTemplate::new(429)
                    .insert_header("retry-after", "5")
                    .set_body_json(serde_json::json!({
                        "type": "error",
                        "error": {"type": "rate_limit_error", "message": "Rate limited"}
                    })),
            )
            .mount(&server)
            .await;

        let adapter = AnthropicAdapter::new_with_base_url(
            SecretString::from("test-key".to_string()),
            server.uri(),
        );
        let req = Request::default()
            .model("test")
            .messages(vec![Message::user("Hi")]);

        let events: Vec<Result<StreamEvent, Error>> = adapter.stream(req).collect().await;
        // Should yield exactly one error
        assert_eq!(events.len(), 1);
        let err = events[0].as_ref().unwrap_err();
        assert_eq!(err.kind, ErrorKind::RateLimit);
    }

    // === Streaming Tool Call Tests (P1-T21) ===

    #[tokio::test]
    async fn test_anthropic_stream_tool_call() {
        use futures::StreamExt;

        let sse_body = build_sse_body(&[
            (
                "message_start",
                r#"{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-20250514","usage":{"input_tokens":10,"output_tokens":0}}}"#,
            ),
            (
                "content_block_start",
                r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather"}}"#,
            ),
            (
                "content_block_delta",
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"city\":"}}"#,
            ),
            (
                "content_block_delta",
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\"SF\"}"}}"#,
            ),
            (
                "content_block_stop",
                r#"{"type":"content_block_stop","index":0}"#,
            ),
            (
                "message_delta",
                r#"{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":15}}"#,
            ),
            ("message_stop", r#"{"type":"message_stop"}"#),
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

        let adapter = AnthropicAdapter::new_with_base_url(
            SecretString::from("test-key".to_string()),
            server.uri(),
        );
        let req = Request::default()
            .model("test")
            .messages(vec![Message::user("What's the weather?")]);

        let events: Vec<Result<StreamEvent, Error>> = adapter.stream(req).collect().await;
        let types: Vec<StreamEventType> = events
            .iter()
            .filter_map(|e| e.as_ref().ok())
            .map(|e| e.event_type.clone())
            .collect();

        assert!(
            types.contains(&StreamEventType::ToolCallStart),
            "Missing ToolCallStart: {:?}",
            types
        );
        assert!(
            types.contains(&StreamEventType::ToolCallDelta),
            "Missing ToolCallDelta: {:?}",
            types
        );
        assert!(
            types.contains(&StreamEventType::ToolCallEnd),
            "Missing ToolCallEnd: {:?}",
            types
        );

        // Verify ToolCallStart has id and name
        let tc_start = events
            .iter()
            .filter_map(|e| e.as_ref().ok())
            .find(|e| e.event_type == StreamEventType::ToolCallStart)
            .unwrap();
        let tc = tc_start.tool_call.as_ref().unwrap();
        assert_eq!(tc.id, "toolu_1");
        assert_eq!(tc.name, "get_weather");

        // Verify deltas contain partial JSON
        let arg_chunks: String = events
            .iter()
            .filter_map(|e| e.as_ref().ok())
            .filter(|e| e.event_type == StreamEventType::ToolCallDelta)
            .filter_map(|e| e.delta.clone())
            .collect();
        assert_eq!(arg_chunks, r#"{"city":"SF"}"#);

        // Verify finish reason is tool_calls
        let finish = events
            .iter()
            .filter_map(|e| e.as_ref().ok())
            .find(|e| e.event_type == StreamEventType::Finish)
            .unwrap();
        assert_eq!(finish.finish_reason.as_ref().unwrap().reason, "tool_calls");
    }

    #[tokio::test]
    async fn test_anthropic_stream_interleaved_text_and_tool_call() {
        use futures::StreamExt;

        let sse_body = build_sse_body(&[
            (
                "message_start",
                r#"{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-20250514","usage":{"input_tokens":10,"output_tokens":0}}}"#,
            ),
            // Text block first
            (
                "content_block_start",
                r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
            ),
            (
                "content_block_delta",
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Let me check."}}"#,
            ),
            (
                "content_block_stop",
                r#"{"type":"content_block_stop","index":0}"#,
            ),
            // Tool call block second
            (
                "content_block_start",
                r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_2","name":"search"}}"#,
            ),
            (
                "content_block_delta",
                r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"q\":\"rust\"}"}}"#,
            ),
            (
                "content_block_stop",
                r#"{"type":"content_block_stop","index":1}"#,
            ),
            (
                "message_delta",
                r#"{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":20}}"#,
            ),
            ("message_stop", r#"{"type":"message_stop"}"#),
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

        let adapter = AnthropicAdapter::new_with_base_url(
            SecretString::from("test-key".to_string()),
            server.uri(),
        );
        let req = Request::default()
            .model("test")
            .messages(vec![Message::user("Search for something")]);

        let events: Vec<Result<StreamEvent, Error>> = adapter.stream(req).collect().await;
        let types: Vec<StreamEventType> = events
            .iter()
            .filter_map(|e| e.as_ref().ok())
            .map(|e| e.event_type.clone())
            .collect();

        // Verify correct ordering: StreamStart, TextStart, TextDelta, TextEnd, ToolCallStart, ToolCallDelta, ToolCallEnd, Finish
        let expected_order = vec![
            StreamEventType::StreamStart,
            StreamEventType::TextStart,
            StreamEventType::TextDelta,
            StreamEventType::TextEnd,
            StreamEventType::ToolCallStart,
            StreamEventType::ToolCallDelta,
            StreamEventType::ToolCallEnd,
            StreamEventType::Finish,
        ];
        assert_eq!(types, expected_order, "Event order mismatch");
    }

    #[tokio::test]
    async fn test_anthropic_stream_multiple_tool_calls() {
        use futures::StreamExt;

        let sse_body = build_sse_body(&[
            (
                "message_start",
                r#"{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"test","usage":{"input_tokens":5,"output_tokens":0}}}"#,
            ),
            // First tool call
            (
                "content_block_start",
                r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_a","name":"get_weather"}}"#,
            ),
            (
                "content_block_delta",
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"city\":\"SF\"}"}}"#,
            ),
            (
                "content_block_stop",
                r#"{"type":"content_block_stop","index":0}"#,
            ),
            // Second tool call
            (
                "content_block_start",
                r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_b","name":"get_time"}}"#,
            ),
            (
                "content_block_delta",
                r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"tz\":\"PST\"}"}}"#,
            ),
            (
                "content_block_stop",
                r#"{"type":"content_block_stop","index":1}"#,
            ),
            (
                "message_delta",
                r#"{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":25}}"#,
            ),
            ("message_stop", r#"{"type":"message_stop"}"#),
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

        let adapter = AnthropicAdapter::new_with_base_url(
            SecretString::from("test-key".to_string()),
            server.uri(),
        );
        let req = Request::default()
            .model("test")
            .messages(vec![Message::user("Do two things")]);

        let events: Vec<Result<StreamEvent, Error>> = adapter.stream(req).collect().await;

        // Count tool call starts
        let tc_starts: Vec<&StreamEvent> = events
            .iter()
            .filter_map(|e| e.as_ref().ok())
            .filter(|e| e.event_type == StreamEventType::ToolCallStart)
            .collect();
        assert_eq!(tc_starts.len(), 2);

        // Verify ids
        assert_eq!(tc_starts[0].tool_call.as_ref().unwrap().id, "toolu_a");
        assert_eq!(tc_starts[0].tool_call.as_ref().unwrap().name, "get_weather");
        assert_eq!(tc_starts[1].tool_call.as_ref().unwrap().id, "toolu_b");
        assert_eq!(tc_starts[1].tool_call.as_ref().unwrap().name, "get_time");

        // Count tool call ends
        let tc_ends = events
            .iter()
            .filter_map(|e| e.as_ref().ok())
            .filter(|e| e.event_type == StreamEventType::ToolCallEnd)
            .count();
        assert_eq!(tc_ends, 2);
    }

    // === Thinking Block Tests (P1-T22) ===

    #[test]
    fn test_parse_thinking_block_with_signature() {
        let raw = serde_json::json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Let me reason about this...", "signature": "sig_abc123xyz"},
                {"type": "text", "text": "The answer is 42"}
            ],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 20}
        });
        let resp = parse_response(raw, &reqwest::header::HeaderMap::new()).unwrap();
        assert_eq!(resp.message.content.len(), 2);

        // Verify thinking block
        match &resp.message.content[0] {
            ContentPart::Thinking { thinking } => {
                assert_eq!(thinking.text, "Let me reason about this...");
                // DoD 8.5.4: signature must round-trip verbatim
                assert_eq!(thinking.signature, Some("sig_abc123xyz".to_string()));
                assert!(!thinking.redacted);
            }
            other => panic!("Expected Thinking, got {:?}", other),
        }
        assert_eq!(resp.text(), "The answer is 42");
        assert_eq!(
            resp.reasoning(),
            Some("Let me reason about this...".to_string())
        );
    }

    #[test]
    fn test_parse_redacted_thinking_block() {
        let raw = serde_json::json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "redacted_thinking", "data": "opaque_binary_data"},
                {"type": "text", "text": "Answer"}
            ],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        });
        let resp = parse_response(raw, &reqwest::header::HeaderMap::new()).unwrap();
        match &resp.message.content[0] {
            ContentPart::RedactedThinking { thinking } => {
                assert!(thinking.redacted);
                assert!(thinking.text.is_empty());
            }
            other => panic!("Expected RedactedThinking, got {:?}", other),
        }
    }

    #[test]
    fn test_thinking_request_passthrough() {
        // Verify that provider_options.anthropic.thinking gets passed through to request body
        let req = Request::default()
            .model("claude-sonnet-4-20250514")
            .messages(vec![Message::user("Think about this")])
            .provider_options(Some(serde_json::json!({
                "anthropic": {
                    "thinking": {"type": "enabled", "budget_tokens": 2048}
                }
            })));
        let (body, _) = translate_request(&req);
        assert!(body.get("thinking").is_some());
        assert_eq!(body["thinking"]["type"], "enabled");
        assert_eq!(body["thinking"]["budget_tokens"], 2048);
    }

    #[tokio::test]
    async fn test_anthropic_stream_thinking_blocks() {
        use futures::StreamExt;

        let sse_body = build_sse_body(&[
            (
                "message_start",
                r#"{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-20250514","usage":{"input_tokens":10,"output_tokens":0}}}"#,
            ),
            // Thinking block
            (
                "content_block_start",
                r#"{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}"#,
            ),
            (
                "content_block_delta",
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Let me "}}"#,
            ),
            (
                "content_block_delta",
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"reason..."}}"#,
            ),
            (
                "content_block_stop",
                r#"{"type":"content_block_stop","index":0}"#,
            ),
            // Text block
            (
                "content_block_start",
                r#"{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}"#,
            ),
            (
                "content_block_delta",
                r#"{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"42"}}"#,
            ),
            (
                "content_block_stop",
                r#"{"type":"content_block_stop","index":1}"#,
            ),
            (
                "message_delta",
                r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":15}}"#,
            ),
            ("message_stop", r#"{"type":"message_stop"}"#),
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

        let adapter = AnthropicAdapter::new_with_base_url(
            SecretString::from("test-key".to_string()),
            server.uri(),
        );
        let req = Request::default()
            .model("test")
            .messages(vec![Message::user("Think hard")]);

        let events: Vec<Result<StreamEvent, Error>> = adapter.stream(req).collect().await;
        let types: Vec<StreamEventType> = events
            .iter()
            .filter_map(|e| e.as_ref().ok())
            .map(|e| e.event_type.clone())
            .collect();

        // Verify thinking events present
        assert!(
            types.contains(&StreamEventType::ReasoningStart),
            "Missing ReasoningStart: {:?}",
            types
        );
        assert!(
            types.contains(&StreamEventType::ReasoningDelta),
            "Missing ReasoningDelta: {:?}",
            types
        );
        assert!(
            types.contains(&StreamEventType::ReasoningEnd),
            "Missing ReasoningEnd: {:?}",
            types
        );

        // Verify correct ordering: reasoning before text
        let expected_order = vec![
            StreamEventType::StreamStart,
            StreamEventType::ReasoningStart,
            StreamEventType::ReasoningDelta,
            StreamEventType::ReasoningDelta,
            StreamEventType::ReasoningEnd,
            StreamEventType::TextStart,
            StreamEventType::TextDelta,
            StreamEventType::TextEnd,
            StreamEventType::Finish,
        ];
        assert_eq!(types, expected_order);

        // Verify reasoning delta content
        let reasoning: String = events
            .iter()
            .filter_map(|e| e.as_ref().ok())
            .filter(|e| e.event_type == StreamEventType::ReasoningDelta)
            .filter_map(|e| e.delta.clone())
            .collect();
        assert_eq!(reasoning, "Let me reason...");
    }

    // === Prompt Caching Tests (P1-T23) ===

    #[test]
    fn test_cache_control_injected_on_system_prompt() {
        // DoD 8.6.3: auto-inject cache_control on last system block
        let req = Request::default().model("test").messages(vec![
            Message::system("System prompt"),
            Message::user("Hello"),
        ]);
        let (body, _) = translate_request_with_cache(&req);
        // System should be an array of content blocks with cache_control on the last one
        let system = body.get("system").expect("should have system");
        let system_arr = system
            .as_array()
            .expect("system should be array when cached");
        let last_block = system_arr.last().unwrap();
        assert_eq!(
            last_block.get("cache_control"),
            Some(&serde_json::json!({"type": "ephemeral"})),
            "Last system block should have cache_control"
        );
    }

    #[test]
    fn test_cache_control_injected_on_tools() {
        // DoD 8.6.3: auto-inject cache_control on tool definitions
        let req = Request::default()
            .model("test")
            .messages(vec![Message::user("Hello")])
            .tools(vec![
                unified_llm_types::ToolDefinition {
                    name: "tool1".into(),
                    description: "desc1".into(),
                    parameters: serde_json::json!({"type": "object"}),
                    strict: None,
                },
                unified_llm_types::ToolDefinition {
                    name: "tool2".into(),
                    description: "desc2".into(),
                    parameters: serde_json::json!({"type": "object"}),
                    strict: None,
                },
            ]);
        let (body, _) = translate_request_with_cache(&req);
        let tools = body.get("tools").expect("should have tools");
        let tools_arr = tools.as_array().expect("tools should be array");
        // Last tool should have cache_control
        let last_tool = tools_arr.last().unwrap();
        assert_eq!(
            last_tool.get("cache_control"),
            Some(&serde_json::json!({"type": "ephemeral"})),
            "Last tool should have cache_control"
        );
        // First tool should NOT have cache_control
        assert!(tools_arr[0].get("cache_control").is_none());
    }

    #[test]
    fn test_cache_control_injected_on_conversation_prefix() {
        // DoD 8.6.3: cache_control on last message before the final user message
        let req = Request::default().model("test").messages(vec![
            Message::user("First question"),
            Message::assistant("First answer"),
            Message::user("Second question"), // This is the final user message
        ]);
        let (body, _) = translate_request_with_cache(&req);
        let messages = body.get("messages").unwrap().as_array().unwrap();
        // messages[0] = user "First question"
        // messages[1] = assistant "First answer" -- this should get cache_control
        // messages[2] = user "Second question" -- final user message, no cache_control
        assert_eq!(messages.len(), 3);
        // The last message before the final user message should have cache_control on its last content block
        let prefix_msg = &messages[1];
        let content = prefix_msg.get("content").unwrap().as_array().unwrap();
        let last_content_block = content.last().unwrap();
        assert_eq!(
            last_content_block.get("cache_control"),
            Some(&serde_json::json!({"type": "ephemeral"})),
            "Last content block of conversation prefix should have cache_control"
        );
        // Final user message should NOT have cache_control
        let final_msg = &messages[2];
        let final_content = final_msg.get("content").unwrap().as_array().unwrap();
        assert!(final_content.last().unwrap().get("cache_control").is_none());
    }

    #[test]
    fn test_cache_auto_cache_false_disables_injection() {
        // DoD 8.6.6: auto_cache = false disables injection
        let req = Request::default()
            .model("test")
            .messages(vec![
                Message::system("System prompt"),
                Message::user("Hello"),
            ])
            .provider_options(Some(serde_json::json!({
                "anthropic": {"auto_cache": false}
            })));
        let (body, _) = translate_request_with_cache(&req);
        // System should be a plain string, not array (no cache injection)
        let system = body.get("system").expect("should have system");
        assert!(
            system.is_string(),
            "System should be plain string when auto_cache is false"
        );
    }

    #[test]
    fn test_cache_control_not_on_single_user_message() {
        // When there's only one user message, no conversation prefix to cache
        let req = Request::default()
            .model("test")
            .messages(vec![Message::user("Hello")]);
        let (body, _) = translate_request_with_cache(&req);
        let messages = body.get("messages").unwrap().as_array().unwrap();
        assert_eq!(messages.len(), 1);
        // The only message is the final user message, no cache_control
        let content = messages[0].get("content").unwrap().as_array().unwrap();
        assert!(content[0].get("cache_control").is_none());
    }

    #[test]
    fn test_should_auto_cache_default_true() {
        assert!(should_auto_cache(None));
        assert!(should_auto_cache(Some(&serde_json::json!({}))));
        assert!(should_auto_cache(Some(
            &serde_json::json!({"anthropic": {}})
        )));
    }

    #[test]
    fn test_should_auto_cache_explicit_false() {
        assert!(!should_auto_cache(Some(
            &serde_json::json!({"anthropic": {"auto_cache": false}})
        )));
    }

    #[test]
    fn test_should_auto_cache_explicit_true() {
        assert!(should_auto_cache(Some(
            &serde_json::json!({"anthropic": {"auto_cache": true}})
        )));
    }

    #[test]
    fn test_cache_control_combined_system_tools_and_prefix() {
        // L-4: Verify cache_control is injected on all three targets simultaneously:
        // system prompt, last tool, and conversation prefix message.
        let req = Request::default()
            .model("test")
            .messages(vec![
                Message::system("You are a helpful assistant"),
                Message::user("First question"),
                Message::assistant("First answer"),
                Message::user("Second question"),
            ])
            .tools(vec![
                unified_llm_types::ToolDefinition {
                    name: "tool_a".into(),
                    description: "Tool A".into(),
                    parameters: serde_json::json!({"type": "object"}),
                    strict: None,
                },
                unified_llm_types::ToolDefinition {
                    name: "tool_b".into(),
                    description: "Tool B".into(),
                    parameters: serde_json::json!({"type": "object"}),
                    strict: None,
                },
            ]);
        let (body, _) = translate_request_with_cache(&req);
        let cache_ephemeral = serde_json::json!({"type": "ephemeral"});

        // 1. System prompt: last block should have cache_control
        let system = body.get("system").expect("should have system");
        let system_arr = system.as_array().expect("system should be array");
        let last_system = system_arr.last().unwrap();
        assert_eq!(
            last_system.get("cache_control"),
            Some(&cache_ephemeral),
            "Last system block should have cache_control"
        );

        // 2. Tools: last tool should have cache_control, first should not
        let tools = body.get("tools").expect("should have tools");
        let tools_arr = tools.as_array().expect("tools should be array");
        assert_eq!(tools_arr.len(), 2);
        assert!(
            tools_arr[0].get("cache_control").is_none(),
            "First tool should NOT have cache_control"
        );
        assert_eq!(
            tools_arr[1].get("cache_control"),
            Some(&cache_ephemeral),
            "Last tool should have cache_control"
        );

        // 3. Conversation prefix: message before final user should have cache_control
        let messages = body.get("messages").unwrap().as_array().unwrap();
        assert_eq!(messages.len(), 3); // user, assistant, user (system extracted)
                                       // messages[1] = assistant "First answer" — this is the prefix message
        let prefix_msg = &messages[1];
        let content = prefix_msg.get("content").unwrap().as_array().unwrap();
        let last_block = content.last().unwrap();
        assert_eq!(
            last_block.get("cache_control"),
            Some(&cache_ephemeral),
            "Last content block of conversation prefix should have cache_control"
        );
        // Final user message should NOT have cache_control
        let final_msg = &messages[2];
        let final_content = final_msg.get("content").unwrap().as_array().unwrap();
        assert!(
            final_content.last().unwrap().get("cache_control").is_none(),
            "Final user message should NOT have cache_control"
        );
    }

    // === Beta Header Tests (P1-T24) ===

    #[test]
    fn test_collect_beta_headers_auto_cache() {
        // DoD 8.6.4: prompt-caching-2024-07-31 auto-added when caching enabled
        let req = Request::default()
            .model("test")
            .messages(vec![Message::user("Hi")]);
        let betas = collect_beta_headers(&req);
        assert!(betas.contains(&"prompt-caching-2024-07-31".to_string()));
    }

    #[test]
    fn test_collect_beta_headers_auto_cache_disabled() {
        // When auto_cache is false, no cache beta header
        let req = Request::default()
            .model("test")
            .messages(vec![Message::user("Hi")])
            .provider_options(Some(serde_json::json!({
                "anthropic": {"auto_cache": false}
            })));
        let betas = collect_beta_headers(&req);
        assert!(!betas.contains(&"prompt-caching-2024-07-31".to_string()));
    }

    #[test]
    fn test_collect_beta_headers_user_specified() {
        // User-specified betas from provider_options.anthropic.betas
        let req = Request::default()
            .model("test")
            .messages(vec![Message::user("Hi")])
            .provider_options(Some(serde_json::json!({
                "anthropic": {
                    "auto_cache": false,
                    "betas": ["interleaved-thinking-2025-05-14", "custom-beta"]
                }
            })));
        let betas = collect_beta_headers(&req);
        assert!(betas.contains(&"interleaved-thinking-2025-05-14".to_string()));
        assert!(betas.contains(&"custom-beta".to_string()));
        assert!(!betas.contains(&"prompt-caching-2024-07-31".to_string()));
    }

    #[test]
    fn test_collect_beta_headers_merge_and_dedup() {
        // User specifies cache beta AND auto-cache is on: should deduplicate
        let req = Request::default()
            .model("test")
            .messages(vec![Message::user("Hi")])
            .provider_options(Some(serde_json::json!({
                "anthropic": {
                    "betas": ["prompt-caching-2024-07-31", "extra-beta"]
                }
            })));
        let betas = collect_beta_headers(&req);
        // Should have exactly one copy of prompt-caching + extra-beta
        let cache_count = betas
            .iter()
            .filter(|b| *b == "prompt-caching-2024-07-31")
            .count();
        assert_eq!(cache_count, 1, "prompt-caching should appear exactly once");
        assert!(betas.contains(&"extra-beta".to_string()));
    }

    // --- M-10: beta_headers (spec name) vs betas (legacy name) ---

    #[test]
    fn test_collect_beta_headers_spec_name() {
        // M-10: provider_options.anthropic.beta_headers is the spec name
        let req = Request::default()
            .model("test")
            .messages(vec![Message::user("Hi")])
            .provider_options(Some(serde_json::json!({
                "anthropic": {
                    "auto_cache": false,
                    "beta_headers": ["interleaved-thinking-2025-05-14", "spec-beta"]
                }
            })));
        let betas = collect_beta_headers(&req);
        assert!(betas.contains(&"interleaved-thinking-2025-05-14".to_string()));
        assert!(betas.contains(&"spec-beta".to_string()));
    }

    #[test]
    fn test_collect_beta_headers_spec_name_takes_precedence() {
        // M-10: When both beta_headers and betas are present, beta_headers wins
        let req = Request::default()
            .model("test")
            .messages(vec![Message::user("Hi")])
            .provider_options(Some(serde_json::json!({
                "anthropic": {
                    "auto_cache": false,
                    "beta_headers": ["from-spec"],
                    "betas": ["from-legacy"]
                }
            })));
        let betas = collect_beta_headers(&req);
        assert!(
            betas.contains(&"from-spec".to_string()),
            "beta_headers should take precedence"
        );
        assert!(
            !betas.contains(&"from-legacy".to_string()),
            "betas should be ignored when beta_headers is present"
        );
    }

    #[test]
    fn test_collect_beta_headers_legacy_name_still_works() {
        // M-10: Legacy betas field still works when beta_headers is absent
        let req = Request::default()
            .model("test")
            .messages(vec![Message::user("Hi")])
            .provider_options(Some(serde_json::json!({
                "anthropic": {
                    "auto_cache": false,
                    "betas": ["legacy-beta"]
                }
            })));
        let betas = collect_beta_headers(&req);
        assert!(betas.contains(&"legacy-beta".to_string()));
    }

    #[test]
    fn test_beta_headers_not_leaked_to_api_body() {
        // M-10: beta_headers should be filtered from API body (internal key)
        let req = Request::default()
            .model("test")
            .messages(vec![Message::user("Hi")])
            .provider_options(Some(serde_json::json!({
                "anthropic": {
                    "beta_headers": ["some-beta"],
                    "auto_cache": false,
                    "custom_field": "value"
                }
            })));
        let (body, _) = translate_request(&req);
        assert!(
            body.get("beta_headers").is_none(),
            "beta_headers should be filtered from body"
        );
        assert!(
            body.get("betas").is_none(),
            "betas should be filtered from body"
        );
        assert_eq!(
            body.get("custom_field").and_then(|v| v.as_str()),
            Some("value"),
            "non-internal keys should pass through"
        );
    }

    #[tokio::test]
    async fn test_anthropic_beta_header_sent_in_request() {
        // DoD 8.2.16: anthropic-beta header sent with comma-separated betas
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/v1/messages"))
            .and(wiremock::matchers::header_exists("anthropic-beta"))
            .respond_with(
                wiremock::ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Ok"}],
                    "model": "test",
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 5, "output_tokens": 2}
                })),
            )
            .expect(1)
            .mount(&server)
            .await;

        let adapter = AnthropicAdapter::new_with_base_url(
            SecretString::from("test-key".to_string()),
            server.uri(),
        );
        // Default request should auto-include prompt-caching beta
        let req = Request::default()
            .model("test")
            .messages(vec![Message::user("Hi")]);
        let resp = adapter.complete(req).await.unwrap();
        assert_eq!(resp.text(), "Ok");
        // If the header matcher failed, wiremock would 404
    }

    #[tokio::test]
    async fn test_anthropic_no_beta_header_when_none() {
        // When auto_cache is false and no user betas, no anthropic-beta header should be sent
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/v1/messages"))
            .respond_with(
                wiremock::ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Ok"}],
                    "model": "test",
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 5, "output_tokens": 2}
                })),
            )
            .expect(1)
            .mount(&server)
            .await;

        let adapter = AnthropicAdapter::new_with_base_url(
            SecretString::from("test-key".to_string()),
            server.uri(),
        );
        let req = Request::default()
            .model("test")
            .messages(vec![Message::user("Hi")])
            .provider_options(Some(serde_json::json!({
                "anthropic": {"auto_cache": false}
            })));
        let resp = adapter.complete(req).await.unwrap();
        assert_eq!(resp.text(), "Ok");
    }

    #[tokio::test]
    async fn test_anthropic_complete_thinking_roundtrip() {
        // Full wiremock roundtrip: complete() returns thinking blocks
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .respond_with(
                wiremock::ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "id": "msg_think",
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "Deep thought...", "signature": "sig_preserve_me"},
                        {"type": "text", "text": "Result: 42"}
                    ],
                    "model": "claude-sonnet-4-20250514",
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 20, "output_tokens": 30}
                })),
            )
            .mount(&server)
            .await;

        let adapter = AnthropicAdapter::new_with_base_url(
            SecretString::from("test-key".to_string()),
            server.uri(),
        );
        let req = Request::default()
            .model("claude-sonnet-4-20250514")
            .messages(vec![Message::user("Think")])
            .provider_options(Some(serde_json::json!({
                "anthropic": {"thinking": {"type": "enabled", "budget_tokens": 1024}}
            })));
        let resp = adapter.complete(req).await.unwrap();

        assert_eq!(resp.reasoning(), Some("Deep thought...".to_string()));
        assert_eq!(resp.text(), "Result: 42");

        // Verify signature preserved verbatim
        match &resp.message.content[0] {
            ContentPart::Thinking { thinking } => {
                assert_eq!(thinking.signature, Some("sig_preserve_me".to_string()));
            }
            other => panic!("Expected Thinking, got {:?}", other),
        }
    }

    #[test]
    fn test_internal_keys_filtered_from_passthrough() {
        // Internal keys (betas, auto_cache) should NOT leak into the API body
        let req = Request::default()
            .model("test")
            .messages(vec![Message::user("Hi")])
            .provider_options(Some(serde_json::json!({
                "anthropic": {
                    "betas": ["some-beta"],
                    "auto_cache": false,
                    "thinking": {"type": "enabled", "budget_tokens": 1024},
                    "metadata": {"user_id": "u123"}
                }
            })));
        let (body, _) = translate_request(&req);
        // Internal keys must NOT appear in the body
        assert!(
            body.get("betas").is_none(),
            "betas should be filtered from body"
        );
        assert!(
            body.get("auto_cache").is_none(),
            "auto_cache should be filtered from body"
        );
        // Real Anthropic API parameters MUST pass through
        assert!(
            body.get("thinking").is_some(),
            "thinking should pass through"
        );
        assert!(
            body.get("metadata").is_some(),
            "metadata should pass through"
        );
    }

    #[test]
    fn test_build_headers_invalid_api_key_returns_error() {
        // API keys with non-ASCII or control characters should return an error, not panic
        let adapter = AnthropicAdapter::new(SecretString::from("valid-key\x00evil".to_string()));
        let result = adapter.build_headers(&[]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.kind, ErrorKind::Configuration);
        assert!(err.message.contains("Invalid API key"));
    }

    #[test]
    fn test_build_headers_invalid_beta_value_returns_error() {
        // Beta header values with control characters should return an error, not panic
        let adapter = AnthropicAdapter::new(SecretString::from("valid-key".to_string()));
        let result = adapter.build_headers(&["bad\x00beta".to_string()]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.kind, ErrorKind::Configuration);
        assert!(err.message.contains("Invalid beta header"));
    }

    #[test]
    fn test_build_headers_valid_key_succeeds() {
        let adapter = AnthropicAdapter::new(SecretString::from("sk-ant-valid-key-123".to_string()));
        let result = adapter.build_headers(&[]);
        assert!(result.is_ok());
        let headers = result.unwrap();
        assert!(headers.contains_key("x-api-key"));
        assert!(headers.contains_key("anthropic-version"));
        assert!(headers.contains_key("content-type"));
    }

    // === AF-04: Anthropic Structured Output Tests ===

    #[test]
    fn test_anthropic_structured_output_uses_tool_extraction() {
        // GAP-2: When response_format is json_schema and no tools are present,
        // the primary strategy is tool-based extraction (synthetic tool + forced tool_choice).
        let req = Request::default()
            .model("claude-sonnet-4-20250514")
            .messages(vec![
                Message::system("You are helpful."),
                Message::user("What is 2+2?"),
            ])
            .response_format(ResponseFormat {
                r#type: "json_schema".to_string(),
                json_schema: Some(serde_json::json!({
                    "type": "object",
                    "properties": {"answer": {"type": "integer"}},
                    "required": ["answer"]
                })),
                strict: true,
            });
        let (body, _) = translate_request(&req);

        // System message preserved, no schema injection in system prompt
        let system_text = body.get("system").and_then(|v| v.as_str()).unwrap_or("");
        assert!(
            system_text.contains("You are helpful."),
            "Original system message should be preserved"
        );

        // Tool-based extraction: synthetic tool should be present
        let tools = body.get("tools").and_then(|v| v.as_array());
        assert!(
            tools.is_some(),
            "Tool-based extraction should add a synthetic tool"
        );
        let tools = tools.unwrap();
        assert_eq!(tools.len(), 1, "Exactly one synthetic tool");
        assert_eq!(
            tools[0].get("name").and_then(|v| v.as_str()),
            Some("structured_output"),
            "Synthetic tool should be named structured_output"
        );
        // The input_schema should be the user's schema
        let input_schema = tools[0].get("input_schema");
        assert!(
            input_schema.is_some(),
            "Synthetic tool should have input_schema"
        );
        assert_eq!(
            input_schema
                .unwrap()
                .get("properties")
                .and_then(|p| p.get("answer")),
            Some(&serde_json::json!({"type": "integer"})),
            "Schema should contain the answer property"
        );

        // tool_choice should force this tool
        let tool_choice = body.get("tool_choice");
        assert!(tool_choice.is_some(), "tool_choice should be set");
        assert_eq!(
            tool_choice.unwrap().get("name").and_then(|v| v.as_str()),
            Some("structured_output"),
            "tool_choice should force the structured_output tool"
        );
    }

    #[test]
    fn test_anthropic_json_object_injects_instruction() {
        let req = Request::default()
            .model("claude-sonnet-4-20250514")
            .messages(vec![Message::user("List items")])
            .response_format(ResponseFormat {
                r#type: "json_object".to_string(),
                json_schema: None,
                strict: false,
            });
        let (body, _) = translate_request(&req);
        let system_text = body.get("system").and_then(|v| v.as_str()).unwrap_or("");
        assert!(
            system_text.contains("valid JSON"),
            "json_object mode should inject JSON instruction into system prompt"
        );
    }

    #[test]
    fn test_anthropic_structured_output_fallback_with_existing_tools() {
        // GAP-2: When tools are already present, tool-based extraction falls back
        // to system-prompt injection to avoid conflicts.
        let req = Request::default()
            .model("claude-sonnet-4-20250514")
            .messages(vec![
                Message::system("You are a math expert."),
                Message::user("Calculate"),
            ])
            .response_format(ResponseFormat {
                r#type: "json_schema".to_string(),
                json_schema: Some(serde_json::json!({"type": "object"})),
                strict: true,
            })
            .tools(vec![unified_llm_types::ToolDefinition {
                name: "calculator".to_string(),
                description: "Adds numbers".to_string(),
                parameters: serde_json::json!({"type": "object"}),
                strict: None,
            }]);
        let (body, _) = translate_request(&req);
        let system_text = body.get("system").and_then(|v| v.as_str()).unwrap_or("");
        assert!(
            system_text.starts_with("You are a math expert."),
            "Original system message should appear first"
        );
        assert!(
            system_text.contains("JSON Schema"),
            "Schema injection should be appended (fallback mode)"
        );
        // Should have the user's tool, NOT the synthetic json_output tool
        let tools = body.get("tools").and_then(|v| v.as_array()).unwrap();
        assert_eq!(tools.len(), 1, "Only the user's tool, no synthetic tool");
        assert_eq!(
            tools[0].get("name").and_then(|v| v.as_str()),
            Some("calculator"),
        );
    }

    // === AF-02: Reasoning Tokens Estimation Tests ===

    #[test]
    fn test_anthropic_reasoning_tokens_estimated_from_thinking() {
        // Mock response with a thinking block containing 400 chars of text.
        // Expected: reasoning_tokens = Some(400 / 4) = Some(100)
        let long_thinking = "a".repeat(400); // 400 chars
        let raw = serde_json::json!({
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "thinking",
                    "thinking": &long_thinking,
                    "signature": "sig_test"
                },
                {
                    "type": "text",
                    "text": "The answer is 42."
                }
            ],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 20, "output_tokens": 10}
        });
        let response = parse_response(raw, &reqwest::header::HeaderMap::new()).unwrap();
        assert_eq!(
            response.usage.reasoning_tokens,
            Some(100),
            "reasoning_tokens should be estimated from thinking block text length / 4"
        );
    }

    #[test]
    fn test_anthropic_reasoning_tokens_none_without_thinking() {
        let raw = serde_json::json!({
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello"}],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        });
        let response = parse_response(raw, &reqwest::header::HeaderMap::new()).unwrap();
        assert_eq!(
            response.usage.reasoning_tokens, None,
            "reasoning_tokens should be None when no thinking blocks present"
        );
    }

    #[tokio::test]
    async fn test_anthropic_complete_auth_error() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/v1/messages"))
            .respond_with(
                wiremock::ResponseTemplate::new(401).set_body_json(serde_json::json!({
                    "type": "error",
                    "error": {"type": "authentication_error", "message": "Invalid API key"}
                })),
            )
            .mount(&server)
            .await;

        let adapter = AnthropicAdapter::new_with_base_url(
            SecretString::from("bad-key".to_string()),
            server.uri(),
        );
        let req = Request::default()
            .model("test")
            .messages(vec![Message::user("Hi")]);
        let err = adapter.complete(req).await.unwrap_err();

        assert_eq!(err.kind, ErrorKind::Authentication);
        assert!(!err.retryable);
    }

    // === AF-06: Outbound thinking/redacted_thinking translation tests ===

    #[test]
    fn test_anthropic_outbound_thinking_block() {
        let parts = vec![ContentPart::Thinking {
            thinking: ThinkingData {
                text: "Let me reason about this...".to_string(),
                signature: Some("sig_abc123".to_string()),
                redacted: false,
                data: None,
            },
        }];
        let (translated, _) = translate_content_parts(&parts);
        assert_eq!(translated.len(), 1, "Thinking block should not be dropped");
        let block = &translated[0];
        assert_eq!(block["type"], "thinking");
        assert_eq!(block["thinking"], "Let me reason about this...");
        assert_eq!(block["signature"], "sig_abc123");
    }

    #[test]
    fn test_anthropic_outbound_redacted_thinking() {
        let parts = vec![ContentPart::RedactedThinking {
            thinking: ThinkingData {
                text: String::new(),
                signature: None,
                redacted: true,
                data: Some("opaque_blob_data".to_string()),
            },
        }];
        let (translated, _) = translate_content_parts(&parts);
        assert_eq!(
            translated.len(),
            1,
            "RedactedThinking block should not be dropped"
        );
        let block = &translated[0];
        assert_eq!(block["type"], "redacted_thinking");
        assert_eq!(block["data"], "opaque_blob_data");
    }

    #[test]
    fn test_anthropic_thinking_roundtrip_outbound() {
        let parts = vec![
            ContentPart::text("Before"),
            ContentPart::Thinking {
                thinking: ThinkingData {
                    text: "thinking...".to_string(),
                    signature: None,
                    redacted: false,
                    data: None,
                },
            },
            ContentPart::text("After"),
        ];
        let (translated, _) = translate_content_parts(&parts);
        assert_eq!(
            translated.len(),
            3,
            "All 3 parts (Text + Thinking + Text) should be present, got {}",
            translated.len()
        );
    }

    // === AF-08: Streaming signature capture + redacted thinking tests ===

    #[test]
    fn test_anthropic_stream_thinking_captures_signature() {
        // Simulate the real Anthropic SSE event sequence for a thinking block:
        // 1. content_block_start (thinking)
        // 2. content_block_delta (thinking_delta)
        // 3. content_block_delta (signature_delta) — signature arrives here
        // 4. content_block_stop — bare event, no signature data
        let mut translator = StreamTranslator::new();

        // 1. content_block_start for thinking
        let _events = translator.process(
            "content_block_start",
            &serde_json::json!({
                "index": 0,
                "content_block": {"type": "thinking", "thinking": ""}
            }),
        );

        // 2. content_block_delta with thinking text
        let _events = translator.process(
            "content_block_delta",
            &serde_json::json!({
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "Let me think..."}
            }),
        );

        // 3. content_block_delta with signature_delta — real API sends signature here
        let _events = translator.process(
            "content_block_delta",
            &serde_json::json!({
                "index": 0,
                "delta": {"type": "signature_delta", "signature": "sig_abc123"}
            }),
        );

        // 4. content_block_stop — bare event, no signature data (real API behavior)
        let events = translator.process(
            "content_block_stop",
            &serde_json::json!({
                "index": 0
            }),
        );

        // The ReasoningEnd event should carry the signature in its raw field
        let reasoning_end = events
            .iter()
            .find(|e| e.event_type == StreamEventType::ReasoningEnd);
        assert!(reasoning_end.is_some(), "Should have a ReasoningEnd event");
        let reasoning_end = reasoning_end.unwrap();
        let sig = reasoning_end
            .raw
            .as_ref()
            .and_then(|d| d.get("signature"))
            .and_then(|v| v.as_str());
        assert_eq!(
            sig,
            Some("sig_abc123"),
            "ReasoningEnd should carry signature 'sig_abc123' in its raw field. Got: {:?}",
            reasoning_end.raw
        );
    }

    #[test]
    fn test_anthropic_stream_signature_delta_captured() {
        // Dedicated test: signature_delta events must be accumulated and
        // attached to the ReasoningEnd event. Tests multiple fragments.
        let mut translator = StreamTranslator::new();

        // content_block_start for thinking
        let _events = translator.process(
            "content_block_start",
            &serde_json::json!({
                "index": 0,
                "content_block": {"type": "thinking", "thinking": ""}
            }),
        );

        // thinking_delta
        let _events = translator.process(
            "content_block_delta",
            &serde_json::json!({
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "reasoning..."}
            }),
        );

        // signature_delta fragment 1
        let _events = translator.process(
            "content_block_delta",
            &serde_json::json!({
                "index": 0,
                "delta": {"type": "signature_delta", "signature": "sig_part1"}
            }),
        );

        // signature_delta fragment 2
        let _events = translator.process(
            "content_block_delta",
            &serde_json::json!({
                "index": 0,
                "delta": {"type": "signature_delta", "signature": "_part2"}
            }),
        );

        // content_block_stop (bare, no signature)
        let events = translator.process("content_block_stop", &serde_json::json!({"index": 0}));

        let reasoning_end = events
            .iter()
            .find(|e| e.event_type == StreamEventType::ReasoningEnd)
            .expect("Should have ReasoningEnd");

        let sig = reasoning_end
            .raw
            .as_ref()
            .and_then(|d| d.get("signature"))
            .and_then(|v| v.as_str());
        assert_eq!(
            sig,
            Some("sig_part1_part2"),
            "Accumulated signature should be 'sig_part1_part2'. Got: {:?}",
            reasoning_end.raw
        );
    }

    #[test]
    fn test_anthropic_stream_redacted_thinking_not_dropped() {
        // Simulate the SSE event sequence for a redacted_thinking block.
        // It should emit ReasoningStart + ReasoningEnd (not silently swallowed).
        let mut translator = StreamTranslator::new();

        // content_block_start for redacted_thinking
        let events = translator.process(
            "content_block_start",
            &serde_json::json!({
                "index": 0,
                "content_block": {"type": "redacted_thinking", "data": "opaque_blob"}
            }),
        );

        let event_types: Vec<_> = events.iter().map(|e| &e.event_type).collect();

        // Verify ReasoningStart and ReasoningEnd events are emitted
        assert!(
            event_types.contains(&&StreamEventType::ReasoningStart),
            "redacted_thinking should emit ReasoningStart. Got event types: {:?}",
            event_types
        );
        assert!(
            event_types.contains(&&StreamEventType::ReasoningEnd),
            "redacted_thinking should emit ReasoningEnd. Got event types: {:?}",
            event_types
        );

        // The ReasoningEnd event for a redacted block should carry redacted: true
        let redacted_end = events.iter().find(|e| {
            e.event_type == StreamEventType::ReasoningEnd
                && e.raw
                    .as_ref()
                    .and_then(|d| d.get("redacted"))
                    .and_then(|v| v.as_bool())
                    == Some(true)
        });
        assert!(
            redacted_end.is_some(),
            "ReasoningEnd for redacted_thinking should carry {{\"redacted\": true}} in raw"
        );
    }

    // === FP-15: reasoning_effort → Anthropic thinking config ===

    #[test]
    fn test_anthropic_reasoning_effort_maps_to_thinking_config() {
        let request = Request::default()
            .model("claude-opus-4-6")
            .messages(vec![Message::user("think hard")])
            .reasoning_effort("high");
        let (body, _) = translate_request(&request);
        assert!(
            body.get("thinking").is_some(),
            "should have thinking config"
        );
        assert_eq!(body["thinking"]["type"], "enabled");
        assert_eq!(body["thinking"]["budget_tokens"], 10000);
    }

    #[test]
    fn test_anthropic_reasoning_effort_low() {
        let request = Request::default()
            .model("claude-opus-4-6")
            .messages(vec![Message::user("quick")])
            .reasoning_effort("low");
        let (body, _) = translate_request(&request);
        assert_eq!(body["thinking"]["budget_tokens"], 1024);
    }

    #[test]
    fn test_anthropic_reasoning_effort_medium() {
        let request = Request::default()
            .model("claude-opus-4-6")
            .messages(vec![Message::user("think")])
            .reasoning_effort("medium");
        let (body, _) = translate_request(&request);
        assert_eq!(body["thinking"]["budget_tokens"], 4096);
    }

    #[test]
    fn test_anthropic_reasoning_effort_does_not_override_explicit_thinking() {
        let request = Request::default()
            .model("claude-opus-4-6")
            .messages(vec![Message::user("think")])
            .reasoning_effort("high")
            .provider_options(Some(serde_json::json!({
                "anthropic": {
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": 2048
                    }
                }
            })));
        let (body, _) = translate_request(&request);
        // Explicit provider_options.anthropic.thinking should be preserved
        assert_eq!(body["thinking"]["budget_tokens"], 2048);
    }

    // === C-2: AdapterTimeout wiring ===

    #[tokio::test]
    async fn test_connect_timeout_is_wired() {
        use std::time::{Duration, Instant};

        // 10.255.255.1 is a non-routable address — connection will hang until timeout
        let adapter = AnthropicAdapter::new_with_base_url_and_timeout(
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

    #[test]
    fn test_unknown_sse_event_forwarded_as_provider_event() {
        let mut translator = StreamTranslator::new();
        // First send message_start to initialize
        let start_data = serde_json::json!({
            "message": {"id": "msg_1", "model": "test", "usage": {"input_tokens": 0}}
        });
        translator.process("message_start", &start_data);

        // Send an unknown event type
        let unknown_data = serde_json::json!({"custom_field": "custom_value"});
        let events = translator.process("some_future_event", &unknown_data);

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, StreamEventType::ProviderEvent);
        let raw = events[0].raw.as_ref().unwrap();
        assert_eq!(raw["sse_event_type"], "some_future_event");
        assert_eq!(raw["data"]["custom_field"], "custom_value");
    }

    // ===== ToolChoice Translation Tests (8.7-9) =====

    fn make_test_tool() -> ToolDefinition {
        ToolDefinition {
            name: "get_weather".into(),
            description: "Get weather".into(),
            parameters: serde_json::json!({"type": "object", "properties": {"city": {"type": "string"}}}),
            strict: None,
        }
    }

    #[test]
    fn test_anthropic_tool_choice_auto() {
        let request = Request::default()
            .model("claude-opus-4-6")
            .messages(vec![Message::user("hi")])
            .tools(vec![make_test_tool()])
            .tool_choice(ToolChoice {
                mode: "auto".into(),
                tool_name: None,
            });
        let (body, _) = translate_request(&request);
        assert_eq!(body["tool_choice"]["type"], "auto");
    }

    #[test]
    fn test_anthropic_tool_choice_none() {
        let request = Request::default()
            .model("claude-opus-4-6")
            .messages(vec![Message::user("hi")])
            .tools(vec![make_test_tool()])
            .tool_choice(ToolChoice {
                mode: "none".into(),
                tool_name: None,
            });
        let (body, _) = translate_request(&request);
        // Anthropic workaround: tools array is removed entirely when tool_choice is "none"
        assert!(
            body.get("tools").is_none(),
            "tools should be removed for tool_choice=none"
        );
        assert!(
            body.get("tool_choice").is_none(),
            "tool_choice should not be set"
        );
    }

    #[test]
    fn test_anthropic_tool_choice_required() {
        let request = Request::default()
            .model("claude-opus-4-6")
            .messages(vec![Message::user("hi")])
            .tools(vec![make_test_tool()])
            .tool_choice(ToolChoice {
                mode: "required".into(),
                tool_name: None,
            });
        let (body, _) = translate_request(&request);
        // Anthropic maps "required" → "any"
        assert_eq!(body["tool_choice"]["type"], "any");
    }

    #[test]
    fn test_anthropic_tool_choice_named() {
        let request = Request::default()
            .model("claude-opus-4-6")
            .messages(vec![Message::user("hi")])
            .tools(vec![make_test_tool()])
            .tool_choice(ToolChoice {
                mode: "named".into(),
                tool_name: Some("get_weather".into()),
            });
        let (body, _) = translate_request(&request);
        assert_eq!(body["tool_choice"]["type"], "tool");
        assert_eq!(body["tool_choice"]["name"], "get_weather");
    }

    // ===== C-2: stream_read timeout stored and wired =====

    #[test]
    fn test_stream_read_timeout_stored_on_adapter() {
        let adapter = AnthropicAdapterBuilder::new(SecretString::from("key".to_string()))
            .timeout(AdapterTimeout {
                connect: 5.0,
                request: 60.0,
                stream_read: 42.0,
            })
            .build();
        assert_eq!(
            adapter.stream_read_timeout,
            std::time::Duration::from_secs_f64(42.0),
            "stream_read_timeout should be wired from AdapterTimeout.stream_read"
        );
    }

    #[test]
    fn test_stream_read_timeout_default() {
        // Default AdapterTimeout has stream_read = 30.0
        let adapter = AnthropicAdapter::new(SecretString::from("key".to_string()));
        assert_eq!(
            adapter.stream_read_timeout,
            std::time::Duration::from_secs_f64(AdapterTimeout::default().stream_read),
            "default stream_read_timeout should match AdapterTimeout default"
        );
    }
}
