// StreamAccumulator — accumulates StreamEvents into a complete Response.

use unified_llm_types::content::{ArgumentValue, ContentPart, ThinkingData, ToolCallData};
use unified_llm_types::message::Message;
use unified_llm_types::response::{FinishReason, Response, Usage};
use unified_llm_types::stream::{StreamEvent, StreamEventType};

/// A single accumulated reasoning/thinking block from the stream.
#[derive(Debug, Clone)]
struct ReasoningBlock {
    text_parts: Vec<String>,
    signature: Option<String>,
    redacted: bool,
    opaque_data: Option<String>,
}

/// Accumulates stream events into a complete Response.
///
/// Process events via `process()`, then call `response()` to get the
/// assembled Response (returns None if no Finish event has been received).
pub struct StreamAccumulator {
    id: String,
    model: String,
    provider: String,
    text_parts: Vec<String>,
    /// Completed reasoning blocks.
    reasoning_blocks: Vec<ReasoningBlock>,
    /// In-progress reasoning block's text parts (between ReasoningStart and ReasoningEnd).
    current_reasoning_parts: Vec<String>,
    /// In-progress reasoning block's signature (set by ReasoningEnd before finalization).
    current_reasoning_signature: Option<String>,
    /// Whether the current in-progress reasoning block is redacted.
    current_reasoning_redacted: bool,
    /// Opaque data for the current redacted reasoning block.
    current_reasoning_data: Option<String>,
    tool_calls: Vec<ToolCallData>,
    /// Buffer for the current in-progress tool call's arguments.
    current_tool_id: Option<String>,
    current_tool_name: Option<String>,
    current_tool_args: String,
    finish_reason: Option<FinishReason>,
    usage: Option<Usage>,
    finished: bool,
    lifecycle: crate::util::stream_lifecycle::StreamLifecycle,
}

impl Default for StreamAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamAccumulator {
    pub fn new() -> Self {
        Self {
            id: String::new(),
            model: String::new(),
            provider: String::new(),
            text_parts: Vec::new(),
            reasoning_blocks: Vec::new(),
            current_reasoning_parts: Vec::new(),
            current_reasoning_signature: None,
            current_reasoning_redacted: false,
            current_reasoning_data: None,
            tool_calls: Vec::new(),
            current_tool_id: None,
            current_tool_name: None,
            current_tool_args: String::new(),
            finish_reason: None,
            usage: None,
            finished: false,
            lifecycle: crate::util::stream_lifecycle::StreamLifecycle::new(),
        }
    }

    /// Process a single stream event, accumulating its data.
    pub fn process(&mut self, event: &StreamEvent) {
        self.lifecycle.observe(&event.event_type);
        match event.event_type {
            StreamEventType::StreamStart => {
                if let Some(id) = &event.id {
                    self.id = id.clone();
                }
            }
            StreamEventType::TextDelta => {
                if let Some(delta) = &event.delta {
                    self.text_parts.push(delta.clone());
                }
            }
            StreamEventType::ReasoningDelta => {
                // Prefer the dedicated reasoning_delta field; fall back to delta
                // for backward compatibility with older providers.
                let text = event.reasoning_delta.as_ref().or(event.delta.as_ref());
                if let Some(delta) = text {
                    self.current_reasoning_parts.push(delta.clone());
                }
            }
            StreamEventType::ToolCallStart => {
                if let Some(tc) = &event.tool_call {
                    self.current_tool_id = Some(tc.id.clone());
                    self.current_tool_name = Some(tc.name.clone());
                }
                self.current_tool_args.clear();
            }
            StreamEventType::ToolCallDelta => {
                if let Some(delta) = &event.delta {
                    self.current_tool_args.push_str(delta);
                }
            }
            StreamEventType::ToolCallEnd => {
                // If the event carries a complete tool_call, prefer its fields
                // over the manually accumulated state.
                if let Some(tc) = &event.tool_call {
                    let id = if !tc.id.is_empty() {
                        tc.id.clone()
                    } else {
                        self.current_tool_id.take().unwrap_or_default()
                    };
                    let name = if !tc.name.is_empty() {
                        tc.name.clone()
                    } else {
                        self.current_tool_name.take().unwrap_or_default()
                    };
                    let arguments = if !tc.arguments.is_empty() {
                        ArgumentValue::Dict(tc.arguments.clone())
                    } else {
                        // Parsed map is empty — try raw_arguments, then accumulated deltas
                        let raw = tc
                            .raw_arguments
                            .clone()
                            .unwrap_or_else(|| std::mem::take(&mut self.current_tool_args));
                        match serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(
                            &raw,
                        ) {
                            Ok(map) => ArgumentValue::Dict(map),
                            Err(_) => ArgumentValue::Raw(raw),
                        }
                    };

                    self.current_tool_id = None;
                    self.current_tool_name = None;
                    self.current_tool_args.clear();

                    self.tool_calls.push(ToolCallData {
                        id,
                        name,
                        arguments,
                        r#type: "function".into(),
                    });
                } else {
                    // Legacy path: assemble entirely from accumulated deltas
                    let id = self.current_tool_id.take().unwrap_or_default();
                    let name = self.current_tool_name.take().unwrap_or_default();
                    let args_str = std::mem::take(&mut self.current_tool_args);

                    let arguments = match serde_json::from_str::<
                        serde_json::Map<String, serde_json::Value>,
                    >(&args_str)
                    {
                        Ok(map) => ArgumentValue::Dict(map),
                        Err(_) => ArgumentValue::Raw(args_str),
                    };

                    self.tool_calls.push(ToolCallData {
                        id,
                        name,
                        arguments,
                        r#type: "function".into(),
                    });
                }
            }
            StreamEventType::Finish => {
                self.finish_reason = event.finish_reason.clone();
                self.usage = event.usage.clone();
                // If a pre-built response is attached, pull model/provider
                // metadata that isn't available on other event types.
                if let Some(resp) = &event.response {
                    if self.model.is_empty() && !resp.model.is_empty() {
                        self.model = resp.model.clone();
                    }
                    if self.provider.is_empty() && !resp.provider.is_empty() {
                        self.provider = resp.provider.clone();
                    }
                }
                self.finished = true;
            }
            StreamEventType::ReasoningEnd => {
                // Capture signature, redacted flag, and opaque data from raw field
                // (Anthropic sends {"signature": "...", "redacted": true, "data": "..."})
                if let Some(raw) = &event.raw {
                    if let Some(sig) = raw.get("signature").and_then(|v| v.as_str()) {
                        self.current_reasoning_signature = Some(sig.to_string());
                    }
                    // CRITICAL-2: Extract redacted flag and opaque data payload
                    if raw.get("redacted").and_then(|v| v.as_bool()) == Some(true) {
                        self.current_reasoning_redacted = true;
                    }
                    if let Some(data) = raw.get("data").and_then(|v| v.as_str()) {
                        self.current_reasoning_data = Some(data.to_string());
                    }
                }
                // Finalize the current reasoning block
                if !self.current_reasoning_parts.is_empty()
                    || self.current_reasoning_signature.is_some()
                    || self.current_reasoning_redacted
                {
                    self.reasoning_blocks.push(ReasoningBlock {
                        text_parts: std::mem::take(&mut self.current_reasoning_parts),
                        signature: self.current_reasoning_signature.take(),
                        redacted: self.current_reasoning_redacted,
                        opaque_data: self.current_reasoning_data.take(),
                    });
                    self.current_reasoning_redacted = false;
                }
            }
            StreamEventType::ReasoningStart => {
                // If there's an in-progress block that was never finalized by
                // ReasoningEnd, push it before starting a new one.
                if !self.current_reasoning_parts.is_empty() || self.current_reasoning_redacted {
                    self.reasoning_blocks.push(ReasoningBlock {
                        text_parts: std::mem::take(&mut self.current_reasoning_parts),
                        signature: self.current_reasoning_signature.take(),
                        redacted: self.current_reasoning_redacted,
                        opaque_data: self.current_reasoning_data.take(),
                    });
                }
                self.current_reasoning_parts.clear();
                self.current_reasoning_signature = None;
                self.current_reasoning_redacted = false;
                self.current_reasoning_data = None;
            }
            // TextStart, TextEnd, Error, ProviderEvent
            // are acknowledged but don't require accumulation logic
            _ => {}
        }
    }

    /// Reset the accumulator for a new tool-loop step, clearing all accumulated
    /// data but preserving identity fields (id, model, provider).
    ///
    /// Call this between tool-loop iterations when reusing a single accumulator
    /// across multiple streaming steps.
    pub fn reset(&mut self) {
        self.text_parts.clear();
        self.reasoning_blocks.clear();
        self.current_reasoning_parts.clear();
        self.current_reasoning_signature = None;
        self.current_reasoning_redacted = false;
        self.current_reasoning_data = None;
        self.tool_calls.clear();
        self.current_tool_id = None;
        self.current_tool_name = None;
        self.current_tool_args.clear();
        self.finish_reason = None;
        self.usage = None;
        self.finished = false;
        self.lifecycle.reset();
    }

    /// Returns the accumulated Response, or None if no Finish event has been processed.
    pub fn response(&self) -> Option<Response> {
        if !self.finished {
            return None;
        }
        self.build_response()
    }

    /// Returns the response accumulated so far, even if the stream hasn't finished.
    /// Returns None if no events have been processed yet (nothing accumulated).
    pub fn current_response(&self) -> Option<Response> {
        if self.text_parts.is_empty()
            && self.reasoning_blocks.is_empty()
            && self.current_reasoning_parts.is_empty()
            && self.tool_calls.is_empty()
            && self.id.is_empty()
        {
            return None;
        }
        self.build_response()
    }

    /// Shared helper that builds a Response from the current accumulated state.
    fn build_response(&self) -> Option<Response> {
        // Build content parts
        let mut content: Vec<ContentPart> = Vec::new();

        // CRITICAL-2: Emit ContentPart::Thinking or ContentPart::RedactedThinking
        // per completed reasoning block, based on the redacted flag.
        for block in &self.reasoning_blocks {
            if block.redacted {
                content.push(ContentPart::RedactedThinking {
                    thinking: ThinkingData {
                        text: block.text_parts.join(""),
                        signature: block.signature.clone(),
                        redacted: true,
                        data: block.opaque_data.clone(),
                    },
                });
            } else {
                content.push(ContentPart::Thinking {
                    thinking: ThinkingData {
                        text: block.text_parts.join(""),
                        signature: block.signature.clone(),
                        redacted: false,
                        data: None,
                    },
                });
            }
        }

        // Finalize any in-progress reasoning block
        if !self.current_reasoning_parts.is_empty() || self.current_reasoning_redacted {
            if self.current_reasoning_redacted {
                content.push(ContentPart::RedactedThinking {
                    thinking: ThinkingData {
                        text: self.current_reasoning_parts.join(""),
                        signature: self.current_reasoning_signature.clone(),
                        redacted: true,
                        data: self.current_reasoning_data.clone(),
                    },
                });
            } else {
                content.push(ContentPart::Thinking {
                    thinking: ThinkingData {
                        text: self.current_reasoning_parts.join(""),
                        signature: self.current_reasoning_signature.clone(),
                        redacted: false,
                        data: None,
                    },
                });
            }
        }

        // Add text content
        let text = self.text_parts.join("");
        if !text.is_empty() {
            content.push(ContentPart::text(text));
        }

        // Add tool calls
        for tc in &self.tool_calls {
            content.push(ContentPart::ToolCall {
                tool_call: tc.clone(),
            });
        }

        // If no content at all, add empty text
        if content.is_empty() {
            content.push(ContentPart::text(""));
        }

        let message = Message {
            role: unified_llm_types::message::Role::Assistant,
            content,
            name: None,
            tool_call_id: None,
        };

        Some(Response {
            id: self.id.clone(),
            model: self.model.clone(),
            provider: self.provider.clone(),
            message,
            finish_reason: self
                .finish_reason
                .clone()
                .unwrap_or_else(FinishReason::stop),
            usage: self.usage.clone().unwrap_or_default(),
            raw: None,
            warnings: vec![],
            rate_limit: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_stream_returns_none() {
        let acc = StreamAccumulator::new();
        assert!(acc.response().is_none());
    }

    #[test]
    fn test_accumulates_text_deltas() {
        let mut acc = StreamAccumulator::new();
        acc.process(&StreamEvent {
            event_type: StreamEventType::StreamStart,
            ..Default::default()
        });
        acc.process(&StreamEvent {
            event_type: StreamEventType::TextDelta,
            delta: Some("Hello ".into()),
            ..Default::default()
        });
        acc.process(&StreamEvent {
            event_type: StreamEventType::TextDelta,
            delta: Some("world".into()),
            ..Default::default()
        });
        acc.process(&StreamEvent {
            event_type: StreamEventType::Finish,
            finish_reason: Some(FinishReason::stop()),
            usage: Some(Usage {
                input_tokens: 10,
                output_tokens: 5,
                total_tokens: 15,
                ..Default::default()
            }),
            ..Default::default()
        });
        let resp = acc.response().expect("should have response after Finish");
        assert_eq!(resp.text(), "Hello world");
        assert_eq!(resp.finish_reason.reason, "stop");
        assert_eq!(resp.usage.total_tokens, 15);
    }

    #[test]
    fn test_accumulates_reasoning_deltas() {
        let mut acc = StreamAccumulator::new();
        acc.process(&StreamEvent {
            event_type: StreamEventType::ReasoningDelta,
            delta: Some("Let me ".into()),
            ..Default::default()
        });
        acc.process(&StreamEvent {
            event_type: StreamEventType::ReasoningDelta,
            delta: Some("think...".into()),
            ..Default::default()
        });
        acc.process(&StreamEvent {
            event_type: StreamEventType::TextDelta,
            delta: Some("42".into()),
            ..Default::default()
        });
        acc.process(&StreamEvent {
            event_type: StreamEventType::Finish,
            finish_reason: Some(FinishReason::stop()),
            usage: Some(Usage::default()),
            ..Default::default()
        });
        let resp = acc.response().unwrap();
        assert_eq!(resp.reasoning(), Some("Let me think...".to_string()));
        assert_eq!(resp.text(), "42");
    }

    #[test]
    fn test_accumulates_tool_calls() {
        let mut acc = StreamAccumulator::new();
        acc.process(&StreamEvent {
            event_type: StreamEventType::ToolCallStart,
            tool_call: Some(unified_llm_types::tool::ToolCall {
                id: "call_1".into(),
                name: "get_weather".into(),
                arguments: serde_json::Map::new(),
                raw_arguments: None,
            }),
            ..Default::default()
        });
        acc.process(&StreamEvent {
            event_type: StreamEventType::ToolCallDelta,
            delta: Some("{\"city\":".into()),
            ..Default::default()
        });
        acc.process(&StreamEvent {
            event_type: StreamEventType::ToolCallDelta,
            delta: Some("\"SF\"}".into()),
            ..Default::default()
        });
        acc.process(&StreamEvent {
            event_type: StreamEventType::ToolCallEnd,
            ..Default::default()
        });
        acc.process(&StreamEvent {
            event_type: StreamEventType::Finish,
            finish_reason: Some(FinishReason::tool_calls()),
            usage: Some(Usage::default()),
            ..Default::default()
        });
        let resp = acc.response().unwrap();
        let calls = resp.tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].name, "get_weather");
    }

    #[test]
    fn test_accumulates_multiple_tool_calls() {
        let mut acc = StreamAccumulator::new();
        // First tool call
        acc.process(&StreamEvent {
            event_type: StreamEventType::ToolCallStart,
            tool_call: Some(unified_llm_types::tool::ToolCall {
                id: "call_1".into(),
                name: "get_weather".into(),
                arguments: serde_json::Map::new(),
                raw_arguments: None,
            }),
            ..Default::default()
        });
        acc.process(&StreamEvent {
            event_type: StreamEventType::ToolCallDelta,
            delta: Some("{\"city\":\"SF\"}".into()),
            ..Default::default()
        });
        acc.process(&StreamEvent {
            event_type: StreamEventType::ToolCallEnd,
            ..Default::default()
        });
        // Second tool call
        acc.process(&StreamEvent {
            event_type: StreamEventType::ToolCallStart,
            tool_call: Some(unified_llm_types::tool::ToolCall {
                id: "call_2".into(),
                name: "get_time".into(),
                arguments: serde_json::Map::new(),
                raw_arguments: None,
            }),
            ..Default::default()
        });
        acc.process(&StreamEvent {
            event_type: StreamEventType::ToolCallDelta,
            delta: Some("{\"tz\":\"PST\"}".into()),
            ..Default::default()
        });
        acc.process(&StreamEvent {
            event_type: StreamEventType::ToolCallEnd,
            ..Default::default()
        });
        acc.process(&StreamEvent {
            event_type: StreamEventType::Finish,
            finish_reason: Some(FinishReason::tool_calls()),
            usage: Some(Usage::default()),
            ..Default::default()
        });
        let resp = acc.response().unwrap();
        assert_eq!(resp.tool_calls().len(), 2);
    }

    #[test]
    fn test_finish_captures_usage() {
        let mut acc = StreamAccumulator::new();
        acc.process(&StreamEvent {
            event_type: StreamEventType::TextDelta,
            delta: Some("Hi".into()),
            ..Default::default()
        });
        acc.process(&StreamEvent {
            event_type: StreamEventType::Finish,
            finish_reason: Some(FinishReason::length()),
            usage: Some(Usage {
                input_tokens: 100,
                output_tokens: 50,
                total_tokens: 150,
                reasoning_tokens: Some(10),
                ..Default::default()
            }),
            ..Default::default()
        });
        let resp = acc.response().unwrap();
        assert_eq!(resp.finish_reason.reason, "length");
        assert_eq!(resp.usage.input_tokens, 100);
        assert_eq!(resp.usage.output_tokens, 50);
        assert_eq!(resp.usage.reasoning_tokens, Some(10));
    }

    #[test]
    fn test_stream_start_captures_id() {
        let mut acc = StreamAccumulator::new();
        acc.process(&StreamEvent {
            event_type: StreamEventType::StreamStart,
            id: Some("resp_abc".into()),
            ..Default::default()
        });
        acc.process(&StreamEvent {
            event_type: StreamEventType::TextDelta,
            delta: Some("Hi".into()),
            ..Default::default()
        });
        acc.process(&StreamEvent {
            event_type: StreamEventType::Finish,
            finish_reason: Some(FinishReason::stop()),
            usage: Some(Usage::default()),
            ..Default::default()
        });
        let resp = acc.response().unwrap();
        assert_eq!(resp.id, "resp_abc");
    }

    // --- Tests for new spec fields ---

    #[test]
    fn test_reasoning_delta_field_preferred_over_delta() {
        let mut acc = StreamAccumulator::new();
        // Uses the new dedicated reasoning_delta field
        acc.process(&StreamEvent {
            event_type: StreamEventType::ReasoningDelta,
            reasoning_delta: Some("thinking hard".into()),
            delta: Some("should be ignored".into()),
            ..Default::default()
        });
        acc.process(&StreamEvent {
            event_type: StreamEventType::Finish,
            finish_reason: Some(FinishReason::stop()),
            usage: Some(Usage::default()),
            ..Default::default()
        });
        let resp = acc.response().unwrap();
        assert_eq!(resp.reasoning(), Some("thinking hard".to_string()));
    }

    #[test]
    fn test_reasoning_delta_falls_back_to_delta() {
        let mut acc = StreamAccumulator::new();
        // Only delta set (legacy provider behavior)
        acc.process(&StreamEvent {
            event_type: StreamEventType::ReasoningDelta,
            delta: Some("fallback reasoning".into()),
            ..Default::default()
        });
        acc.process(&StreamEvent {
            event_type: StreamEventType::Finish,
            finish_reason: Some(FinishReason::stop()),
            usage: Some(Usage::default()),
            ..Default::default()
        });
        let resp = acc.response().unwrap();
        assert_eq!(resp.reasoning(), Some("fallback reasoning".to_string()));
    }

    #[test]
    fn test_tool_call_start_from_tool_call_field() {
        use unified_llm_types::tool::ToolCall;

        let mut acc = StreamAccumulator::new();
        acc.process(&StreamEvent {
            event_type: StreamEventType::ToolCallStart,
            tool_call: Some(ToolCall {
                id: "call_new".into(),
                name: "search".into(),
                arguments: serde_json::Map::new(),
                raw_arguments: None,
            }),
            ..Default::default()
        });
        acc.process(&StreamEvent {
            event_type: StreamEventType::ToolCallDelta,
            delta: Some("{\"q\":\"rust\"}".into()),
            ..Default::default()
        });
        acc.process(&StreamEvent {
            event_type: StreamEventType::ToolCallEnd,
            ..Default::default()
        });
        acc.process(&StreamEvent {
            event_type: StreamEventType::Finish,
            finish_reason: Some(FinishReason::tool_calls()),
            usage: Some(Usage::default()),
            ..Default::default()
        });
        let resp = acc.response().unwrap();
        let calls = resp.tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_new");
        assert_eq!(calls[0].name, "search");
    }

    #[test]
    fn test_tool_call_end_with_complete_tool_call() {
        use unified_llm_types::tool::ToolCall;

        let mut acc = StreamAccumulator::new();
        // Start with tool_call field
        acc.process(&StreamEvent {
            event_type: StreamEventType::ToolCallStart,
            tool_call: Some(unified_llm_types::tool::ToolCall {
                id: "call_x".into(),
                name: "calc".into(),
                arguments: serde_json::Map::new(),
                raw_arguments: None,
            }),
            ..Default::default()
        });
        // End carries the complete parsed tool call
        let mut args = serde_json::Map::new();
        args.insert("expr".into(), serde_json::json!("2+2"));
        acc.process(&StreamEvent {
            event_type: StreamEventType::ToolCallEnd,
            tool_call: Some(ToolCall {
                id: "call_x".into(),
                name: "calc".into(),
                arguments: args,
                raw_arguments: None,
            }),
            ..Default::default()
        });
        acc.process(&StreamEvent {
            event_type: StreamEventType::Finish,
            finish_reason: Some(FinishReason::tool_calls()),
            usage: Some(Usage::default()),
            ..Default::default()
        });
        let resp = acc.response().unwrap();
        let calls = resp.tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_x");
        assert_eq!(calls[0].name, "calc");
        // ToolCall.arguments is already a parsed Map<String, Value>
        assert_eq!(
            calls[0].arguments.get("expr").unwrap(),
            &serde_json::json!("2+2")
        );
    }

    #[test]
    fn test_accumulator_reset_clears_between_steps() {
        let mut acc = StreamAccumulator::new();

        // Step 1: accumulate "Step1"
        acc.process(&StreamEvent {
            event_type: StreamEventType::StreamStart,
            id: Some("resp_1".into()),
            ..Default::default()
        });
        acc.process(&StreamEvent {
            event_type: StreamEventType::TextDelta,
            delta: Some("Step1".into()),
            ..Default::default()
        });
        acc.process(&StreamEvent {
            event_type: StreamEventType::Finish,
            finish_reason: Some(FinishReason::tool_calls()),
            usage: Some(Usage {
                input_tokens: 10,
                output_tokens: 5,
                total_tokens: 15,
                ..Default::default()
            }),
            ..Default::default()
        });
        let step1_resp = acc.response().unwrap();
        assert_eq!(step1_resp.text(), "Step1");
        assert_eq!(step1_resp.usage.input_tokens, 10);
        assert_eq!(step1_resp.id, "resp_1");

        // Reset for step 2
        acc.reset();

        // Step 2: accumulate "Step2"
        acc.process(&StreamEvent {
            event_type: StreamEventType::TextDelta,
            delta: Some("Step2".into()),
            ..Default::default()
        });
        acc.process(&StreamEvent {
            event_type: StreamEventType::Finish,
            finish_reason: Some(FinishReason::stop()),
            usage: Some(Usage {
                input_tokens: 20,
                output_tokens: 8,
                total_tokens: 28,
                ..Default::default()
            }),
            ..Default::default()
        });
        let step2_resp = acc.response().unwrap();

        // Step2 text should NOT contain Step1 text
        assert_eq!(
            step2_resp.text(),
            "Step2",
            "reset() should clear text_parts"
        );
        assert_eq!(
            step2_resp.usage.input_tokens, 20,
            "reset() should clear usage"
        );
        assert_eq!(
            step2_resp.finish_reason.reason, "stop",
            "reset() should clear finish_reason"
        );
        // Identity fields should be preserved
        assert_eq!(step2_resp.id, "resp_1", "reset() should preserve id");
    }

    #[test]
    fn test_accumulator_reset_clears_tool_calls() {
        let mut acc = StreamAccumulator::new();

        // Accumulate a tool call
        acc.process(&StreamEvent {
            event_type: StreamEventType::ToolCallStart,
            tool_call: Some(unified_llm_types::tool::ToolCall {
                id: "call_1".into(),
                name: "search".into(),
                arguments: serde_json::Map::new(),
                raw_arguments: None,
            }),
            ..Default::default()
        });
        acc.process(&StreamEvent {
            event_type: StreamEventType::ToolCallDelta,
            delta: Some("{\"q\":\"rust\"}".into()),
            ..Default::default()
        });
        acc.process(&StreamEvent {
            event_type: StreamEventType::ToolCallEnd,
            ..Default::default()
        });
        acc.process(&StreamEvent {
            event_type: StreamEventType::Finish,
            finish_reason: Some(FinishReason::tool_calls()),
            usage: Some(Usage::default()),
            ..Default::default()
        });
        assert_eq!(acc.response().unwrap().tool_calls().len(), 1);

        // Reset
        acc.reset();

        // Accumulate text only
        acc.process(&StreamEvent {
            event_type: StreamEventType::TextDelta,
            delta: Some("Done".into()),
            ..Default::default()
        });
        acc.process(&StreamEvent {
            event_type: StreamEventType::Finish,
            finish_reason: Some(FinishReason::stop()),
            usage: Some(Usage::default()),
            ..Default::default()
        });
        let resp = acc.response().unwrap();
        assert_eq!(resp.text(), "Done");
        assert!(
            resp.tool_calls().is_empty(),
            "reset() should clear tool_calls"
        );
    }

    #[test]
    fn test_thinking_signature_survives_streaming_accumulation() {
        let mut acc = StreamAccumulator::new();
        acc.process(&StreamEvent {
            event_type: StreamEventType::ReasoningStart,
            ..Default::default()
        });
        acc.process(&StreamEvent {
            event_type: StreamEventType::ReasoningDelta,
            reasoning_delta: Some("Let me think...".into()),
            ..Default::default()
        });
        acc.process(&StreamEvent {
            event_type: StreamEventType::ReasoningEnd,
            raw: Some(serde_json::json!({"signature": "sig_abc123"})),
            ..Default::default()
        });
        acc.process(&StreamEvent {
            event_type: StreamEventType::TextDelta,
            delta: Some("42".into()),
            ..Default::default()
        });
        acc.process(&StreamEvent {
            event_type: StreamEventType::Finish,
            finish_reason: Some(FinishReason::stop()),
            usage: Some(Usage::default()),
            ..Default::default()
        });
        let resp = acc.response().unwrap();
        assert_eq!(resp.reasoning(), Some("Let me think...".to_string()));
        // The signature must survive streaming accumulation (F-6)
        let thinking = resp
            .message
            .content
            .iter()
            .find_map(|p| {
                if let ContentPart::Thinking { thinking } = p {
                    Some(thinking)
                } else {
                    None
                }
            })
            .expect("should have Thinking content part");
        assert_eq!(
            thinking.signature,
            Some("sig_abc123".to_string()),
            "F-6: signature must be preserved through streaming accumulation"
        );
    }

    #[test]
    fn test_multiple_thinking_blocks_preserved_separately() {
        // BUG-2: When Anthropic sends multiple thinking blocks
        // (ReasoningStart/Delta/End repeated), each block must be
        // emitted as a separate ContentPart::Thinking with its own signature.
        let mut acc = StreamAccumulator::new();

        // --- Block 1 ---
        acc.process(&StreamEvent {
            event_type: StreamEventType::ReasoningStart,
            ..Default::default()
        });
        acc.process(&StreamEvent {
            event_type: StreamEventType::ReasoningDelta,
            reasoning_delta: Some("First ".into()),
            ..Default::default()
        });
        acc.process(&StreamEvent {
            event_type: StreamEventType::ReasoningDelta,
            reasoning_delta: Some("thought".into()),
            ..Default::default()
        });
        acc.process(&StreamEvent {
            event_type: StreamEventType::ReasoningEnd,
            raw: Some(serde_json::json!({"signature": "sig_block1"})),
            ..Default::default()
        });

        // --- Block 2 ---
        acc.process(&StreamEvent {
            event_type: StreamEventType::ReasoningStart,
            ..Default::default()
        });
        acc.process(&StreamEvent {
            event_type: StreamEventType::ReasoningDelta,
            reasoning_delta: Some("Second thought".into()),
            ..Default::default()
        });
        acc.process(&StreamEvent {
            event_type: StreamEventType::ReasoningEnd,
            raw: Some(serde_json::json!({"signature": "sig_block2"})),
            ..Default::default()
        });

        // --- Text + Finish ---
        acc.process(&StreamEvent {
            event_type: StreamEventType::TextDelta,
            delta: Some("answer".into()),
            ..Default::default()
        });
        acc.process(&StreamEvent {
            event_type: StreamEventType::Finish,
            finish_reason: Some(FinishReason::stop()),
            usage: Some(Usage::default()),
            ..Default::default()
        });

        let resp = acc.response().unwrap();

        // Collect all Thinking content parts
        let thinking_parts: Vec<&ThinkingData> = resp
            .message
            .content
            .iter()
            .filter_map(|p| {
                if let ContentPart::Thinking { thinking } = p {
                    Some(thinking)
                } else {
                    None
                }
            })
            .collect();

        assert_eq!(
            thinking_parts.len(),
            2,
            "should emit 2 separate Thinking blocks, got {}",
            thinking_parts.len()
        );

        // Block 1
        assert_eq!(thinking_parts[0].text, "First thought");
        assert_eq!(
            thinking_parts[0].signature,
            Some("sig_block1".to_string()),
            "block 1 must retain its own signature"
        );

        // Block 2
        assert_eq!(thinking_parts[1].text, "Second thought");
        assert_eq!(
            thinking_parts[1].signature,
            Some("sig_block2".to_string()),
            "block 2 must retain its own signature"
        );

        // Text is still correct
        assert_eq!(resp.text(), "answer");
    }

    #[test]
    fn test_three_thinking_blocks_with_mixed_signatures() {
        // Edge case: block without a signature (no raw on ReasoningEnd)
        let mut acc = StreamAccumulator::new();

        // Block 1: has signature
        acc.process(&StreamEvent {
            event_type: StreamEventType::ReasoningStart,
            ..Default::default()
        });
        acc.process(&StreamEvent {
            event_type: StreamEventType::ReasoningDelta,
            reasoning_delta: Some("A".into()),
            ..Default::default()
        });
        acc.process(&StreamEvent {
            event_type: StreamEventType::ReasoningEnd,
            raw: Some(serde_json::json!({"signature": "sig_A"})),
            ..Default::default()
        });

        // Block 2: NO signature
        acc.process(&StreamEvent {
            event_type: StreamEventType::ReasoningStart,
            ..Default::default()
        });
        acc.process(&StreamEvent {
            event_type: StreamEventType::ReasoningDelta,
            reasoning_delta: Some("B".into()),
            ..Default::default()
        });
        acc.process(&StreamEvent {
            event_type: StreamEventType::ReasoningEnd,
            ..Default::default()
        });

        // Block 3: has signature
        acc.process(&StreamEvent {
            event_type: StreamEventType::ReasoningStart,
            ..Default::default()
        });
        acc.process(&StreamEvent {
            event_type: StreamEventType::ReasoningDelta,
            reasoning_delta: Some("C".into()),
            ..Default::default()
        });
        acc.process(&StreamEvent {
            event_type: StreamEventType::ReasoningEnd,
            raw: Some(serde_json::json!({"signature": "sig_C"})),
            ..Default::default()
        });

        acc.process(&StreamEvent {
            event_type: StreamEventType::Finish,
            finish_reason: Some(FinishReason::stop()),
            usage: Some(Usage::default()),
            ..Default::default()
        });

        let resp = acc.response().unwrap();
        let thinking_parts: Vec<&ThinkingData> = resp
            .message
            .content
            .iter()
            .filter_map(|p| {
                if let ContentPart::Thinking { thinking } = p {
                    Some(thinking)
                } else {
                    None
                }
            })
            .collect();

        assert_eq!(thinking_parts.len(), 3);
        assert_eq!(thinking_parts[0].text, "A");
        assert_eq!(thinking_parts[0].signature, Some("sig_A".to_string()));
        assert_eq!(thinking_parts[1].text, "B");
        assert_eq!(
            thinking_parts[1].signature, None,
            "block 2 has no signature"
        );
        assert_eq!(thinking_parts[2].text, "C");
        assert_eq!(thinking_parts[2].signature, Some("sig_C".to_string()));
    }

    #[test]
    fn test_finish_extracts_model_from_response() {
        let mut acc = StreamAccumulator::new();
        acc.process(&StreamEvent {
            event_type: StreamEventType::TextDelta,
            delta: Some("Hi".into()),
            ..Default::default()
        });
        acc.process(&StreamEvent {
            event_type: StreamEventType::Finish,
            finish_reason: Some(FinishReason::stop()),
            usage: Some(Usage::default()),
            response: Some(Box::new(Response {
                id: String::new(),
                model: "gpt-4".into(),
                provider: "openai".into(),
                message: Message {
                    role: unified_llm_types::message::Role::Assistant,
                    content: vec![],
                    name: None,
                    tool_call_id: None,
                },
                finish_reason: FinishReason::stop(),
                usage: Usage::default(),
                raw: None,
                warnings: vec![],
                rate_limit: None,
            })),
            ..Default::default()
        });
        let resp = acc.response().unwrap();
        assert_eq!(resp.model, "gpt-4");
        assert_eq!(resp.provider, "openai");
        assert_eq!(resp.text(), "Hi");
    }
}
