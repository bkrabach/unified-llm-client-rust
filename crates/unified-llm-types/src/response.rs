use serde::{Deserialize, Serialize};

use crate::content::{ContentPart, ThinkingData, ToolCallData};
use crate::message::Message;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Response {
    pub id: String,
    pub model: String,
    pub provider: String,
    pub message: Message,
    pub finish_reason: FinishReason,
    pub usage: Usage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw: Option<serde_json::Value>,
    #[serde(default)]
    pub warnings: Vec<Warning>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rate_limit: Option<RateLimitInfo>,
}

impl Response {
    /// Concatenated text from all text content parts.
    pub fn text(&self) -> String {
        self.message.text()
    }

    /// Extract tool calls from the message.
    pub fn tool_calls(&self) -> Vec<&ToolCallData> {
        self.message
            .content
            .iter()
            .filter_map(|p| match p {
                ContentPart::ToolCall { tool_call } => Some(tool_call),
                _ => None,
            })
            .collect()
    }

    /// Concatenated reasoning/thinking text.
    pub fn reasoning(&self) -> Option<String> {
        let parts: Vec<&str> = self
            .message
            .content
            .iter()
            .filter_map(|p| match p {
                ContentPart::Thinking {
                    thinking: ThinkingData { text, .. },
                } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        if parts.is_empty() {
            None
        } else {
            Some(parts.join(""))
        }
    }
}

/// Dual representation: unified reason + provider-native raw.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinishReason {
    pub reason: String, // "stop", "length", "tool_calls", "content_filter", "error", "other"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw: Option<String>,
}

impl FinishReason {
    pub fn stop() -> Self {
        Self {
            reason: "stop".into(),
            raw: None,
        }
    }
    pub fn length() -> Self {
        Self {
            reason: "length".into(),
            raw: None,
        }
    }
    pub fn tool_calls() -> Self {
        Self {
            reason: "tool_calls".into(),
            raw: None,
        }
    }
    pub fn content_filter() -> Self {
        Self {
            reason: "content_filter".into(),
            raw: None,
        }
    }
    pub fn error() -> Self {
        Self {
            reason: "error".into(),
            raw: None,
        }
    }
    pub fn other() -> Self {
        Self {
            reason: "other".into(),
            raw: None,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_write_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw: Option<serde_json::Value>,
}

fn add_optional(a: Option<u32>, b: Option<u32>) -> Option<u32> {
    match (a, b) {
        (None, None) => None,
        (Some(v), None) | (None, Some(v)) => Some(v),
        (Some(a), Some(b)) => Some(a + b),
    }
}

impl std::ops::Add for Usage {
    type Output = Usage;
    fn add(self, rhs: Usage) -> Usage {
        Usage {
            input_tokens: self.input_tokens + rhs.input_tokens,
            output_tokens: self.output_tokens + rhs.output_tokens,
            total_tokens: self.total_tokens + rhs.total_tokens,
            reasoning_tokens: add_optional(self.reasoning_tokens, rhs.reasoning_tokens),
            cache_read_tokens: add_optional(self.cache_read_tokens, rhs.cache_read_tokens),
            cache_write_tokens: add_optional(self.cache_write_tokens, rhs.cache_write_tokens),
            raw: None, // Aggregated usage drops raw
        }
    }
}

impl std::ops::AddAssign for Usage {
    fn add_assign(&mut self, rhs: Usage) {
        self.input_tokens += rhs.input_tokens;
        self.output_tokens += rhs.output_tokens;
        self.total_tokens += rhs.total_tokens;
        self.reasoning_tokens = add_optional(self.reasoning_tokens, rhs.reasoning_tokens);
        self.cache_read_tokens = add_optional(self.cache_read_tokens, rhs.cache_read_tokens);
        self.cache_write_tokens = add_optional(self.cache_write_tokens, rhs.cache_write_tokens);
        self.raw = None;
    }
}

impl std::ops::Add for &Usage {
    type Output = Usage;
    fn add(self, rhs: &Usage) -> Usage {
        Usage {
            input_tokens: self.input_tokens + rhs.input_tokens,
            output_tokens: self.output_tokens + rhs.output_tokens,
            total_tokens: self.total_tokens + rhs.total_tokens,
            reasoning_tokens: add_optional(self.reasoning_tokens, rhs.reasoning_tokens),
            cache_read_tokens: add_optional(self.cache_read_tokens, rhs.cache_read_tokens),
            cache_write_tokens: add_optional(self.cache_write_tokens, rhs.cache_write_tokens),
            raw: None,
        }
    }
}

/// Response format configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseFormat {
    pub r#type: String, // "text", "json", "json_schema"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub json_schema: Option<serde_json::Value>,
    #[serde(default)]
    pub strict: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Warning {
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RateLimitInfo {
    pub requests_remaining: Option<u32>,
    pub requests_limit: Option<u32>,
    pub tokens_remaining: Option<u32>,
    pub tokens_limit: Option<u32>,
    pub reset_at: Option<String>, // ISO 8601 timestamp
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- FinishReason tests ---

    #[test]
    fn test_finish_reason_constructors() {
        assert_eq!(FinishReason::stop().reason, "stop");
        assert_eq!(FinishReason::length().reason, "length");
        assert_eq!(FinishReason::tool_calls().reason, "tool_calls");
        assert_eq!(FinishReason::content_filter().reason, "content_filter");
    }

    #[test]
    fn test_finish_reason_raw_none_by_default() {
        assert!(FinishReason::stop().raw.is_none());
    }

    #[test]
    fn test_finish_reason_with_raw() {
        let fr = FinishReason {
            reason: "stop".into(),
            raw: Some("end_turn".into()),
        };
        let json = serde_json::to_string(&fr).unwrap();
        assert!(json.contains("end_turn"));
        let back: FinishReason = serde_json::from_str(&json).unwrap();
        assert_eq!(back.raw, Some("end_turn".into()));
    }

    #[test]
    fn test_finish_reason_raw_omitted_when_none() {
        let fr = FinishReason::stop();
        let json = serde_json::to_string(&fr).unwrap();
        assert!(!json.contains("raw"));
    }

    // --- Usage tests ---

    #[test]
    fn test_usage_default() {
        let u = Usage::default();
        assert_eq!(u.input_tokens, 0);
        assert_eq!(u.output_tokens, 0);
        assert_eq!(u.total_tokens, 0);
        assert_eq!(u.reasoning_tokens, None);
        assert_eq!(u.cache_read_tokens, None);
        assert_eq!(u.cache_write_tokens, None);
    }

    #[test]
    fn test_usage_addition_both_some() {
        let a = Usage {
            input_tokens: 10,
            output_tokens: 5,
            total_tokens: 15,
            reasoning_tokens: Some(3),
            cache_read_tokens: None,
            cache_write_tokens: None,
            raw: None,
        };
        let b = Usage {
            input_tokens: 20,
            output_tokens: 10,
            total_tokens: 30,
            reasoning_tokens: Some(7),
            cache_read_tokens: Some(5),
            cache_write_tokens: None,
            raw: None,
        };
        let sum = a + b;
        assert_eq!(sum.input_tokens, 30);
        assert_eq!(sum.output_tokens, 15);
        assert_eq!(sum.total_tokens, 45);
        assert_eq!(sum.reasoning_tokens, Some(10));
        assert_eq!(sum.cache_read_tokens, Some(5)); // None + Some(5) = Some(5)
        assert_eq!(sum.cache_write_tokens, None); // None + None = None
    }

    #[test]
    fn test_usage_addition_none_plus_none() {
        let a = Usage::default();
        let b = Usage::default();
        let sum = a + b;
        assert_eq!(sum.reasoning_tokens, None);
        assert_eq!(sum.cache_read_tokens, None);
        assert_eq!(sum.cache_write_tokens, None);
    }

    #[test]
    fn test_usage_addition_some_plus_none() {
        let a = Usage {
            reasoning_tokens: Some(5),
            ..Default::default()
        };
        let b = Usage::default();
        let sum = a + b;
        assert_eq!(sum.reasoning_tokens, Some(5));
    }

    #[test]
    fn test_usage_addition_none_plus_some() {
        let a = Usage::default();
        let b = Usage {
            cache_read_tokens: Some(10),
            ..Default::default()
        };
        let sum = a + b;
        assert_eq!(sum.cache_read_tokens, Some(10));
    }

    #[test]
    fn test_usage_addition_drops_raw() {
        let a = Usage {
            raw: Some(serde_json::json!({"foo": "bar"})),
            ..Default::default()
        };
        let b = Usage {
            raw: Some(serde_json::json!({"baz": "qux"})),
            ..Default::default()
        };
        let sum = a + b;
        assert!(sum.raw.is_none());
    }

    #[test]
    fn test_usage_serde_roundtrip() {
        let u = Usage {
            input_tokens: 100,
            output_tokens: 50,
            total_tokens: 150,
            reasoning_tokens: Some(20),
            cache_read_tokens: Some(80),
            cache_write_tokens: None,
            raw: None,
        };
        let json = serde_json::to_string(&u).unwrap();
        let back: Usage = serde_json::from_str(&json).unwrap();
        assert_eq!(back.input_tokens, 100);
        assert_eq!(back.reasoning_tokens, Some(20));
        assert_eq!(back.cache_write_tokens, None);
    }

    #[test]
    fn test_usage_optional_fields_omitted() {
        let u = Usage::default();
        let json = serde_json::to_string(&u).unwrap();
        assert!(!json.contains("reasoning_tokens"));
        assert!(!json.contains("cache_read_tokens"));
        assert!(!json.contains("cache_write_tokens"));
        assert!(!json.contains("raw"));
    }

    // --- ResponseFormat tests ---

    #[test]
    fn test_response_format_json_schema() {
        let fmt = ResponseFormat {
            r#type: "json_schema".into(),
            json_schema: Some(serde_json::json!({"type": "object"})),
            strict: true,
        };
        let json = serde_json::to_string(&fmt).unwrap();
        assert!(json.contains("json_schema"));
        let back: ResponseFormat = serde_json::from_str(&json).unwrap();
        assert_eq!(back.r#type, "json_schema");
        assert!(back.strict);
    }

    #[test]
    fn test_response_format_text() {
        let fmt = ResponseFormat {
            r#type: "text".into(),
            json_schema: None,
            strict: false,
        };
        let json = serde_json::to_string(&fmt).unwrap();
        assert!(!json.contains("json_schema"));
    }

    // --- Warning tests ---

    #[test]
    fn test_warning_serde() {
        let w = Warning {
            message: "Deprecated model".into(),
            code: Some("model_deprecated".into()),
        };
        let json = serde_json::to_string(&w).unwrap();
        let back: Warning = serde_json::from_str(&json).unwrap();
        assert_eq!(back.message, "Deprecated model");
        assert_eq!(back.code, Some("model_deprecated".into()));
    }

    #[test]
    fn test_warning_code_omitted_when_none() {
        let w = Warning {
            message: "test".into(),
            code: None,
        };
        let json = serde_json::to_string(&w).unwrap();
        assert!(!json.contains("code"));
    }

    // --- RateLimitInfo tests ---

    #[test]
    fn test_rate_limit_info_default() {
        let r = RateLimitInfo::default();
        assert!(r.requests_remaining.is_none());
        assert!(r.tokens_limit.is_none());
    }

    #[test]
    fn test_rate_limit_info_serde() {
        let r = RateLimitInfo {
            requests_remaining: Some(99),
            requests_limit: Some(100),
            tokens_remaining: Some(90000),
            tokens_limit: Some(100000),
            reset_at: Some("2025-01-01T00:00:00Z".into()),
        };
        let json = serde_json::to_string(&r).unwrap();
        let back: RateLimitInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(back.requests_remaining, Some(99));
        assert_eq!(back.reset_at, Some("2025-01-01T00:00:00Z".into()));
    }

    // --- Response tests ---

    #[test]
    fn test_response_text_accessor() {
        let resp = Response {
            id: "resp_1".into(),
            model: "test".into(),
            provider: "test".into(),
            message: Message::assistant("Hello world"),
            finish_reason: FinishReason::stop(),
            usage: Usage::default(),
            raw: None,
            warnings: vec![],
            rate_limit: None,
        };
        assert_eq!(resp.text(), "Hello world");
    }

    #[test]
    fn test_response_tool_calls_accessor() {
        use crate::content::{ArgumentValue, ToolCallData};
        let resp = Response {
            id: "resp_1".into(),
            model: "test".into(),
            provider: "test".into(),
            message: Message {
                role: crate::message::Role::Assistant,
                content: vec![
                    ContentPart::text("Let me check the weather."),
                    ContentPart::ToolCall {
                        tool_call: ToolCallData {
                            id: "call_1".into(),
                            name: "get_weather".into(),
                            arguments: ArgumentValue::Dict(serde_json::Map::new()),
                            r#type: "function".into(),
                        },
                    },
                ],
                name: None,
                tool_call_id: None,
            },
            finish_reason: FinishReason::tool_calls(),
            usage: Usage::default(),
            raw: None,
            warnings: vec![],
            rate_limit: None,
        };
        let calls = resp.tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
    }

    #[test]
    fn test_response_tool_calls_empty_when_none() {
        let resp = Response {
            id: "resp_1".into(),
            model: "test".into(),
            provider: "test".into(),
            message: Message::assistant("Hello"),
            finish_reason: FinishReason::stop(),
            usage: Usage::default(),
            raw: None,
            warnings: vec![],
            rate_limit: None,
        };
        assert!(resp.tool_calls().is_empty());
    }

    #[test]
    fn test_response_reasoning_accessor() {
        let resp = Response {
            id: "resp_1".into(),
            model: "test".into(),
            provider: "test".into(),
            message: Message {
                role: crate::message::Role::Assistant,
                content: vec![
                    ContentPart::Thinking {
                        thinking: ThinkingData {
                            text: "Let me think...".into(),
                            signature: None,
                            redacted: false,
                            data: None,
                        },
                    },
                    ContentPart::text("The answer is 42."),
                ],
                name: None,
                tool_call_id: None,
            },
            finish_reason: FinishReason::stop(),
            usage: Usage::default(),
            raw: None,
            warnings: vec![],
            rate_limit: None,
        };
        assert_eq!(resp.reasoning(), Some("Let me think...".to_string()));
        assert_eq!(resp.text(), "The answer is 42.");
    }

    #[test]
    fn test_response_reasoning_none_when_no_thinking() {
        let resp = Response {
            id: "resp_1".into(),
            model: "test".into(),
            provider: "test".into(),
            message: Message::assistant("Hello"),
            finish_reason: FinishReason::stop(),
            usage: Usage::default(),
            raw: None,
            warnings: vec![],
            rate_limit: None,
        };
        assert_eq!(resp.reasoning(), None);
    }

    #[test]
    fn test_response_serde_roundtrip() {
        let resp = Response {
            id: "resp_1".into(),
            model: "claude-opus-4-6".into(),
            provider: "anthropic".into(),
            message: Message::assistant("Hello"),
            finish_reason: FinishReason::stop(),
            usage: Usage {
                input_tokens: 10,
                output_tokens: 5,
                total_tokens: 15,
                ..Default::default()
            },
            raw: None,
            warnings: vec![],
            rate_limit: None,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let back: Response = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, "resp_1");
        assert_eq!(back.model, "claude-opus-4-6");
        assert_eq!(back.provider, "anthropic");
        assert_eq!(back.text(), "Hello");
        assert_eq!(back.finish_reason.reason, "stop");
        assert_eq!(back.usage.input_tokens, 10);
    }

    // --- FP-04: FinishReason::error() and FinishReason::other() ---

    #[test]
    fn test_finish_reason_error_constructor() {
        let fr = FinishReason::error();
        assert_eq!(fr.reason, "error");
        assert!(fr.raw.is_none());
    }

    #[test]
    fn test_finish_reason_other_constructor() {
        let fr = FinishReason::other();
        assert_eq!(fr.reason, "other");
        assert!(fr.raw.is_none());
    }

    // --- FP-10: Usage AddAssign and Add<&Usage> ---

    #[test]
    fn test_usage_add_assign() {
        let mut a = Usage {
            input_tokens: 10,
            output_tokens: 5,
            total_tokens: 15,
            ..Default::default()
        };
        let b = Usage {
            input_tokens: 20,
            output_tokens: 10,
            total_tokens: 30,
            ..Default::default()
        };
        a += b;
        assert_eq!(a.input_tokens, 30);
        assert_eq!(a.output_tokens, 15);
        assert_eq!(a.total_tokens, 45);
    }

    #[test]
    fn test_usage_add_ref() {
        let a = Usage {
            input_tokens: 10,
            output_tokens: 5,
            total_tokens: 15,
            ..Default::default()
        };
        let b = Usage {
            input_tokens: 20,
            output_tokens: 10,
            total_tokens: 30,
            ..Default::default()
        };
        let sum = &a + &b;
        assert_eq!(sum.input_tokens, 30);
        // Originals are not consumed
        assert_eq!(a.input_tokens, 10);
        assert_eq!(b.input_tokens, 20);
    }
}
