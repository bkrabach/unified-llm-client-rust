use serde::{Deserialize, Serialize};

use crate::response::{FinishReason, Response, Usage};
use crate::tool::ToolCall;

/// Stream event types: 13 from spec (§3.13) + 1 library extension (`StepFinish`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StreamEventType {
    StreamStart,
    TextStart,
    TextDelta,
    TextEnd,
    ReasoningStart,
    ReasoningDelta,
    ReasoningEnd,
    ToolCallStart,
    ToolCallDelta,
    ToolCallEnd,
    /// **Library extension** (not in the base spec's 13 event types).
    ///
    /// Marks the end of one LLM call within a multi-step streaming tool loop.
    /// Emitted between tool execution rounds so consumers can distinguish events
    /// from different steps. Carries step metadata (e.g.,
    /// `{"step": 0, "tool_calls": 2}`), plus `usage` and `finish_reason`.
    StepFinish,
    Finish,
    Error,
    ProviderEvent,
    /// Catch-all for unrecognized event types (forward compatibility).
    Unknown(String),
}

impl Serialize for StreamEventType {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let s = match self {
            Self::StreamStart => "STREAM_START",
            Self::TextStart => "TEXT_START",
            Self::TextDelta => "TEXT_DELTA",
            Self::TextEnd => "TEXT_END",
            Self::ReasoningStart => "REASONING_START",
            Self::ReasoningDelta => "REASONING_DELTA",
            Self::ReasoningEnd => "REASONING_END",
            Self::ToolCallStart => "TOOL_CALL_START",
            Self::ToolCallDelta => "TOOL_CALL_DELTA",
            Self::ToolCallEnd => "TOOL_CALL_END",
            Self::StepFinish => "STEP_FINISH",
            Self::Finish => "FINISH",
            Self::Error => "ERROR",
            Self::ProviderEvent => "PROVIDER_EVENT",
            Self::Unknown(s) => s.as_str(),
        };
        serializer.serialize_str(s)
    }
}

impl<'de> Deserialize<'de> for StreamEventType {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        Ok(match s.as_str() {
            "STREAM_START" => Self::StreamStart,
            "TEXT_START" => Self::TextStart,
            "TEXT_DELTA" => Self::TextDelta,
            "TEXT_END" => Self::TextEnd,
            "REASONING_START" => Self::ReasoningStart,
            "REASONING_DELTA" => Self::ReasoningDelta,
            "REASONING_END" => Self::ReasoningEnd,
            "TOOL_CALL_START" => Self::ToolCallStart,
            "TOOL_CALL_DELTA" => Self::ToolCallDelta,
            "TOOL_CALL_END" => Self::ToolCallEnd,
            "STEP_FINISH" => Self::StepFinish,
            "FINISH" => Self::Finish,
            "ERROR" => Self::Error,
            "PROVIDER_EVENT" => Self::ProviderEvent,
            _ => Self::Unknown(s),
        })
    }
}

/// Structured error carried on `StreamEventType::Error` events (spec §3.13).
///
/// Captures the essential error classification from the full `Error` type
/// in a serializable form suitable for stream events.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamError {
    /// Error classification (e.g., `RateLimit`, `Server`, `Network`).
    pub kind: crate::error::ErrorKind,
    /// Human-readable error description.
    pub message: String,
    /// Whether the operation that caused this error can be retried.
    pub retryable: bool,
    /// Provider that originated the error (e.g., `"anthropic"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    /// HTTP status code from the provider response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status_code: Option<u16>,
    /// How long to wait before retrying (from provider `Retry-After` header).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retry_after: Option<std::time::Duration>,
}

impl StreamError {
    /// Create from a message string with `Stream` kind (retryable).
    pub fn stream(message: impl Into<String>) -> Self {
        Self {
            kind: crate::error::ErrorKind::Stream,
            message: message.into(),
            retryable: true,
            provider: None,
            status_code: None,
            retry_after: None,
        }
    }
    /// Create from a message string with `RequestTimeout` kind (retryable per spec §6.3).
    pub fn timeout(message: impl Into<String>) -> Self {
        Self {
            kind: crate::error::ErrorKind::RequestTimeout,
            message: message.into(),
            retryable: true,
            provider: None,
            status_code: None,
            retry_after: None,
        }
    }
    /// Create from a full `Error` reference, preserving provider context.
    pub fn from_error(error: &crate::error::Error) -> Self {
        Self {
            kind: error.kind,
            message: error.message.clone(),
            retryable: error.retryable,
            provider: error.provider.clone(),
            status_code: error.status_code,
            retry_after: error.retry_after,
        }
    }
}

/// A single event in a streaming response (spec §3.13).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamEvent {
    /// The type of this stream event.
    #[serde(rename = "type")]
    pub event_type: StreamEventType,

    /// The event ID (e.g., response ID from provider).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,

    /// Incremental text content (spec §3.13: TEXT_DELTA, TOOL_CALL_DELTA).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delta: Option<String>,

    /// Identifies which text segment this delta belongs to (spec §3.13).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_id: Option<String>,

    /// Incremental reasoning/thinking text (spec §3.13: REASONING_DELTA).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_delta: Option<String>,

    /// Partial or complete tool call (spec §3.13: TOOL_CALL_START/DELTA/END).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call: Option<ToolCall>,

    /// Finish reason (present on Finish events).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,

    /// Token usage (present on Finish events).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,

    /// Full accumulated response (spec §3.13: present on FINISH).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<Box<Response>>,

    /// Structured error (present on Error events).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<Box<StreamError>>,

    /// Raw provider event data for passthrough.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw: Option<serde_json::Value>,
}

impl Default for StreamEvent {
    fn default() -> Self {
        Self {
            event_type: StreamEventType::StreamStart,
            id: None,
            delta: None,
            text_id: None,
            reasoning_delta: None,
            tool_call: None,
            finish_reason: None,
            usage: None,
            response: None,
            error: None,
            raw: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- StreamEventType tests ---

    #[test]
    fn test_all_event_types_exist() {
        // 13 from spec + 1 library extension (StepFinish) = 14 total
        let types = vec![
            // --- 13 spec-defined event types (§3.13) ---
            StreamEventType::StreamStart,
            StreamEventType::TextStart,
            StreamEventType::TextDelta,
            StreamEventType::TextEnd,
            StreamEventType::ReasoningStart,
            StreamEventType::ReasoningDelta,
            StreamEventType::ReasoningEnd,
            StreamEventType::ToolCallStart,
            StreamEventType::ToolCallDelta,
            StreamEventType::ToolCallEnd,
            StreamEventType::Finish,
            StreamEventType::Error,
            StreamEventType::ProviderEvent,
            // --- 1 library extension ---
            StreamEventType::StepFinish,
        ];
        assert_eq!(types.len(), 14, "13 from spec + 1 extension (StepFinish)");
    }

    #[test]
    fn test_stream_event_type_serde_screaming_snake_case() {
        let cases = vec![
            (StreamEventType::StreamStart, "\"STREAM_START\""),
            (StreamEventType::TextStart, "\"TEXT_START\""),
            (StreamEventType::TextDelta, "\"TEXT_DELTA\""),
            (StreamEventType::TextEnd, "\"TEXT_END\""),
            (StreamEventType::ReasoningStart, "\"REASONING_START\""),
            (StreamEventType::ReasoningDelta, "\"REASONING_DELTA\""),
            (StreamEventType::ReasoningEnd, "\"REASONING_END\""),
            (StreamEventType::ToolCallStart, "\"TOOL_CALL_START\""),
            (StreamEventType::ToolCallDelta, "\"TOOL_CALL_DELTA\""),
            (StreamEventType::ToolCallEnd, "\"TOOL_CALL_END\""),
            (StreamEventType::StepFinish, "\"STEP_FINISH\""),
            (StreamEventType::Finish, "\"FINISH\""),
            (StreamEventType::Error, "\"ERROR\""),
            (StreamEventType::ProviderEvent, "\"PROVIDER_EVENT\""),
        ];
        for (variant, expected_json) in cases {
            let json = serde_json::to_string(&variant).unwrap();
            assert_eq!(json, expected_json, "Failed for {:?}", variant);
            let back: StreamEventType = serde_json::from_str(&json).unwrap();
            assert_eq!(back, variant);
        }
    }

    #[test]
    fn test_stream_event_type_text_delta_serde() {
        let t = StreamEventType::TextDelta;
        let json = serde_json::to_string(&t).unwrap();
        assert_eq!(json, "\"TEXT_DELTA\"");
    }

    // --- StreamEvent tests ---

    #[test]
    fn test_stream_event_default() {
        let evt = StreamEvent::default();
        assert_eq!(evt.event_type, StreamEventType::StreamStart);
        assert!(evt.id.is_none());
        assert!(evt.delta.is_none());
        assert!(evt.finish_reason.is_none());
        assert!(evt.usage.is_none());
        assert!(evt.error.is_none());
        assert!(evt.raw.is_none());
    }

    #[test]
    fn test_stream_event_text_delta() {
        let evt = StreamEvent {
            event_type: StreamEventType::TextDelta,
            delta: Some("Hello".into()),
            ..Default::default()
        };
        assert_eq!(evt.event_type, StreamEventType::TextDelta);
        assert_eq!(evt.delta, Some("Hello".into()));
    }

    #[test]
    fn test_stream_event_finish() {
        let evt = StreamEvent {
            event_type: StreamEventType::Finish,
            finish_reason: Some(FinishReason::stop()),
            usage: Some(Usage {
                input_tokens: 10,
                output_tokens: 5,
                total_tokens: 15,
                ..Default::default()
            }),
            ..Default::default()
        };
        assert_eq!(evt.event_type, StreamEventType::Finish);
        assert_eq!(evt.finish_reason.as_ref().unwrap().reason, "stop");
        assert_eq!(evt.usage.as_ref().unwrap().total_tokens, 15);
    }

    #[test]
    fn test_stream_event_error() {
        let evt = StreamEvent {
            event_type: StreamEventType::Error,
            error: Some(Box::new(StreamError::stream("Connection lost"))),
            ..Default::default()
        };
        assert_eq!(evt.event_type, StreamEventType::Error);
        assert_eq!(evt.error.as_ref().unwrap().message, "Connection lost");
    }

    #[test]
    fn test_stream_event_provider_event() {
        let evt = StreamEvent {
            event_type: StreamEventType::ProviderEvent,
            raw: Some(serde_json::json!({"raw": true})),
            ..Default::default()
        };
        assert_eq!(evt.event_type, StreamEventType::ProviderEvent);
        assert!(evt.raw.is_some());
    }

    #[test]
    fn test_stream_event_serde_roundtrip() {
        let evt = StreamEvent {
            event_type: StreamEventType::TextDelta,
            id: Some("evt_1".into()),
            delta: Some("world".into()),
            ..Default::default()
        };
        let json = serde_json::to_string(&evt).unwrap();
        assert!(json.contains("\"TEXT_DELTA\""));
        let back: StreamEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back.event_type, StreamEventType::TextDelta);
        assert_eq!(back.delta, Some("world".into()));
        assert_eq!(back.id, Some("evt_1".into()));
    }

    #[test]
    fn test_stream_event_optional_fields_omitted() {
        let evt = StreamEvent {
            event_type: StreamEventType::StreamStart,
            ..Default::default()
        };
        let json = serde_json::to_string(&evt).unwrap();
        assert!(!json.contains("delta"));
        assert!(!json.contains("finish_reason"));
        assert!(!json.contains("usage"));
        assert!(!json.contains("error"));
        assert!(!json.contains("raw"));
        assert!(!json.contains("\"id\""));
    }

    #[test]
    fn test_stream_event_reasoning_delta() {
        let evt = StreamEvent {
            event_type: StreamEventType::ReasoningDelta,
            delta: Some("Let me think...".into()),
            ..Default::default()
        };
        let json = serde_json::to_string(&evt).unwrap();
        let back: StreamEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back.event_type, StreamEventType::ReasoningDelta);
        assert_eq!(back.delta, Some("Let me think...".into()));
    }

    #[test]
    fn test_stream_event_tool_call_start() {
        let evt = StreamEvent {
            event_type: StreamEventType::ToolCallStart,
            tool_call: Some(ToolCall {
                id: "call_1".into(),
                name: "get_weather".into(),
                arguments: serde_json::Map::new(),
                raw_arguments: None,
            }),
            ..Default::default()
        };
        let json = serde_json::to_string(&evt).unwrap();
        let back: StreamEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back.event_type, StreamEventType::ToolCallStart);
        assert_eq!(back.tool_call.as_ref().unwrap().name, "get_weather");
    }

    #[test]
    fn test_unknown_event_type_deserializes() {
        let v: StreamEventType = serde_json::from_str("\"SOME_FUTURE_EVENT\"").unwrap();
        assert_eq!(v, StreamEventType::Unknown("SOME_FUTURE_EVENT".to_string()));
        // Round-trip
        let json = serde_json::to_string(&v).unwrap();
        assert_eq!(json, "\"SOME_FUTURE_EVENT\"");
    }

    #[test]
    fn test_stream_error_from_error_preserves_retry_after() {
        let error = crate::error::Error::from_http_status(
            429,
            "Rate limited".into(),
            "anthropic",
            None,
            Some(std::time::Duration::from_secs(30)),
        );
        let stream_err = StreamError::from_error(&error);
        assert_eq!(stream_err.kind, crate::error::ErrorKind::RateLimit);
        assert_eq!(
            stream_err.retry_after,
            Some(std::time::Duration::from_secs(30))
        );
        assert_eq!(stream_err.status_code, Some(429));
        assert_eq!(stream_err.provider, Some("anthropic".to_string()));
        assert!(stream_err.retryable);
    }

    #[test]
    fn test_stream_error_constructors_default_new_fields_to_none() {
        let stream_err = StreamError::stream("connection lost");
        assert!(stream_err.provider.is_none());
        assert!(stream_err.status_code.is_none());
        assert!(stream_err.retry_after.is_none());

        let timeout_err = StreamError::timeout("timed out");
        assert!(timeout_err.provider.is_none());
        assert!(timeout_err.status_code.is_none());
        assert!(timeout_err.retry_after.is_none());
    }

    #[test]
    fn test_stream_event_error_preserves_kind_and_retryable() {
        let evt = StreamEvent {
            event_type: StreamEventType::Error,
            error: Some(Box::new(StreamError {
                kind: crate::error::ErrorKind::RateLimit,
                message: "Too many requests".into(),
                retryable: true,
                provider: None,
                status_code: None,
                retry_after: None,
            })),
            ..Default::default()
        };
        let err = evt.error.as_ref().unwrap();
        assert_eq!(err.kind, crate::error::ErrorKind::RateLimit);
        assert_eq!(err.message, "Too many requests");
        assert!(err.retryable);
        // Verify serde roundtrip preserves error info
        let json = serde_json::to_string(&evt).unwrap();
        let back: StreamEvent = serde_json::from_str(&json).unwrap();
        let back_err = back.error.as_ref().unwrap();
        assert_eq!(back_err.kind, crate::error::ErrorKind::RateLimit);
        assert!(back_err.retryable);
    }
}
