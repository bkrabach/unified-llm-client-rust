//! Bidirectional conversions between content types (ToolCallData, ToolResultData)
//! and high-level types (ToolCall, ToolResult) from the tool module.

use crate::content::{ArgumentValue, ToolCallData, ToolResultData};
use crate::tool::{ToolCall, ToolResult};

impl From<ToolCallData> for ToolCall {
    fn from(data: ToolCallData) -> Self {
        let (arguments, raw_arguments) = match data.arguments {
            ArgumentValue::Dict(map) => (map, None),
            ArgumentValue::Raw(s) => {
                let parsed = serde_json::from_str(&s).unwrap_or_default();
                (parsed, Some(s))
            }
        };
        ToolCall {
            id: data.id,
            name: data.name,
            arguments,
            raw_arguments,
        }
    }
}

impl From<ToolCall> for ToolCallData {
    fn from(call: ToolCall) -> Self {
        ToolCallData {
            id: call.id,
            name: call.name,
            arguments: ArgumentValue::Dict(call.arguments),
            r#type: "function".to_string(),
        }
    }
}

impl From<ToolResultData> for ToolResult {
    fn from(data: ToolResultData) -> Self {
        ToolResult {
            tool_call_id: data.tool_call_id,
            content: data.content,
            is_error: data.is_error,
        }
    }
}

impl From<ToolResult> for ToolResultData {
    fn from(result: ToolResult) -> Self {
        ToolResultData {
            tool_call_id: result.tool_call_id,
            content: result.content,
            is_error: result.is_error,
            image_data: None,
            image_media_type: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_tool_call_data_to_tool_call_dict() {
        let data = ToolCallData {
            id: "call_1".into(),
            name: "get_weather".into(),
            arguments: ArgumentValue::Dict({
                let mut m = serde_json::Map::new();
                m.insert("city".into(), json!("SF"));
                m
            }),
            r#type: "function".into(),
        };
        let call: ToolCall = data.into();
        assert_eq!(call.id, "call_1");
        assert_eq!(call.name, "get_weather");
        assert_eq!(call.arguments["city"], "SF");
        assert!(call.raw_arguments.is_none());
    }

    #[test]
    fn test_tool_call_data_to_tool_call_raw() {
        let data = ToolCallData {
            id: "call_2".into(),
            name: "search".into(),
            arguments: ArgumentValue::Raw("{\"q\":\"rust\"}".into()),
            r#type: "function".into(),
        };
        let call: ToolCall = data.into();
        assert_eq!(call.arguments["q"], "rust");
        assert_eq!(call.raw_arguments, Some("{\"q\":\"rust\"}".into()));
    }

    #[test]
    fn test_tool_call_to_tool_call_data() {
        let call = ToolCall {
            id: "call_3".into(),
            name: "fn".into(),
            arguments: serde_json::Map::new(),
            raw_arguments: Some("{}".into()),
        };
        let data: ToolCallData = call.into();
        assert_eq!(data.id, "call_3");
        assert_eq!(data.r#type, "function");
        assert!(matches!(data.arguments, ArgumentValue::Dict(_)));
    }

    #[test]
    fn test_tool_result_data_to_tool_result() {
        let data = ToolResultData {
            tool_call_id: "call_1".into(),
            content: json!("sunny"),
            is_error: false,
            image_data: Some(vec![1, 2, 3]),
            image_media_type: Some("image/png".into()),
        };
        let result: ToolResult = data.into();
        assert_eq!(result.tool_call_id, "call_1");
        assert!(!result.is_error);
        // image_data is lost in the conversion (ToolResult doesn't have it)
    }

    #[test]
    fn test_tool_result_to_tool_result_data() {
        let result = ToolResult {
            tool_call_id: "call_2".into(),
            content: json!({"temp": "72F"}),
            is_error: true,
        };
        let data: ToolResultData = result.into();
        assert_eq!(data.tool_call_id, "call_2");
        assert!(data.is_error);
        assert!(data.image_data.is_none());
        assert!(data.image_media_type.is_none());
    }

    #[test]
    fn test_round_trip_tool_call() {
        let original = ToolCallData {
            id: "call_rt".into(),
            name: "echo".into(),
            arguments: ArgumentValue::Dict(serde_json::Map::new()),
            r#type: "function".into(),
        };
        let call: ToolCall = original.clone().into();
        let back: ToolCallData = call.into();
        assert_eq!(back.id, "call_rt");
        assert_eq!(back.name, "echo");
    }
}
