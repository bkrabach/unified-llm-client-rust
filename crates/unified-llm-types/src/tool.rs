use serde::{Deserialize, Serialize};

/// Tool definition sent to the provider (serializable subset, no execute handler).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
    /// If true, the provider enforces strict schema adherence (OpenAI-specific).
    /// Defaults to `true` when `None` for OpenAI, ignored by other providers.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub strict: Option<bool>,
}

impl ToolDefinition {
    /// Validate tool name format per spec §5.1: `[a-zA-Z][a-zA-Z0-9_]{0,63}`
    /// and that parameters have `"type": "object"` at root.
    pub fn validate(&self) -> Result<(), crate::error::Error> {
        // Validate name: 1-64 chars, starts with letter, then alphanumeric or underscore
        if self.name.is_empty() || self.name.len() > 64 {
            return Err(crate::error::Error::configuration(format!(
                "Tool name '{}' must be 1-64 characters",
                self.name
            )));
        }
        let valid = self.name.chars().enumerate().all(|(i, c)| {
            if i == 0 {
                c.is_ascii_alphabetic()
            } else {
                c.is_ascii_alphanumeric() || c == '_'
            }
        });
        if !valid {
            return Err(crate::error::Error::configuration(format!(
                "Tool name '{}' must match [a-zA-Z][a-zA-Z0-9_]{{0,63}}",
                self.name
            )));
        }

        // Validate parameters: must be a JSON object with "type": "object"
        if let Some(obj) = self.parameters.as_object() {
            if obj.get("type").and_then(|v| v.as_str()) != Some("object") {
                return Err(crate::error::Error::configuration(
                    "Tool parameters must have \"type\": \"object\" at root",
                ));
            }
        } else {
            return Err(crate::error::Error::configuration(
                "Tool parameters must be a JSON object",
            ));
        }

        Ok(())
    }
}

/// Tool choice configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolChoice {
    pub mode: String, // "auto", "none", "required", "named"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
}

/// Valid tool choice modes recognized by all providers.
const VALID_TOOL_CHOICE_MODES: &[&str] = &["auto", "none", "required", "named"];

impl ToolChoice {
    /// Validate that the tool choice mode is a recognized value.
    /// Returns `Err(InvalidRequest)` for unknown modes.
    pub fn validate(&self) -> Result<(), crate::error::Error> {
        if !VALID_TOOL_CHOICE_MODES.contains(&self.mode.as_str()) {
            return Err(crate::error::Error::unsupported_tool_choice(&self.mode));
        }
        if self.mode == "named" && self.tool_name.is_none() {
            return Err(crate::error::Error::configuration(
                "ToolChoice mode \"named\" requires tool_name to be set",
            ));
        }
        Ok(())
    }
}

/// A parsed tool call from a provider response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Map<String, serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_arguments: Option<String>,
}

/// Result of a tool execution, sent back to the provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub content: serde_json::Value,
    #[serde(default)]
    pub is_error: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_choice_modes() {
        for mode in ["auto", "none", "required", "named"] {
            let tc = ToolChoice {
                mode: mode.into(),
                tool_name: None,
            };
            assert_eq!(tc.mode, mode);
        }
    }

    #[test]
    fn test_tool_choice_named_with_tool_name() {
        let tc = ToolChoice {
            mode: "named".into(),
            tool_name: Some("get_weather".into()),
        };
        let json = serde_json::to_string(&tc).unwrap();
        assert!(json.contains("get_weather"));
        let back: ToolChoice = serde_json::from_str(&json).unwrap();
        assert_eq!(back.mode, "named");
        assert_eq!(back.tool_name, Some("get_weather".into()));
    }

    #[test]
    fn test_tool_choice_tool_name_omitted_when_none() {
        let tc = ToolChoice {
            mode: "auto".into(),
            tool_name: None,
        };
        let json = serde_json::to_string(&tc).unwrap();
        assert!(!json.contains("tool_name"));
    }

    #[test]
    fn test_tool_definition_serde() {
        let def = ToolDefinition {
            name: "get_weather".into(),
            description: "Get weather".into(),
            parameters: serde_json::json!({"type": "object", "properties": {}}),
            strict: None,
        };
        let json = serde_json::to_string(&def).unwrap();
        assert!(json.contains("get_weather"));
        let back: ToolDefinition = serde_json::from_str(&json).unwrap();
        assert_eq!(back.name, "get_weather");
        assert_eq!(back.description, "Get weather");
    }

    #[test]
    fn test_tool_definition_roundtrip() {
        let def = ToolDefinition {
            name: "search".into(),
            description: "Search the web".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }),
            strict: None,
        };
        let json = serde_json::to_string(&def).unwrap();
        let back: ToolDefinition = serde_json::from_str(&json).unwrap();
        assert_eq!(back.name, "search");
        assert_eq!(back.parameters["properties"]["query"]["type"], "string");
    }

    #[test]
    fn test_tool_call_serde() {
        let mut args = serde_json::Map::new();
        args.insert(
            "city".to_string(),
            serde_json::Value::String("SF".to_string()),
        );
        let call = ToolCall {
            id: "call_1".into(),
            name: "get_weather".into(),
            arguments: args,
            raw_arguments: Some("{\"city\":\"SF\"}".into()),
        };
        let json = serde_json::to_string(&call).unwrap();
        let back: ToolCall = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, "call_1");
        assert_eq!(back.arguments["city"], "SF");
        assert_eq!(back.raw_arguments, Some("{\"city\":\"SF\"}".into()));
    }

    #[test]
    fn test_tool_call_raw_arguments_omitted_when_none() {
        let call = ToolCall {
            id: "call_1".into(),
            name: "fn".into(),
            arguments: serde_json::Map::new(),
            raw_arguments: None,
        };
        let json = serde_json::to_string(&call).unwrap();
        assert!(!json.contains("raw_arguments"));
    }

    #[test]
    fn test_tool_result_serde() {
        let result = ToolResult {
            tool_call_id: "call_1".into(),
            content: serde_json::Value::String("sunny".into()),
            is_error: false,
        };
        let json = serde_json::to_string(&result).unwrap();
        let back: ToolResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.tool_call_id, "call_1");
        assert!(!back.is_error);
    }

    #[test]
    fn test_tool_result_is_error_defaults_false() {
        let json = r#"{"tool_call_id":"call_1","content":"result"}"#;
        let result: ToolResult = serde_json::from_str(json).unwrap();
        assert!(!result.is_error);
    }

    // --- AF-16: Tool choice validation ---

    #[test]
    fn test_tool_choice_valid_modes() {
        for mode in ["auto", "none", "required"] {
            let tc = ToolChoice {
                mode: mode.into(),
                tool_name: None,
            };
            assert!(tc.validate().is_ok(), "Mode '{mode}' should be valid");
        }
        // "named" requires tool_name
        let tc = ToolChoice {
            mode: "named".into(),
            tool_name: Some("some_tool".into()),
        };
        assert!(
            tc.validate().is_ok(),
            "Mode 'named' with tool_name should be valid"
        );
    }

    #[test]
    fn test_tool_choice_named_without_tool_name_errors() {
        let tc = ToolChoice {
            mode: "named".into(),
            tool_name: None,
        };
        let err = tc.validate().unwrap_err();
        assert_eq!(err.kind, crate::error::ErrorKind::Configuration);
        assert!(
            err.message.contains("tool_name"),
            "Error should mention tool_name"
        );
    }

    #[test]
    fn test_tool_choice_named_with_tool_name_valid() {
        let tc = ToolChoice {
            mode: "named".into(),
            tool_name: Some("get_weather".into()),
        };
        assert!(tc.validate().is_ok());
    }

    #[test]
    fn test_tool_choice_invalid_mode_errors() {
        let tc = ToolChoice {
            mode: "invalid_mode".into(),
            tool_name: None,
        };
        let err = tc.validate().unwrap_err();
        assert_eq!(err.kind, crate::error::ErrorKind::InvalidRequest);
        assert!(err.message.contains("invalid_mode"));
    }

    #[test]
    fn test_tool_choice_unknown_mode_not_silently_auto() {
        let tc = ToolChoice {
            mode: "something_else".into(),
            tool_name: None,
        };
        assert!(
            tc.validate().is_err(),
            "Unknown mode should error, not silently become auto"
        );
    }

    // --- AF-14: Tool name validation ---

    #[test]
    fn test_tool_name_validation_valid() {
        let def = ToolDefinition {
            name: "get_weather".into(),
            description: "Get weather".into(),
            parameters: serde_json::json!({"type": "object"}),
            strict: None,
        };
        assert!(def.validate().is_ok());
    }

    #[test]
    fn test_tool_name_validation_single_char() {
        let def = ToolDefinition {
            name: "a".into(),
            description: "Single char".into(),
            parameters: serde_json::json!({"type": "object"}),
            strict: None,
        };
        assert!(def.validate().is_ok());
    }

    #[test]
    fn test_tool_name_validation_alphanumeric_underscore() {
        let def = ToolDefinition {
            name: "a1_b2".into(),
            description: "Mixed".into(),
            parameters: serde_json::json!({"type": "object"}),
            strict: None,
        };
        assert!(def.validate().is_ok());
    }

    #[test]
    fn test_tool_name_validation_empty() {
        let def = ToolDefinition {
            name: "".into(),
            description: "Empty name".into(),
            parameters: serde_json::json!({"type": "object"}),
            strict: None,
        };
        assert!(def.validate().is_err());
    }

    #[test]
    fn test_tool_name_validation_starts_with_number() {
        let def = ToolDefinition {
            name: "1tool".into(),
            description: "Starts with number".into(),
            parameters: serde_json::json!({"type": "object"}),
            strict: None,
        };
        assert!(def.validate().is_err());
    }

    #[test]
    fn test_tool_name_validation_too_long() {
        let def = ToolDefinition {
            name: "a".repeat(65),
            description: "Too long".into(),
            parameters: serde_json::json!({"type": "object"}),
            strict: None,
        };
        assert!(def.validate().is_err());
    }

    #[test]
    fn test_tool_name_validation_special_chars() {
        let def = ToolDefinition {
            name: "get-weather".into(),
            description: "Has hyphens".into(),
            parameters: serde_json::json!({"type": "object"}),
            strict: None,
        };
        assert!(def.validate().is_err());
    }

    // --- AF-15: Parameter schema root validation ---

    #[test]
    fn test_tool_parameters_must_be_object_type() {
        let def = ToolDefinition {
            name: "test_tool".into(),
            description: "test".into(),
            parameters: serde_json::json!({"type": "object", "properties": {}}),
            strict: None,
        };
        assert!(def.validate().is_ok());
    }

    #[test]
    fn test_tool_parameters_rejects_non_object_type() {
        let def = ToolDefinition {
            name: "test_tool".into(),
            description: "test".into(),
            parameters: serde_json::json!({"type": "string"}),
            strict: None,
        };
        assert!(
            def.validate().is_err(),
            "Parameters must have type: object at root"
        );
    }

    #[test]
    fn test_tool_parameters_rejects_missing_type() {
        let def = ToolDefinition {
            name: "test_tool".into(),
            description: "test".into(),
            parameters: serde_json::json!({}),
            strict: None,
        };
        assert!(
            def.validate().is_err(),
            "Parameters without type field should fail"
        );
    }

    #[test]
    fn test_tool_definition_strict_field_serde() {
        let def = ToolDefinition {
            name: "test".into(),
            description: "test".into(),
            parameters: serde_json::json!({"type": "object"}),
            strict: Some(false),
        };
        let json = serde_json::to_string(&def).unwrap();
        assert!(json.contains("\"strict\":false"));
    }

    #[test]
    fn test_tool_definition_strict_none_omitted() {
        let def = ToolDefinition {
            name: "test".into(),
            description: "test".into(),
            parameters: serde_json::json!({"type": "object"}),
            strict: None,
        };
        let json = serde_json::to_string(&def).unwrap();
        assert!(!json.contains("strict"));
    }

    #[test]
    fn test_tool_parameters_rejects_non_object_json() {
        let def = ToolDefinition {
            name: "test_tool".into(),
            description: "test".into(),
            parameters: serde_json::json!("just a string"),
            strict: None,
        };
        assert!(def.validate().is_err(), "Parameters must be a JSON object");
    }
}
