use serde::{Deserialize, Serialize};

use crate::content::{ContentPart, ToolResultData};

/// The five roles covering all major providers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
    Developer,
}

/// The fundamental unit of conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentPart>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl Message {
    /// Convenience: create a system message from text.
    pub fn system(text: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: vec![ContentPart::text(text)],
            name: None,
            tool_call_id: None,
        }
    }

    /// Convenience: create a user message from text.
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: vec![ContentPart::text(text)],
            name: None,
            tool_call_id: None,
        }
    }

    /// Convenience: create an assistant message from text.
    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: vec![ContentPart::text(text)],
            name: None,
            tool_call_id: None,
        }
    }

    /// Convenience: create a tool result message.
    pub fn tool_result(
        tool_call_id: impl Into<String>,
        content: impl Into<String>,
        is_error: bool,
    ) -> Self {
        Self {
            role: Role::Tool,
            content: vec![ContentPart::ToolResult {
                tool_result: ToolResultData {
                    tool_call_id: tool_call_id.into(),
                    content: serde_json::Value::String(content.into()),
                    is_error,
                    image_data: None,
                    image_media_type: None,
                },
            }],
            name: None,
            tool_call_id: None,
        }
    }

    /// Concatenate text from all TEXT content parts.
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|p| match p {
                ContentPart::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_role_serde_roundtrip() {
        for (role, expected_json) in [
            (Role::System, "\"system\""),
            (Role::User, "\"user\""),
            (Role::Assistant, "\"assistant\""),
            (Role::Tool, "\"tool\""),
            (Role::Developer, "\"developer\""),
        ] {
            let json = serde_json::to_string(&role).unwrap();
            assert_eq!(json, expected_json);
            let back: Role = serde_json::from_str(&json).unwrap();
            assert_eq!(back, role);
        }
    }

    #[test]
    fn test_message_system_constructor() {
        let msg = Message::system("You are helpful.");
        assert_eq!(msg.role, Role::System);
        assert_eq!(msg.text(), "You are helpful.");
    }

    #[test]
    fn test_message_user_constructor() {
        let msg = Message::user("Hello");
        assert_eq!(msg.role, Role::User);
        assert_eq!(msg.text(), "Hello");
    }

    #[test]
    fn test_message_assistant_constructor() {
        let msg = Message::assistant("Hi there");
        assert_eq!(msg.role, Role::Assistant);
        assert_eq!(msg.text(), "Hi there");
    }

    #[test]
    fn test_message_tool_result_constructor() {
        let msg = Message::tool_result("call_1", "sunny, 72F", false);
        assert_eq!(msg.role, Role::Tool);
        match &msg.content[0] {
            ContentPart::ToolResult { tool_result } => {
                assert_eq!(tool_result.tool_call_id, "call_1");
                assert!(!tool_result.is_error);
            }
            _ => panic!("Expected ToolResult"),
        }
    }

    #[test]
    fn test_message_tool_result_error() {
        let msg = Message::tool_result("call_2", "network error", true);
        match &msg.content[0] {
            ContentPart::ToolResult { tool_result } => {
                assert!(tool_result.is_error);
            }
            _ => panic!("Expected ToolResult"),
        }
    }

    #[test]
    fn test_message_text_concatenation() {
        let msg = Message {
            role: Role::Assistant,
            content: vec![ContentPart::text("Hello "), ContentPart::text("world")],
            name: None,
            tool_call_id: None,
        };
        assert_eq!(msg.text(), "Hello world");
    }

    #[test]
    fn test_message_text_empty_when_no_text_parts() {
        let msg = Message {
            role: Role::User,
            content: vec![],
            name: None,
            tool_call_id: None,
        };
        assert_eq!(msg.text(), "");
    }

    #[test]
    fn test_message_text_ignores_non_text_parts() {
        let msg = Message {
            role: Role::Assistant,
            content: vec![
                ContentPart::text("Answer: "),
                ContentPart::image_url("https://example.com/img.png"),
                ContentPart::text("42"),
            ],
            name: None,
            tool_call_id: None,
        };
        assert_eq!(msg.text(), "Answer: 42");
    }

    #[test]
    fn test_message_serde_roundtrip() {
        let msg = Message::user("Hello");
        let json = serde_json::to_string(&msg).unwrap();
        let back: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(back.role, Role::User);
        assert_eq!(back.text(), "Hello");
    }

    #[test]
    fn test_message_optional_fields_omitted() {
        let msg = Message::user("Hello");
        let json = serde_json::to_string(&msg).unwrap();
        assert!(!json.contains("name"));
        assert!(!json.contains("tool_call_id"));
    }

    #[test]
    fn test_message_with_name() {
        let msg = Message {
            role: Role::User,
            content: vec![ContentPart::text("Hello")],
            name: Some("alice".to_string()),
            tool_call_id: None,
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"name\":\"alice\""));
        let back: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(back.name, Some("alice".to_string()));
    }
}
