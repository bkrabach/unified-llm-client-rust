use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::Error;
use crate::message::Message;
use crate::response::ResponseFormat;
use crate::tool::{ToolChoice, ToolDefinition};

/// The main request struct sent to providers.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Request {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDefinition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_options: Option<serde_json::Value>,
    /// Arbitrary key-value metadata attached to this request.
    /// Providers may forward this in their API calls where supported.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, String>>,
    /// Abort signal stub — will be replaced with CancellationToken in unified-llm crate.
    #[serde(skip)]
    pub abort_signal: Option<()>,
}

impl Request {
    /// Validate that the request has the minimum required fields.
    ///
    /// Returns `ConfigurationError` if:
    /// - `model` is empty or whitespace-only
    /// - `messages` is empty
    pub fn validate(&self) -> Result<(), Error> {
        if self.model.trim().is_empty() {
            return Err(Error::configuration("Request model must not be empty"));
        }
        if self.messages.is_empty() {
            return Err(Error::configuration("Request messages must not be empty"));
        }
        Ok(())
    }

    /// Builder-style setter for model.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Builder-style setter for messages.
    pub fn messages(mut self, messages: Vec<Message>) -> Self {
        self.messages = messages;
        self
    }

    /// Builder-style setter for provider.
    pub fn provider(mut self, provider: Option<String>) -> Self {
        self.provider = provider;
        self
    }

    /// Builder-style setter for tools.
    pub fn tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Builder-style setter for tool_choice.
    pub fn tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = Some(tool_choice);
        self
    }

    /// Builder-style setter for response_format.
    pub fn response_format(mut self, response_format: ResponseFormat) -> Self {
        self.response_format = Some(response_format);
        self
    }

    /// Builder-style setter for temperature.
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Builder-style setter for top_p.
    pub fn top_p(mut self, top_p: f64) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Builder-style setter for max_tokens.
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Builder-style setter for stop_sequences.
    pub fn stop_sequences(mut self, stop_sequences: Vec<String>) -> Self {
        self.stop_sequences = Some(stop_sequences);
        self
    }

    /// Builder-style setter for reasoning_effort.
    pub fn reasoning_effort(mut self, reasoning_effort: impl Into<String>) -> Self {
        self.reasoning_effort = Some(reasoning_effort.into());
        self
    }

    /// Builder-style setter for provider_options.
    pub fn provider_options(mut self, provider_options: Option<serde_json::Value>) -> Self {
        self.provider_options = provider_options;
        self
    }

    /// Builder-style setter for metadata.
    pub fn metadata(mut self, metadata: HashMap<String, String>) -> Self {
        self.metadata = Some(metadata);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_default() {
        let req = Request::default();
        assert!(req.model.is_empty());
        assert!(req.messages.is_empty());
        assert!(req.provider.is_none());
        assert!(req.tools.is_none());
        assert!(req.tool_choice.is_none());
        assert!(req.response_format.is_none());
        assert!(req.temperature.is_none());
        assert!(req.top_p.is_none());
        assert!(req.max_tokens.is_none());
        assert!(req.stop_sequences.is_none());
        assert!(req.reasoning_effort.is_none());
        assert!(req.provider_options.is_none());
        assert!(req.metadata.is_none());
        assert!(req.abort_signal.is_none());
    }

    #[test]
    fn test_request_builder_chain() {
        let req = Request::default()
            .model("claude-opus-4-6")
            .messages(vec![Message::user("Hello")])
            .temperature(0.7)
            .max_tokens(1024);
        assert_eq!(req.model, "claude-opus-4-6");
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.temperature, Some(0.7));
        assert_eq!(req.max_tokens, Some(1024));
    }

    #[test]
    fn test_request_builder_provider() {
        let req = Request::default()
            .model("gpt-4")
            .provider(Some("openai".into()));
        assert_eq!(req.provider, Some("openai".into()));
    }

    #[test]
    fn test_request_builder_tools() {
        let tools = vec![ToolDefinition {
            name: "get_weather".into(),
            description: "Get weather".into(),
            parameters: serde_json::json!({"type": "object"}),
            strict: None,
        }];
        let req = Request::default().model("test").tools(tools);
        assert_eq!(req.tools.as_ref().unwrap().len(), 1);
    }

    #[test]
    fn test_request_builder_stop_sequences() {
        let req = Request::default()
            .model("test")
            .stop_sequences(vec!["STOP".into(), "END".into()]);
        assert_eq!(req.stop_sequences.as_ref().unwrap().len(), 2);
    }

    #[test]
    fn test_request_builder_reasoning_effort() {
        let req = Request::default().model("test").reasoning_effort("high");
        assert_eq!(req.reasoning_effort, Some("high".into()));
    }

    #[test]
    fn test_request_builder_provider_options() {
        let req = Request::default()
            .model("test")
            .provider_options(Some(serde_json::json!({"anthropic": {"thinking": true}})));
        assert!(req.provider_options.is_some());
    }

    #[test]
    fn test_request_serde_roundtrip() {
        let req = Request::default()
            .model("claude-opus-4-6")
            .messages(vec![Message::user("Hello")])
            .temperature(0.7)
            .max_tokens(1024);
        let json = serde_json::to_string(&req).unwrap();
        let back: Request = serde_json::from_str(&json).unwrap();
        assert_eq!(back.model, "claude-opus-4-6");
        assert_eq!(back.messages.len(), 1);
        assert_eq!(back.temperature, Some(0.7));
        assert_eq!(back.max_tokens, Some(1024));
    }

    #[test]
    fn test_request_optional_fields_omitted_in_json() {
        let req = Request::default().model("test").messages(vec![]);
        let json = serde_json::to_string(&req).unwrap();
        assert!(!json.contains("provider_options"));
        assert!(!json.contains("temperature"));
        assert!(!json.contains("top_p"));
        assert!(!json.contains("max_tokens"));
        assert!(!json.contains("stop_sequences"));
        assert!(!json.contains("reasoning_effort"));
        assert!(!json.contains("tools"));
        assert!(!json.contains("tool_choice"));
        assert!(!json.contains("response_format"));
        assert!(!json.contains("system"));
        assert!(!json.contains("metadata"));
    }

    // --- M-12: Request.metadata field ---

    #[test]
    fn test_request_builder_metadata() {
        let mut meta = HashMap::new();
        meta.insert("user_id".to_string(), "u123".to_string());
        meta.insert("session".to_string(), "s456".to_string());
        let req = Request::default().model("test").metadata(meta);
        assert!(req.metadata.is_some());
        let m = req.metadata.unwrap();
        assert_eq!(m.get("user_id"), Some(&"u123".to_string()));
        assert_eq!(m.get("session"), Some(&"s456".to_string()));
    }

    #[test]
    fn test_request_metadata_serde_roundtrip() {
        let mut meta = HashMap::new();
        meta.insert("key".to_string(), "value".to_string());
        let req = Request::default()
            .model("test")
            .messages(vec![Message::user("Hi")])
            .metadata(meta);
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("metadata"));
        assert!(json.contains("key"));
        assert!(json.contains("value"));
        let back: Request = serde_json::from_str(&json).unwrap();
        assert_eq!(
            back.metadata.as_ref().unwrap().get("key"),
            Some(&"value".to_string())
        );
    }

    #[test]
    fn test_request_metadata_none_by_default() {
        let req = Request::default();
        assert!(req.metadata.is_none());
    }

    #[test]
    fn test_request_abort_signal_not_serialized() {
        let req = Request {
            abort_signal: Some(()),
            ..Default::default()
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(!json.contains("abort_signal"));
    }

    // --- GAP-4: Request::validate() tests ---

    #[test]
    fn test_validate_ok() {
        let req = Request::default()
            .model("gpt-4")
            .messages(vec![Message::user("Hello")]);
        assert!(req.validate().is_ok());
    }

    #[test]
    fn test_validate_empty_model() {
        let req = Request::default()
            .model("")
            .messages(vec![Message::user("Hello")]);
        let err = req.validate().unwrap_err();
        assert_eq!(err.kind, crate::ErrorKind::Configuration);
        assert!(err.message.contains("model"));
    }

    #[test]
    fn test_validate_whitespace_model() {
        let req = Request::default()
            .model("   ")
            .messages(vec![Message::user("Hello")]);
        let err = req.validate().unwrap_err();
        assert_eq!(err.kind, crate::ErrorKind::Configuration);
        assert!(err.message.contains("model"));
    }

    #[test]
    fn test_validate_empty_messages() {
        let req = Request::default().model("gpt-4");
        let err = req.validate().unwrap_err();
        assert_eq!(err.kind, crate::ErrorKind::Configuration);
        assert!(err.message.contains("messages"));
    }

    #[test]
    fn test_validate_default_request_fails() {
        let req = Request::default();
        assert!(req.validate().is_err());
    }
}
