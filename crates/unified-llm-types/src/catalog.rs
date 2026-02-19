use serde::{Deserialize, Serialize};

/// Information about a model available in the catalog.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model identifier (e.g., "claude-opus-4-6").
    pub id: String,
    /// Provider name (e.g., "anthropic").
    pub provider: String,
    /// Human-readable display name.
    pub display_name: String,
    /// Context window size in tokens.
    pub context_window: u32,
    /// Maximum output tokens (if known).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output: Option<u32>,
    /// Whether the model supports tool calling.
    #[serde(default)]
    pub supports_tools: bool,
    /// Whether the model supports vision/image input.
    #[serde(default)]
    pub supports_vision: bool,
    /// Whether the model supports reasoning/thinking.
    #[serde(default)]
    pub supports_reasoning: bool,
    /// Input cost per million tokens (USD).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_cost_per_million: Option<f64>,
    /// Output cost per million tokens (USD).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_cost_per_million: Option<f64>,
    /// Alternative names for this model.
    #[serde(default)]
    pub aliases: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_info_construction() {
        let info = ModelInfo {
            id: "claude-opus-4-6".into(),
            provider: "anthropic".into(),
            display_name: "Claude Opus 4.6".into(),
            context_window: 200_000,
            max_output: None,
            supports_tools: true,
            supports_vision: true,
            supports_reasoning: true,
            input_cost_per_million: Some(15.0),
            output_cost_per_million: Some(75.0),
            aliases: vec![],
        };
        assert_eq!(info.id, "claude-opus-4-6");
        assert_eq!(info.provider, "anthropic");
        assert_eq!(info.context_window, 200_000);
        assert!(info.supports_tools);
        assert!(info.supports_vision);
        assert!(info.supports_reasoning);
    }

    #[test]
    fn test_model_info_serde_roundtrip() {
        let info = ModelInfo {
            id: "gpt-4o".into(),
            provider: "openai".into(),
            display_name: "GPT-4o".into(),
            context_window: 128_000,
            max_output: Some(16_384),
            supports_tools: true,
            supports_vision: true,
            supports_reasoning: false,
            input_cost_per_million: Some(5.0),
            output_cost_per_million: Some(15.0),
            aliases: vec!["gpt4o".into()],
        };
        let json = serde_json::to_string(&info).unwrap();
        let back: ModelInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, "gpt-4o");
        assert_eq!(back.max_output, Some(16_384));
        assert_eq!(back.aliases, vec!["gpt4o".to_string()]);
    }

    #[test]
    fn test_model_info_optional_fields_omitted() {
        let info = ModelInfo {
            id: "test".into(),
            provider: "test".into(),
            display_name: "Test".into(),
            context_window: 4096,
            max_output: None,
            supports_tools: false,
            supports_vision: false,
            supports_reasoning: false,
            input_cost_per_million: None,
            output_cost_per_million: None,
            aliases: vec![],
        };
        let json = serde_json::to_string(&info).unwrap();
        assert!(!json.contains("max_output"));
        assert!(!json.contains("input_cost_per_million"));
        assert!(!json.contains("output_cost_per_million"));
    }
}
