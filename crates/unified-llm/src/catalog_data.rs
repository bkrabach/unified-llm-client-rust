// catalog_data.rs -- Model catalog with embedded JSON data (Layer 3).
//
// Spec reference: S2.9 Model Catalog
// DoD: 8.1.8

use std::sync::RwLock;

use std::sync::LazyLock;
use unified_llm_types::{Error, ModelInfo};

/// Embedded model catalog JSON, loaded at compile time.
static CATALOG_JSON: &str = include_str!("catalog.json");

/// Parsed catalog, lazily initialized on first access.
static CATALOG: LazyLock<Vec<ModelInfo>> =
    LazyLock::new(|| serde_json::from_str(CATALOG_JSON).expect("catalog.json must be valid JSON"));

/// Additional models loaded at runtime (merged or replaced catalogs).
static EXTRA_MODELS: LazyLock<RwLock<Vec<ModelInfo>>> = LazyLock::new(|| RwLock::new(Vec::new()));

/// Load a custom catalog from a JSON string.
/// The custom models are merged into the catalog (they take precedence
/// over built-in models with the same ID).
pub fn load_catalog_from_json(json: &str) -> Result<(), Error> {
    let models: Vec<ModelInfo> = serde_json::from_str(json)
        .map_err(|e| Error::configuration(format!("Invalid catalog JSON: {e}")))?;
    let mut extra = EXTRA_MODELS.write().unwrap_or_else(|e| e.into_inner());
    extra.extend(models);
    Ok(())
}

/// Merge additional models into the catalog at runtime.
/// Merged models take precedence over built-in models with the same ID.
/// GAP-4: Deduplicates by model `id` — newer entries replace older ones.
pub fn merge_catalog(models: Vec<ModelInfo>) {
    let mut extra = EXTRA_MODELS.write().unwrap_or_else(|e| e.into_inner());
    for model in models {
        if let Some(existing) = extra.iter_mut().find(|m| m.id == model.id) {
            *existing = model;
        } else {
            extra.push(model);
        }
    }
}

/// Look up a model by its exact `id` or by any of its `aliases`.
/// Returns `None` if no match is found. Unknown model strings are not errors --
/// callers can still pass them through to the provider (spec S2.9).
pub fn get_model_info(model_id: &str) -> Option<ModelInfo> {
    // Check runtime-loaded models first (they take precedence)
    if let Ok(extra) = EXTRA_MODELS.read() {
        if let Some(m) = extra.iter().find(|m| m.id == model_id) {
            return Some(m.clone());
        }
        if let Some(m) = extra
            .iter()
            .find(|m| m.aliases.iter().any(|a| a == model_id))
        {
            return Some(m.clone());
        }
    }

    // Fall back to built-in catalog
    if let Some(m) = CATALOG.iter().find(|m| m.id == model_id) {
        return Some(m.clone());
    }
    CATALOG
        .iter()
        .find(|m| m.aliases.iter().any(|a| a == model_id))
        .cloned()
}

/// Return all models in the catalog, optionally filtered by provider name.
/// When `provider` is `None`, returns the full catalog.
/// Runtime-loaded models are included and listed first.
pub fn list_models(provider: Option<&str>) -> Vec<ModelInfo> {
    let mut result = Vec::new();

    // Add runtime-loaded models first
    if let Ok(extra) = EXTRA_MODELS.read() {
        match provider {
            None => result.extend(extra.iter().cloned()),
            Some(p) => result.extend(extra.iter().filter(|m| m.provider == p).cloned()),
        }
    }

    // Add built-in catalog
    match provider {
        None => result.extend(CATALOG.iter().cloned()),
        Some(p) => result.extend(CATALOG.iter().filter(|m| m.provider == p).cloned()),
    }

    result
}

/// Return the first (newest/best) model for a given provider, optionally
/// filtered by capability. Capability values: `"tools"`, `"vision"`, `"reasoning"`.
/// The catalog JSON is ordered newest-first per provider, so the first match wins.
///
/// Spec S2.9: "prefer the latest models -- they are generally more capable."
pub fn get_latest_model(provider: &str, capability: Option<&str>) -> Option<ModelInfo> {
    let cap_filter = |m: &&ModelInfo| match capability {
        Some("tools") => m.supports_tools,
        Some("vision") => m.supports_vision,
        Some("reasoning") => m.supports_reasoning,
        Some(_) => false,
        None => true,
    };

    // Check runtime-loaded models first (they take precedence)
    if let Ok(extra) = EXTRA_MODELS.read() {
        if let Some(m) = extra
            .iter()
            .filter(|m| m.provider == provider)
            .find(cap_filter)
        {
            return Some(m.clone());
        }
    }

    // Fall back to built-in catalog
    CATALOG
        .iter()
        .filter(|m| m.provider == provider)
        .find(cap_filter)
        .cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_catalog_parses_successfully() {
        assert!(
            !CATALOG.is_empty(),
            "Catalog should contain at least one model"
        );
    }

    #[test]
    fn test_get_model_info_exact_match() {
        let info = get_model_info("claude-sonnet-4-20250514").unwrap();
        assert_eq!(info.id, "claude-sonnet-4-20250514");
        assert_eq!(info.provider, "anthropic");
        assert_eq!(info.display_name, "Claude Sonnet 4");
        assert!(info.supports_tools);
        assert!(info.supports_vision);
        assert!(info.supports_reasoning);
        assert_eq!(info.context_window, 200_000);
    }

    #[test]
    fn test_get_model_info_by_alias() {
        let info = get_model_info("sonnet").unwrap();
        assert!(
            info.id.contains("sonnet"),
            "Alias 'sonnet' should resolve to a Sonnet model, got: {}",
            info.id
        );
        assert_eq!(info.provider, "anthropic");
    }

    #[test]
    fn test_get_model_info_unknown_returns_none() {
        assert!(get_model_info("nonexistent-model-xyz").is_none());
    }

    #[test]
    fn test_list_models_all() {
        let models = list_models(None);
        assert!(
            models.len() >= 7,
            "Catalog should have at least 7 models (spec S2.9), got {}",
            models.len()
        );
        let providers: std::collections::HashSet<&str> =
            models.iter().map(|m| m.provider.as_str()).collect();
        assert!(providers.contains("anthropic"));
        assert!(providers.contains("openai"));
        assert!(providers.contains("gemini"));
    }

    #[test]
    fn test_list_models_by_provider() {
        let anthropic_models = list_models(Some("anthropic"));
        assert!(
            anthropic_models.len() >= 2,
            "Should have at least 2 Anthropic models"
        );
        for m in &anthropic_models {
            assert_eq!(m.provider, "anthropic");
        }

        let openai_models = list_models(Some("openai"));
        assert!(
            openai_models.len() >= 2,
            "Should have at least 2 OpenAI models"
        );
        for m in &openai_models {
            assert_eq!(m.provider, "openai");
        }
    }

    #[test]
    fn test_list_models_unknown_provider_returns_empty() {
        let models = list_models(Some("unknown_provider"));
        assert!(models.is_empty());
    }

    #[test]
    fn test_get_latest_model_anthropic() {
        let latest = get_latest_model("anthropic", None).unwrap();
        assert_eq!(latest.provider, "anthropic");
        assert_eq!(latest.id, "claude-opus-4-20250514");
    }

    #[test]
    fn test_get_latest_model_openai() {
        let latest = get_latest_model("openai", None).unwrap();
        assert_eq!(latest.provider, "openai");
    }

    #[test]
    fn test_get_latest_model_gemini() {
        let latest = get_latest_model("gemini", None).unwrap();
        assert_eq!(latest.provider, "gemini");
    }

    #[test]
    fn test_get_latest_model_with_capability_filter() {
        let latest_tools = get_latest_model("anthropic", Some("tools")).unwrap();
        assert!(latest_tools.supports_tools);

        let latest_vision = get_latest_model("openai", Some("vision")).unwrap();
        assert!(latest_vision.supports_vision);

        let latest_reasoning = get_latest_model("gemini", Some("reasoning")).unwrap();
        assert!(latest_reasoning.supports_reasoning);
    }

    #[test]
    fn test_get_latest_model_unknown_provider_returns_none() {
        assert!(get_latest_model("unknown_provider", None).is_none());
    }

    #[test]
    fn test_all_models_have_required_fields() {
        for model in list_models(None) {
            assert!(!model.id.is_empty(), "Model id must not be empty");
            assert!(
                !model.provider.is_empty(),
                "Model provider must not be empty"
            );
            assert!(
                !model.display_name.is_empty(),
                "Model display_name must not be empty"
            );
            assert!(
                model.context_window > 0,
                "Model context_window must be > 0 for {}",
                model.id
            );
        }
    }

    // --- AF-18: Runtime catalog loading ---

    #[test]
    fn test_load_catalog_from_json_valid() {
        let custom_json = r#"[{
            "id": "custom-test-model-af18",
            "provider": "custom_test",
            "display_name": "Custom Test Model",
            "context_window": 4096,
            "supports_tools": false,
            "supports_vision": false,
            "supports_reasoning": false,
            "aliases": [],
            "input_cost_per_million": 0.5,
            "output_cost_per_million": 1.5
        }]"#;
        let result = load_catalog_from_json(custom_json);
        assert!(result.is_ok(), "Valid JSON should load successfully");

        // Verify the custom model is findable via get_model_info
        let info = get_model_info("custom-test-model-af18");
        assert!(
            info.is_some(),
            "Custom model should be findable after loading"
        );
        let info = info.unwrap();
        assert_eq!(info.provider, "custom_test");
        // Verify cost fields are actually deserialized (not silently dropped)
        assert_eq!(info.input_cost_per_million, Some(0.5));
        assert_eq!(info.output_cost_per_million, Some(1.5));
    }

    #[test]
    fn test_load_catalog_from_json_invalid() {
        let bad_json = "not valid json";
        let result = load_catalog_from_json(bad_json);
        assert!(result.is_err(), "Invalid JSON should return error");
    }

    #[test]
    fn test_merge_catalog_adds_models() {
        let custom_json = r#"[{
            "id": "merge-test-model-af18",
            "provider": "merge_test",
            "display_name": "Merge Test Model",
            "context_window": 8192,
            "supports_tools": true,
            "supports_vision": false,
            "supports_reasoning": false,
            "aliases": ["merge-alias"],
            "input_cost_per_million": 1.0,
            "output_cost_per_million": 2.0
        }]"#;
        let models: Vec<ModelInfo> = serde_json::from_str(custom_json).unwrap();
        merge_catalog(models);

        // Verify the merged model is findable
        let info = get_model_info("merge-test-model-af18");
        assert!(info.is_some(), "Merged model should be findable");
        let info = info.unwrap();
        assert_eq!(info.context_window, 8192);
        // Verify cost fields are actually deserialized
        assert_eq!(info.input_cost_per_million, Some(1.0));
        assert_eq!(info.output_cost_per_million, Some(2.0));

        // Verify by alias too
        let info = get_model_info("merge-alias");
        assert!(info.is_some(), "Merged model should be findable by alias");
    }

    #[test]
    fn test_model_aliases_resolve_to_correct_provider() {
        // Common short aliases should resolve to real deployed models
        let cases = vec![
            ("opus", "anthropic", "claude-opus-4-20250514"),
            ("sonnet", "anthropic", "claude-sonnet-4-20250514"),
            ("haiku", "anthropic", "claude-3-5-haiku-20241022"),
            ("gemini-flash", "gemini", "gemini-2.5-flash-preview-05-20"),
            ("gemini-pro", "gemini", "gemini-2.5-pro-preview-05-06"),
            ("4o", "openai", "gpt-4o"),
            ("4o-mini", "openai", "gpt-4o-mini"),
        ];
        for (alias, expected_provider, expected_id) in cases {
            let info = get_model_info(alias);
            assert!(
                info.is_some(),
                "Alias '{}' should resolve to a model",
                alias
            );
            let info = info.unwrap();
            assert_eq!(
                info.provider, expected_provider,
                "Alias '{}' should map to provider '{}'",
                alias, expected_provider
            );
            assert_eq!(
                info.id, expected_id,
                "Alias '{}' should map to model '{}'",
                alias, expected_id
            );
        }
    }

    #[test]
    fn test_get_latest_model_finds_runtime_loaded_model() {
        // M-3: get_latest_model() should search EXTRA_MODELS, not just static CATALOG.
        let custom_json = r#"[{
            "id": "runtime-latest-test-model",
            "provider": "runtime_latest_test",
            "display_name": "Runtime Latest Test",
            "context_window": 16384,
            "supports_tools": true,
            "supports_vision": false,
            "supports_reasoning": false,
            "aliases": [],
            "input_cost_per_million": 0.0,
            "output_cost_per_million": 0.0
        }]"#;
        load_catalog_from_json(custom_json).unwrap();

        let latest = get_latest_model("runtime_latest_test", None);
        assert!(
            latest.is_some(),
            "get_latest_model should find runtime-loaded models"
        );
        assert_eq!(latest.unwrap().id, "runtime-latest-test-model");

        // Also verify capability filtering works on runtime models
        let with_tools = get_latest_model("runtime_latest_test", Some("tools"));
        assert!(
            with_tools.is_some(),
            "get_latest_model should find runtime models with capability filter"
        );

        let with_vision = get_latest_model("runtime_latest_test", Some("vision"));
        assert!(
            with_vision.is_none(),
            "runtime model doesn't support vision, should return None"
        );
    }

    #[test]
    fn test_all_builtin_models_have_cost_fields() {
        // DEFECT-3: Verify cost fields deserialize to non-None values for all built-in models.
        for model in CATALOG.iter() {
            assert!(
                model.input_cost_per_million.is_some(),
                "Model '{}' must have input_cost_per_million",
                model.id
            );
            assert!(
                model.output_cost_per_million.is_some(),
                "Model '{}' must have output_cost_per_million",
                model.id
            );
            // Costs should be non-negative
            assert!(
                model.input_cost_per_million.unwrap() >= 0.0,
                "Model '{}' input_cost_per_million must be >= 0",
                model.id
            );
            assert!(
                model.output_cost_per_million.unwrap() >= 0.0,
                "Model '{}' output_cost_per_million must be >= 0",
                model.id
            );
        }
    }

    #[test]
    fn test_catalog_advisory_not_restrictive() {
        // Verify unknown model IDs can still be used with Client (catalog doesn't block).
        // get_model_info returning None is NOT an error -- the caller can still proceed.
        let unknown = get_model_info("my-custom-fine-tuned-model");
        assert!(
            unknown.is_none(),
            "Unknown models should return None, not error"
        );
        // The Client does not consult the catalog; it passes model strings directly.
        // This test documents the advisory (not restrictive) nature of the catalog.
    }
}
