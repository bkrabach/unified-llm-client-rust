// Provider options utilities for provider adapters.

use serde_json::Value;

/// Extract provider-specific options from the nested `provider_options` map.
/// Returns `Some(Value)` if the provider key exists, `None` otherwise.
pub fn get_provider_options(opts: &Option<Value>, provider: &str) -> Option<Value> {
    opts.as_ref().and_then(|v| v.get(provider)).cloned()
}

/// Merge provider-specific options into a request body, filtering out internal-only keys.
///
/// `opts` should be the provider-specific options value (e.g., the result of
/// `get_provider_options()`). Each key in `opts` is inserted into `body` unless
/// it appears in `internal_keys`.
pub fn merge_provider_options(body: &mut Value, opts: &Value, internal_keys: &[&str]) {
    if let (Some(body_obj), Some(opts_obj)) = (body.as_object_mut(), opts.as_object()) {
        for (key, value) in opts_obj {
            if !internal_keys.contains(&key.as_str()) {
                body_obj.insert(key.clone(), value.clone());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_provider_options_present() {
        let opts = serde_json::json!({
            "anthropic": {"thinking": {"type": "enabled"}},
            "openai": {"store": true}
        });
        let result = get_provider_options(&Some(opts), "anthropic");
        assert!(result.is_some());
        assert!(result.unwrap().get("thinking").is_some());
    }

    #[test]
    fn test_get_provider_options_absent() {
        let opts = serde_json::json!({"anthropic": {}});
        let result = get_provider_options(&Some(opts), "openai");
        assert!(result.is_none());
    }

    #[test]
    fn test_get_provider_options_none_input() {
        let result = get_provider_options(&None, "anthropic");
        assert!(result.is_none());
    }

    #[test]
    fn test_merge_provider_options() {
        let mut body = serde_json::json!({"model": "test"});
        let opts = serde_json::json!({"store": true, "metadata": {"user": "test"}});
        let internal = &["auto_cache", "betas"];
        merge_provider_options(&mut body, &opts, internal);
        assert_eq!(body["store"], true);
        assert_eq!(body["metadata"]["user"], "test");
    }

    #[test]
    fn test_merge_provider_options_filters_internal_keys() {
        let mut body = serde_json::json!({"model": "test"});
        let opts =
            serde_json::json!({"auto_cache": false, "betas": ["beta1"], "real_param": "value"});
        let internal = &["auto_cache", "betas"];
        merge_provider_options(&mut body, &opts, internal);
        assert!(body.get("auto_cache").is_none());
        assert!(body.get("betas").is_none());
        assert_eq!(body["real_param"], "value");
    }

    #[test]
    fn test_merge_provider_options_overwrites_existing() {
        let mut body = serde_json::json!({"model": "test", "temperature": 0.5});
        let opts = serde_json::json!({"temperature": 0.9, "extra": true});
        merge_provider_options(&mut body, &opts, &[]);
        // Provider options should overwrite existing keys (escape hatch semantics)
        assert_eq!(body["temperature"], 0.9);
        assert_eq!(body["extra"], true);
    }
}
