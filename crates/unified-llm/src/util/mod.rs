pub(crate) mod http;
pub mod image;
pub(crate) mod provider_options;
pub mod retry;
pub(crate) mod sse;
pub mod stream_accumulator;
pub(crate) mod stream_lifecycle;

/// Normalize a base URL by stripping trailing slashes and common API path suffixes.
///
/// Ensures consistent base URLs regardless of how they're provided:
/// - Strips trailing slashes: `"https://api.example.com/"` → `"https://api.example.com"`
/// - Strips `/v1` suffix: `"https://api.openai.com/v1"` → `"https://api.openai.com"`
/// - Strips `/v1beta` suffix: `"https://generativelanguage.googleapis.com/v1beta"` → `"https://generativelanguage.googleapis.com"`
///
/// Adapters append their own versioned paths internally (e.g. `/v1/messages`),
/// so a base URL that already contains `/v1` would cause double-pathing.
pub fn normalize_base_url(url: &str) -> String {
    let mut url = url.trim_end_matches('/').to_string();
    for suffix in &["/v1beta", "/v1"] {
        if url.ends_with(suffix) {
            url.truncate(url.len() - suffix.len());
            break;
        }
    }
    // Strip any trailing slash left after suffix removal
    url.trim_end_matches('/').to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_base_url_no_op() {
        assert_eq!(
            normalize_base_url("https://api.anthropic.com"),
            "https://api.anthropic.com"
        );
    }

    #[test]
    fn test_normalize_strips_trailing_slash() {
        assert_eq!(
            normalize_base_url("https://api.anthropic.com/"),
            "https://api.anthropic.com"
        );
    }

    #[test]
    fn test_normalize_strips_v1() {
        assert_eq!(
            normalize_base_url("https://api.openai.com/v1"),
            "https://api.openai.com"
        );
    }

    #[test]
    fn test_normalize_strips_v1_with_trailing_slash() {
        assert_eq!(
            normalize_base_url("https://api.openai.com/v1/"),
            "https://api.openai.com"
        );
    }

    #[test]
    fn test_normalize_strips_v1beta() {
        assert_eq!(
            normalize_base_url("https://generativelanguage.googleapis.com/v1beta"),
            "https://generativelanguage.googleapis.com"
        );
    }

    #[test]
    fn test_normalize_preserves_other_paths() {
        assert_eq!(
            normalize_base_url("https://example.com/api/v2"),
            "https://example.com/api/v2"
        );
    }

    #[test]
    fn test_normalize_multiple_trailing_slashes() {
        assert_eq!(
            normalize_base_url("https://example.com///"),
            "https://example.com"
        );
    }
}
