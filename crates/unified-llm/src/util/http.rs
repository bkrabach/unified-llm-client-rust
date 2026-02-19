// Shared HTTP utilities for provider adapters.

use std::time::Duration;

use reqwest::header::HeaderMap;
use serde_json::Value;

/// Parse the `Retry-After` header as either numeric seconds or HTTP-date (RFC 7231).
/// Returns `None` if the header is missing, cannot be parsed, or the date is in the past.
pub fn parse_retry_after(headers: &HeaderMap) -> Option<Duration> {
    let value = headers.get("retry-after")?.to_str().ok()?;

    // Try numeric seconds first (most common for APIs)
    // PANIC-1: guard against negative, NaN, and infinity values that would
    // panic in Duration::from_secs_f64()
    if let Ok(secs) = value.parse::<f64>() {
        if secs >= 0.0 && secs.is_finite() {
            return Some(Duration::from_secs_f64(secs));
        }
        return None;
    }

    // Try HTTP-date format (RFC 7231)
    if let Ok(date) = httpdate::parse_http_date(value) {
        let now = std::time::SystemTime::now();
        if let Ok(duration) = date.duration_since(now) {
            return Some(duration);
        }
        // Date is in the past — return None
    }

    None
}

/// Walk a nested JSON value using an array of string keys.
/// Returns `Some(&Value)` at the end of the path, or `None` if any key is missing.
pub fn extract_json_path<'a>(value: &'a Value, path: &[&str]) -> Option<&'a Value> {
    let mut current = value;
    for key in path {
        current = current.get(*key)?;
    }
    Some(current)
}

/// Extract an error message and optional error code from a JSON body
/// using configurable JSON paths.
///
/// Returns `(message, Option<code>)`. If the message path doesn't resolve,
/// falls back to the full JSON body serialized as a string.
pub fn parse_provider_error_message(
    body: &Value,
    message_path: &[&str],
    code_path: &[&str],
) -> (String, Option<String>) {
    let message = extract_json_path(body, message_path)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| body.to_string());

    let code = extract_json_path(body, code_path)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    (message, code)
}

/// Parse rate limit information from HTTP response headers.
/// Returns `None` if no rate limit headers are present.
/// Handles the `x-ratelimit-*` header convention used by OpenAI, Anthropic, etc.
pub fn parse_rate_limit_headers(headers: &HeaderMap) -> Option<unified_llm_types::RateLimitInfo> {
    let get_u32 = |name: &str| -> Option<u32> {
        headers
            .get(name)
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u32>().ok())
    };
    let get_str = |name: &str| -> Option<String> {
        headers
            .get(name)
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string())
    };

    let requests_remaining = get_u32("x-ratelimit-remaining-requests");
    let requests_limit = get_u32("x-ratelimit-limit-requests");
    let tokens_remaining = get_u32("x-ratelimit-remaining-tokens");
    let tokens_limit = get_u32("x-ratelimit-limit-tokens");
    let reset_at = get_str("x-ratelimit-reset-requests")
        .or_else(|| get_str("x-ratelimit-reset-tokens"))
        .or_else(|| get_str("x-ratelimit-reset"));

    // Only return Some if at least one header was found
    if requests_remaining.is_none()
        && requests_limit.is_none()
        && tokens_remaining.is_none()
        && tokens_limit.is_none()
        && reset_at.is_none()
    {
        return None;
    }

    Some(unified_llm_types::RateLimitInfo {
        requests_remaining,
        requests_limit,
        tokens_remaining,
        tokens_limit,
        reset_at,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- parse_retry_after tests ---

    #[test]
    fn test_parse_retry_after_valid_integer() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("retry-after", "30".parse().unwrap());
        let result = parse_retry_after(&headers);
        assert_eq!(result, Some(std::time::Duration::from_secs(30)));
    }

    #[test]
    fn test_parse_retry_after_valid_float() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("retry-after", "1.5".parse().unwrap());
        let result = parse_retry_after(&headers);
        assert_eq!(result, Some(std::time::Duration::from_millis(1500)));
    }

    #[test]
    fn test_parse_retry_after_missing() {
        let headers = reqwest::header::HeaderMap::new();
        assert_eq!(parse_retry_after(&headers), None);
    }

    #[test]
    fn test_parse_retry_after_invalid() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("retry-after", "not-a-number".parse().unwrap());
        assert_eq!(parse_retry_after(&headers), None);
    }

    // --- PANIC-1: Negative/NaN/Infinity Retry-After guard ---

    #[test]
    fn test_parse_retry_after_negative_returns_none() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("retry-after", "-1".parse().unwrap());
        assert_eq!(
            parse_retry_after(&headers),
            None,
            "Negative retry-after should return None"
        );
    }

    #[test]
    fn test_parse_retry_after_nan_returns_none() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("retry-after", "NaN".parse().unwrap());
        assert_eq!(
            parse_retry_after(&headers),
            None,
            "NaN retry-after should return None"
        );
    }

    #[test]
    fn test_parse_retry_after_infinity_returns_none() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("retry-after", "inf".parse().unwrap());
        assert_eq!(
            parse_retry_after(&headers),
            None,
            "Infinity retry-after should return None"
        );
    }

    #[test]
    fn test_parse_retry_after_zero_returns_some() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("retry-after", "0".parse().unwrap());
        let duration = parse_retry_after(&headers).unwrap();
        assert_eq!(duration, Duration::from_secs(0));
    }

    // --- AF-20: HTTP-date Retry-After parsing ---

    #[test]
    fn test_parse_retry_after_http_date_future() {
        let mut headers = reqwest::header::HeaderMap::new();
        // Use a date 60 seconds in the future
        let future = std::time::SystemTime::now() + std::time::Duration::from_secs(60);
        let date_str = httpdate::fmt_http_date(future);
        headers.insert("retry-after", date_str.parse().unwrap());

        let result = parse_retry_after(&headers);
        assert!(result.is_some(), "Should parse HTTP-date format");
        let duration = result.unwrap();
        // Should be approximately 60 seconds (within tolerance)
        assert!(
            duration.as_secs() >= 55 && duration.as_secs() <= 65,
            "Duration should be ~60s, got {:?}",
            duration
        );
    }

    #[test]
    fn test_parse_retry_after_http_date_in_past() {
        let mut headers = reqwest::header::HeaderMap::new();
        // A date in the past should return None
        headers.insert(
            "retry-after",
            "Thu, 01 Jan 2020 00:00:00 GMT".parse().unwrap(),
        );
        let result = parse_retry_after(&headers);
        // Past dates should be handled gracefully — return None
        assert!(
            result.is_none(),
            "Past HTTP-date should return None, got {:?}",
            result
        );
    }

    #[test]
    fn test_parse_retry_after_numeric_still_works() {
        // Verify existing numeric parsing is preserved
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("retry-after", "5".parse().unwrap());
        let result = parse_retry_after(&headers);
        assert_eq!(result, Some(std::time::Duration::from_secs(5)));
    }

    // --- parse_provider_error_message tests ---

    #[test]
    fn test_parse_provider_error_anthropic_format() {
        let body = serde_json::json!({
            "type": "error",
            "error": {"type": "invalid_request_error", "message": "bad request"}
        });
        let (msg, code) =
            parse_provider_error_message(&body, &["error", "message"], &["error", "type"]);
        assert_eq!(msg, "bad request");
        assert_eq!(code, Some("invalid_request_error".into()));
    }

    #[test]
    fn test_parse_provider_error_openai_format() {
        let body = serde_json::json!({
            "error": {"message": "model not found", "code": "model_not_found"}
        });
        let (msg, code) =
            parse_provider_error_message(&body, &["error", "message"], &["error", "code"]);
        assert_eq!(msg, "model not found");
        assert_eq!(code, Some("model_not_found".into()));
    }

    #[test]
    fn test_parse_provider_error_fallback() {
        let body = serde_json::json!({"unexpected": "format"});
        let (msg, code) =
            parse_provider_error_message(&body, &["error", "message"], &["error", "code"]);
        assert!(msg.contains("unexpected")); // Falls back to full body as string
        assert_eq!(code, None);
    }

    // --- extract_json_path tests ---

    #[test]
    fn test_extract_json_path_nested() {
        let val = serde_json::json!({"a": {"b": {"c": "deep"}}});
        let result = extract_json_path(&val, &["a", "b", "c"]);
        assert_eq!(result.unwrap().as_str().unwrap(), "deep");
    }

    #[test]
    fn test_extract_json_path_missing() {
        let val = serde_json::json!({"a": 1});
        assert!(extract_json_path(&val, &["a", "b"]).is_none());
    }

    #[test]
    fn test_extract_json_path_empty() {
        let val = serde_json::json!({"a": 1});
        assert_eq!(extract_json_path(&val, &[]).unwrap(), &val);
    }

    // --- parse_rate_limit_headers tests ---

    #[test]
    fn test_parse_rate_limit_headers_all_present() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("x-ratelimit-remaining-requests", "95".parse().unwrap());
        headers.insert("x-ratelimit-limit-requests", "100".parse().unwrap());
        headers.insert("x-ratelimit-remaining-tokens", "9500".parse().unwrap());
        headers.insert("x-ratelimit-limit-tokens", "10000".parse().unwrap());
        headers.insert(
            "x-ratelimit-reset-requests",
            "2025-01-01T00:00:30Z".parse().unwrap(),
        );

        let info = parse_rate_limit_headers(&headers);
        assert!(info.is_some());
        let info = info.unwrap();
        assert_eq!(info.requests_remaining, Some(95));
        assert_eq!(info.requests_limit, Some(100));
        assert_eq!(info.tokens_remaining, Some(9500));
        assert_eq!(info.tokens_limit, Some(10000));
        assert_eq!(info.reset_at, Some("2025-01-01T00:00:30Z".to_string()));
    }

    #[test]
    fn test_parse_rate_limit_headers_none_present() {
        let headers = reqwest::header::HeaderMap::new();
        let info = parse_rate_limit_headers(&headers);
        assert!(info.is_none());
    }

    #[test]
    fn test_parse_rate_limit_headers_partial() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("x-ratelimit-remaining-requests", "50".parse().unwrap());
        // Only one header present — should still return Some

        let info = parse_rate_limit_headers(&headers);
        assert!(info.is_some());
        let info = info.unwrap();
        assert_eq!(info.requests_remaining, Some(50));
        assert_eq!(info.requests_limit, None);
        assert_eq!(info.tokens_remaining, None);
    }
}
