// Error hierarchy — unified error type for the entire library.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Discriminator covering the full spec hierarchy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ErrorKind {
    // Provider errors (from HTTP responses)
    Authentication,
    AccessDenied,
    NotFound,
    InvalidRequest,
    RateLimit,
    Server,
    ContentFilter,
    ContextLength,
    QuotaExceeded,

    // Client-side errors
    RequestTimeout,
    Abort,
    Network,
    Stream,
    InvalidToolCall,
    NoObjectGenerated,
    UnsupportedToolChoice,
    Configuration,
}

impl ErrorKind {
    /// Returns `true` if this is a provider-originated error (spec §6.1 ProviderError subtypes).
    ///
    /// The spec defines `ProviderError` as a parent class with 9 subtypes. This flat enum
    /// bridges that hierarchy gap by providing a predicate to test group membership.
    pub fn is_provider_error(&self) -> bool {
        matches!(
            self,
            Self::Authentication
                | Self::AccessDenied
                | Self::NotFound
                | Self::InvalidRequest
                | Self::RateLimit
                | Self::Server
                | Self::ContentFilter
                | Self::ContextLength
                | Self::QuotaExceeded
        )
    }
}

/// The single error type for the entire library.
#[derive(Debug)]
pub struct Error {
    pub kind: ErrorKind,
    pub message: String,
    pub retryable: bool,
    pub source: Option<Box<dyn std::error::Error + Send + Sync>>,

    // Provider error fields
    pub provider: Option<String>,
    pub status_code: Option<u16>,
    pub error_code: Option<String>,
    pub retry_after: Option<Duration>,
    pub raw: Option<serde_json::Value>,
}

impl Error {
    /// Construct from HTTP status code (for provider adapters).
    pub fn from_http_status(
        status: u16,
        message: String,
        provider: &str,
        raw: Option<serde_json::Value>,
        retry_after: Option<Duration>,
    ) -> Self {
        let (kind, retryable) = match status {
            400 | 422 => (ErrorKind::InvalidRequest, false),
            401 => (ErrorKind::Authentication, false),
            403 => (ErrorKind::AccessDenied, false),
            404 => (ErrorKind::NotFound, false),
            408 => (ErrorKind::RequestTimeout, true),
            413 => (ErrorKind::ContextLength, false),
            429 => (ErrorKind::RateLimit, true),
            500..=599 => (ErrorKind::Server, true),
            _ => (ErrorKind::Server, true), // Unknown defaults to retryable
        };

        // Apply message-based classification for ambiguous cases.
        // Reclassification may change retryability (e.g. a 5xx with "context length"
        // in the message becomes ContextLength which is non-retryable).
        let kind = Self::classify_by_message(&message, kind);
        let retryable = match kind {
            ErrorKind::Authentication
            | ErrorKind::AccessDenied
            | ErrorKind::NotFound
            | ErrorKind::InvalidRequest
            | ErrorKind::ContextLength
            | ErrorKind::QuotaExceeded
            | ErrorKind::ContentFilter
            | ErrorKind::Configuration => false,
            _ => retryable,
        };

        Self {
            kind,
            message,
            retryable,
            source: None,
            provider: Some(provider.to_string()),
            status_code: Some(status),
            error_code: None,
            retry_after,
            raw,
        }
    }

    /// Convenience: configuration error.
    pub fn configuration(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::Configuration,
            message: message.into(),
            retryable: false,
            source: None,
            provider: None,
            status_code: None,
            error_code: None,
            retry_after: None,
            raw: None,
        }
    }

    /// Convenience: network error with source.
    pub fn network(
        message: impl Into<String>,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self {
            kind: ErrorKind::Network,
            message: message.into(),
            retryable: true,
            source: Some(Box::new(source)),
            provider: None,
            status_code: None,
            error_code: None,
            retry_after: None,
            raw: None,
        }
    }

    /// Convenience: abort error (cancelled via abort signal).
    pub fn abort() -> Self {
        Self {
            kind: ErrorKind::Abort,
            message: "operation aborted".into(),
            retryable: false,
            source: None,
            provider: None,
            status_code: None,
            error_code: None,
            retry_after: None,
            raw: None,
        }
    }

    /// Convenience: unsupported tool choice mode error.
    pub fn unsupported_tool_choice(mode: &str) -> Self {
        Self {
            kind: ErrorKind::UnsupportedToolChoice,
            message: format!("Unsupported tool choice mode: '{mode}'"),
            retryable: false,
            source: None,
            provider: None,
            status_code: None,
            error_code: None,
            retry_after: None,
            raw: None,
        }
    }

    /// Convenience: stream error with source.
    pub fn stream(
        message: impl Into<String>,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self {
            kind: ErrorKind::Stream,
            message: message.into(),
            retryable: true,
            source: Some(Box::new(source)),
            provider: None,
            status_code: None,
            error_code: None,
            retry_after: None,
            raw: None,
        }
    }

    /// Convenience: no object generated error (structured output parsing/validation failure).
    pub fn no_object_generated(reason: impl Into<String>, raw_text: impl Into<String>) -> Self {
        let reason = reason.into();
        let raw_text = raw_text.into();
        Self {
            kind: ErrorKind::NoObjectGenerated,
            message: format!(
                "Failed to generate structured output: {}. Raw text: {}",
                reason, raw_text
            ),
            retryable: false,
            source: None,
            provider: None,
            status_code: None,
            error_code: None,
            retry_after: None,
            raw: None,
        }
    }

    /// Convenience: invalid tool call error.
    pub fn invalid_tool_call(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::InvalidToolCall,
            message: message.into(),
            retryable: false,
            source: None,
            provider: None,
            status_code: None,
            error_code: None,
            retry_after: None,
            raw: None,
        }
    }

    /// Reclassify an error kind based on the error message body.
    /// Public so provider adapters can apply message-based reclassification
    /// after gRPC/provider-specific overrides.
    pub fn classify_by_message_pub(message: &str, default: ErrorKind) -> ErrorKind {
        Self::classify_by_message(message, default)
    }

    fn classify_by_message(message: &str, default: ErrorKind) -> ErrorKind {
        let lower = message.to_lowercase();
        if lower.contains("not found") || lower.contains("does not exist") {
            ErrorKind::NotFound
        } else if lower.contains("unauthorized")
            || lower.contains("invalid key")
            || lower.contains("api key not valid")
        {
            ErrorKind::Authentication
        } else if lower.contains("context length")
            || lower.contains("context window")
            || lower.contains("too many tokens")
        {
            ErrorKind::ContextLength
        } else if lower.contains("content filter")
            || lower.contains("safety")
            || lower.contains("blocked")
        {
            ErrorKind::ContentFilter
        } else if lower.contains("quota")
            || lower.contains("billing")
            || lower.contains("insufficient funds")
        {
            ErrorKind::QuotaExceeded
        } else {
            default
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source
            .as_ref()
            .map(|e| e.as_ref() as &(dyn std::error::Error + 'static))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- ErrorKind variant existence ---

    #[test]
    fn test_all_17_error_kinds_exist() {
        let kinds = vec![
            ErrorKind::Authentication,
            ErrorKind::AccessDenied,
            ErrorKind::NotFound,
            ErrorKind::InvalidRequest,
            ErrorKind::RateLimit,
            ErrorKind::Server,
            ErrorKind::ContentFilter,
            ErrorKind::ContextLength,
            ErrorKind::QuotaExceeded,
            ErrorKind::RequestTimeout,
            ErrorKind::Abort,
            ErrorKind::Network,
            ErrorKind::Stream,
            ErrorKind::InvalidToolCall,
            ErrorKind::NoObjectGenerated,
            ErrorKind::UnsupportedToolChoice,
            ErrorKind::Configuration,
        ];
        assert_eq!(kinds.len(), 17);
    }

    // --- from_http_status mapping ---

    #[test]
    fn test_error_from_http_status_401() {
        let err = Error::from_http_status(401, "Unauthorized".into(), "anthropic", None, None);
        assert_eq!(err.kind, ErrorKind::Authentication);
        assert!(!err.retryable);
        assert_eq!(err.provider, Some("anthropic".to_string()));
        assert_eq!(err.status_code, Some(401));
    }

    #[test]
    fn test_error_from_http_status_403() {
        let err = Error::from_http_status(403, "Forbidden".into(), "openai", None, None);
        assert_eq!(err.kind, ErrorKind::AccessDenied);
        assert!(!err.retryable);
    }

    #[test]
    fn test_error_from_http_status_404() {
        let err = Error::from_http_status(404, "Not Found".into(), "test", None, None);
        assert_eq!(err.kind, ErrorKind::NotFound);
        assert!(!err.retryable);
    }

    #[test]
    fn test_error_from_http_status_400() {
        let err = Error::from_http_status(400, "Bad Request".into(), "test", None, None);
        assert_eq!(err.kind, ErrorKind::InvalidRequest);
        assert!(!err.retryable);
    }

    #[test]
    fn test_error_from_http_status_422() {
        let err = Error::from_http_status(422, "Unprocessable".into(), "test", None, None);
        assert_eq!(err.kind, ErrorKind::InvalidRequest);
        assert!(!err.retryable);
    }

    #[test]
    fn test_error_from_http_status_429() {
        let err = Error::from_http_status(429, "Rate limited".into(), "openai", None, None);
        assert_eq!(err.kind, ErrorKind::RateLimit);
        assert!(err.retryable);
    }

    #[test]
    fn test_error_from_http_status_408() {
        let err = Error::from_http_status(408, "Timeout".into(), "test", None, None);
        assert_eq!(err.kind, ErrorKind::RequestTimeout);
        assert!(err.retryable);
    }

    #[test]
    fn test_error_from_http_status_500_to_599() {
        for status in [500, 501, 502, 503, 504, 550, 599] {
            let err = Error::from_http_status(status, "Server error".into(), "test", None, None);
            assert_eq!(err.kind, ErrorKind::Server, "status {status}");
            assert!(err.retryable, "status {status}");
        }
    }

    #[test]
    fn test_error_from_http_status_unknown_defaults_to_retryable() {
        let err = Error::from_http_status(999, "Unknown".into(), "test", None, None);
        assert_eq!(err.kind, ErrorKind::Server);
        assert!(err.retryable);
    }

    #[test]
    fn test_all_status_codes_map_correctly() {
        let cases = vec![
            (400, ErrorKind::InvalidRequest, false),
            (401, ErrorKind::Authentication, false),
            (403, ErrorKind::AccessDenied, false),
            (404, ErrorKind::NotFound, false),
            (408, ErrorKind::RequestTimeout, true),
            (422, ErrorKind::InvalidRequest, false),
            (429, ErrorKind::RateLimit, true),
            (500, ErrorKind::Server, true),
            (502, ErrorKind::Server, true),
            (503, ErrorKind::Server, true),
            (504, ErrorKind::Server, true),
        ];
        for (status, expected_kind, expected_retryable) in cases {
            let err = Error::from_http_status(status, "test".into(), "test", None, None);
            assert_eq!(err.kind, expected_kind, "status {status}");
            assert_eq!(err.retryable, expected_retryable, "status {status}");
        }
    }

    // --- Message-based classification ---

    #[test]
    fn test_message_classification_context_length() {
        let err =
            Error::from_http_status(400, "context length exceeded".into(), "test", None, None);
        assert_eq!(err.kind, ErrorKind::ContextLength);
    }

    #[test]
    fn test_message_classification_content_filter() {
        let err =
            Error::from_http_status(400, "content filter triggered".into(), "test", None, None);
        assert_eq!(err.kind, ErrorKind::ContentFilter);
    }

    // --- retry_after field ---

    #[test]
    fn test_error_from_http_status_with_retry_after() {
        let err = Error::from_http_status(
            429,
            "Rate limited".into(),
            "openai",
            None,
            Some(Duration::from_secs(5)),
        );
        assert_eq!(err.retry_after, Some(Duration::from_secs(5)));
    }

    #[test]
    fn test_error_from_http_status_with_raw() {
        let raw = serde_json::json!({"error": {"type": "rate_limit"}});
        let err = Error::from_http_status(
            429,
            "Rate limited".into(),
            "openai",
            Some(raw.clone()),
            None,
        );
        assert_eq!(err.raw, Some(raw));
    }

    // --- Display and std::error::Error ---

    #[test]
    fn test_error_display_output() {
        let err = Error::from_http_status(500, "Server error".into(), "test", None, None);
        let display = format!("{}", err);
        assert!(display.contains("Server"));
        assert!(display.contains("Server error"));
    }

    #[test]
    fn test_error_implements_std_error() {
        let err = Error::from_http_status(500, "Server error".into(), "test", None, None);
        let _: &dyn std::error::Error = &err;
    }

    #[test]
    fn test_error_source_chain() {
        let inner = std::io::Error::new(std::io::ErrorKind::ConnectionRefused, "refused");
        let err = Error::network("connection failed", inner);
        let source = std::error::Error::source(&err);
        assert!(source.is_some());
    }

    // --- Convenience constructors ---

    #[test]
    fn test_error_configuration() {
        let err = Error::configuration("missing API key");
        assert_eq!(err.kind, ErrorKind::Configuration);
        assert_eq!(err.message, "missing API key");
        assert!(!err.retryable);
    }

    #[test]
    fn test_error_network() {
        let inner = std::io::Error::new(std::io::ErrorKind::ConnectionRefused, "refused");
        let err = Error::network("connection failed", inner);
        assert_eq!(err.kind, ErrorKind::Network);
        assert!(err.retryable);
        assert!(err.source.is_some());
    }

    #[test]
    fn test_error_abort() {
        let err = Error::abort();
        assert_eq!(err.kind, ErrorKind::Abort);
        assert!(!err.retryable);
    }

    // --- D-3: Missing message classification patterns ---

    #[test]
    fn test_message_classification_context_window() {
        let err =
            Error::from_http_status(400, "context window exceeded".into(), "test", None, None);
        assert_eq!(err.kind, ErrorKind::ContextLength);
        assert!(!err.retryable);
    }

    #[test]
    fn test_message_classification_blocked() {
        let err = Error::from_http_status(
            400,
            "response blocked by safety filter".into(),
            "test",
            None,
            None,
        );
        assert_eq!(err.kind, ErrorKind::ContentFilter);
        assert!(!err.retryable);
    }

    #[test]
    fn test_message_classification_quota_exceeded() {
        let err = Error::from_http_status(400, "quota exceeded".into(), "test", None, None);
        assert_eq!(err.kind, ErrorKind::QuotaExceeded);
        assert!(!err.retryable);
    }

    #[test]
    fn test_message_classification_billing() {
        let err = Error::from_http_status(402, "billing limit reached".into(), "test", None, None);
        assert_eq!(err.kind, ErrorKind::QuotaExceeded);
        assert!(!err.retryable);
    }

    #[test]
    fn test_message_classification_insufficient_funds() {
        let err = Error::from_http_status(400, "insufficient funds".into(), "test", None, None);
        assert_eq!(err.kind, ErrorKind::QuotaExceeded);
        assert!(!err.retryable);
    }

    // --- Gemini 400 "API key not valid" should map to Authentication ---

    #[test]
    fn test_message_classification_api_key_not_valid() {
        // Gemini returns HTTP 400 (not 401) for invalid API keys with message
        // "API key not valid. Please pass a valid API key."
        let err = Error::from_http_status(
            400,
            "API key not valid. Please pass a valid API key.".into(),
            "gemini",
            None,
            None,
        );
        assert_eq!(
            err.kind,
            ErrorKind::Authentication,
            "HTTP 400 with 'API key not valid' should be reclassified to Authentication"
        );
        assert!(!err.retryable);
    }

    // --- FP-07: Error::stream() constructor ---

    #[test]
    fn test_error_stream_constructor() {
        let inner = std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "eof");
        let err = Error::stream("Stream read error", inner);
        assert_eq!(err.kind, ErrorKind::Stream);
        assert!(err.retryable);
        assert!(err.source.is_some());
        assert!(err.message.contains("Stream read error"));
    }

    // --- FP-08: Error::invalid_tool_call() constructor ---

    #[test]
    fn test_error_invalid_tool_call_constructor() {
        let err = Error::invalid_tool_call("malformed JSON arguments");
        assert_eq!(err.kind, ErrorKind::InvalidToolCall);
        assert!(!err.retryable);
        assert!(err.message.contains("malformed JSON"));
    }
}
