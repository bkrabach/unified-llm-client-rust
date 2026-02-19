use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Type alias for the on_retry callback to reduce complexity.
pub type OnRetryCallback = Arc<dyn Fn(&crate::error::Error, u32, Duration) + Send + Sync>;

/// Retry policy configuration for failed requests.
#[derive(Serialize, Deserialize)]
pub struct RetryPolicy {
    /// Maximum number of retry attempts (default: 2).
    pub max_retries: u32,
    /// Base delay in seconds before first retry (default: 1.0).
    pub base_delay: f64,
    /// Maximum delay in seconds between retries (default: 60.0).
    pub max_delay: f64,
    /// Multiplier for exponential backoff (default: 2.0).
    pub backoff_multiplier: f64,
    /// Whether to add random jitter to delays (default: true).
    pub jitter: bool,
    /// Called before each retry attempt with (error, attempt_number, delay).
    /// Useful for logging, metrics, or custom backoff logic.
    #[serde(skip)]
    pub on_retry: Option<OnRetryCallback>,
}

impl Clone for RetryPolicy {
    fn clone(&self) -> Self {
        Self {
            max_retries: self.max_retries,
            base_delay: self.base_delay,
            max_delay: self.max_delay,
            backoff_multiplier: self.backoff_multiplier,
            jitter: self.jitter,
            on_retry: self.on_retry.clone(), // Arc::clone is cheap
        }
    }
}

impl std::fmt::Debug for RetryPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RetryPolicy")
            .field("max_retries", &self.max_retries)
            .field("base_delay", &self.base_delay)
            .field("max_delay", &self.max_delay)
            .field("backoff_multiplier", &self.backoff_multiplier)
            .field("jitter", &self.jitter)
            .field("on_retry", &self.on_retry.as_ref().map(|_| "..."))
            .finish()
    }
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 2,
            base_delay: 1.0,
            max_delay: 60.0,
            backoff_multiplier: 2.0,
            jitter: true,
            on_retry: None,
        }
    }
}

/// Timeout configuration for operations.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TimeoutConfig {
    /// Total timeout for the entire operation in seconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total: Option<f64>,
    /// Timeout per individual step in seconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub per_step: Option<f64>,
}

impl From<f64> for TimeoutConfig {
    fn from(total: f64) -> Self {
        Self {
            total: Some(total),
            per_step: None,
        }
    }
}

/// Adapter-level timeout configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterTimeout {
    /// Connection timeout in seconds (default: 10.0).
    pub connect: f64,
    /// Request timeout in seconds (default: 120.0).
    pub request: f64,
    /// Stream read timeout in seconds (default: 30.0).
    pub stream_read: f64,
}

impl Default for AdapterTimeout {
    fn default() -> Self {
        Self {
            connect: 10.0,
            request: 120.0,
            stream_read: 30.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retry_policy_defaults() {
        let p = RetryPolicy::default();
        assert_eq!(p.max_retries, 2);
        assert_eq!(p.base_delay, 1.0);
        assert_eq!(p.max_delay, 60.0);
        assert_eq!(p.backoff_multiplier, 2.0);
        assert!(p.jitter);
    }

    #[test]
    fn test_retry_policy_serde_roundtrip() {
        let p = RetryPolicy {
            max_retries: 5,
            base_delay: 0.5,
            max_delay: 30.0,
            backoff_multiplier: 3.0,
            jitter: false,
            on_retry: None,
        };
        let json = serde_json::to_string(&p).unwrap();
        let back: RetryPolicy = serde_json::from_str(&json).unwrap();
        assert_eq!(back.max_retries, 5);
        assert_eq!(back.base_delay, 0.5);
        assert!(!back.jitter);
    }

    #[test]
    fn test_retry_policy_on_retry_default_none() {
        let p = RetryPolicy::default();
        assert!(p.on_retry.is_none());
    }

    #[test]
    fn test_retry_policy_serde_skips_on_retry() {
        let p = RetryPolicy {
            on_retry: Some(Arc::new(|_, _, _| {})),
            ..Default::default()
        };
        let json = serde_json::to_string(&p).unwrap();
        assert!(!json.contains("on_retry"));
        let back: RetryPolicy = serde_json::from_str(&json).unwrap();
        assert!(back.on_retry.is_none()); // Deserialized without callback
    }

    #[test]
    fn test_timeout_config_defaults() {
        let t = TimeoutConfig::default();
        assert!(t.total.is_none());
        assert!(t.per_step.is_none());
    }

    #[test]
    fn test_adapter_timeout_defaults() {
        let t = AdapterTimeout::default();
        assert_eq!(t.connect, 10.0);
        assert_eq!(t.request, 120.0);
        assert_eq!(t.stream_read, 30.0);
    }

    #[test]
    fn test_timeout_config_from_f64() {
        let tc = TimeoutConfig::from(30.0);
        assert_eq!(tc.total, Some(30.0));
        assert_eq!(tc.per_step, None);
    }
}
