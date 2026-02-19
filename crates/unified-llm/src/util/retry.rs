// Retry utility — exponential backoff with jitter for retryable operations.

use std::future::Future;
use std::time::Duration;

use rand::Rng;
use unified_llm_types::config::RetryPolicy;
use unified_llm_types::error::Error;

/// Execute an async operation with retry logic.
///
/// ## Retry behavior
/// - Exponential backoff: delay = base_delay * backoff_multiplier^attempt, capped at max_delay
/// - Jitter: delay * random(0.5, 1.5) when enabled, re-clamped to max_delay
/// - Respects `Retry-After` on Error: if <= max_delay use it as delay, if > max_delay raise immediately
/// - Non-retryable errors (401, 403, 404, etc.) raise immediately without retry
///
/// ## Retry budget
/// The retry budget is shared across all retryable error types within a single
/// operation (429 rate limits, 5xx server errors, timeouts). This is intentional:
/// the spec requires retries to be "transparent" to callers, and a shared budget
/// prevents runaway retries when multiple error types occur in sequence.
///
/// For separate rate-limit handling (e.g., proactive throttling before hitting
/// 429), use a middleware that inspects `RateLimitInfo` from response headers.
///
/// ## `on_retry` callback
/// If `policy.on_retry` is set, it is called before each retry sleep with
/// `(error, attempt_number, delay_duration)`.
pub async fn with_retry<F, Fut, T>(policy: &RetryPolicy, mut operation: F) -> Result<T, Error>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, Error>>,
{
    let mut last_error: Option<Error> = None;

    for attempt in 0..=policy.max_retries {
        match operation().await {
            Ok(value) => return Ok(value),
            Err(err) => {
                // Non-retryable errors raise immediately
                if !err.retryable {
                    return Err(err);
                }

                // If retry_after exceeds max_delay, raise immediately
                if let Some(retry_after) = &err.retry_after {
                    if retry_after.as_secs_f64() > policy.max_delay {
                        return Err(err);
                    }
                }

                // If we've exhausted all retries, return the last error
                if attempt >= policy.max_retries {
                    return Err(err);
                }

                // Calculate delay and sleep
                let delay = calculate_delay(policy, attempt, err.retry_after);
                // Call on_retry callback if configured
                if let Some(ref callback) = policy.on_retry {
                    callback(&err, attempt, delay);
                }
                tokio::time::sleep(delay).await;

                last_error = Some(err);
            }
        }
    }

    // Should not reach here, but just in case
    Err(last_error.unwrap_or_else(|| Error::configuration("retry loop ended unexpectedly")))
}

/// Calculate the delay for a given retry attempt.
///
/// Formula: base_delay * backoff_multiplier^attempt, capped at max_delay.
/// If retry_after is provided and <= max_delay, use it instead.
/// If jitter is enabled, multiply by random(0.5, 1.5).
pub(crate) fn calculate_delay(
    policy: &RetryPolicy,
    attempt: u32,
    retry_after: Option<Duration>,
) -> Duration {
    // If retry_after is provided, use it instead of calculated backoff
    if let Some(retry_after) = retry_after {
        return retry_after;
    }

    // Exponential backoff: base_delay * multiplier^attempt
    let delay_secs = policy.base_delay * policy.backoff_multiplier.powi(attempt as i32);

    // Cap at max_delay
    let delay_secs = delay_secs.min(policy.max_delay);

    // Apply jitter if enabled: delay * random(0.5, 1.5)
    let delay_secs = if policy.jitter {
        let mut rng = rand::thread_rng();
        let jitter_factor: f64 = rng.gen_range(0.5..=1.5);
        delay_secs * jitter_factor
    } else {
        delay_secs
    };

    let delay_secs = delay_secs.min(policy.max_delay); // re-clamp after jitter

    Duration::from_secs_f64(delay_secs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;
    use unified_llm_types::error::ErrorKind;

    #[tokio::test]
    async fn test_succeeds_on_first_try() {
        let policy = RetryPolicy::default();
        let result = with_retry(&policy, || async { Ok::<_, Error>(42) }).await;
        assert_eq!(result.unwrap(), 42);
    }

    #[tokio::test]
    async fn test_retries_on_retryable_error_then_succeeds() {
        let policy = RetryPolicy {
            max_retries: 3,
            base_delay: 0.001, // 1ms for fast tests
            max_delay: 1.0,
            backoff_multiplier: 2.0,
            jitter: false,
            on_retry: None,
        };
        let attempt = Arc::new(AtomicU32::new(0));
        let attempt_clone = attempt.clone();

        let result = with_retry(&policy, || {
            let attempt = attempt_clone.clone();
            async move {
                let n = attempt.fetch_add(1, Ordering::SeqCst);
                if n < 2 {
                    Err(Error::from_http_status(
                        500,
                        "Server error".into(),
                        "test",
                        None,
                        None,
                    ))
                } else {
                    Ok(42)
                }
            }
        })
        .await;

        assert_eq!(result.unwrap(), 42);
        assert_eq!(attempt.load(Ordering::SeqCst), 3); // 1 initial + 2 retries
    }

    #[tokio::test]
    async fn test_stops_on_non_retryable_error() {
        let policy = RetryPolicy {
            max_retries: 3,
            base_delay: 0.001,
            max_delay: 1.0,
            backoff_multiplier: 2.0,
            jitter: false,
            on_retry: None,
        };
        let attempt = Arc::new(AtomicU32::new(0));
        let attempt_clone = attempt.clone();

        let result: Result<i32, Error> = with_retry(&policy, || {
            let attempt = attempt_clone.clone();
            async move {
                attempt.fetch_add(1, Ordering::SeqCst);
                Err(Error::from_http_status(
                    401,
                    "Unauthorized".into(),
                    "test",
                    None,
                    None,
                ))
            }
        })
        .await;

        let err = result.unwrap_err();
        assert_eq!(err.kind, ErrorKind::Authentication);
        assert_eq!(attempt.load(Ordering::SeqCst), 1); // No retries
    }

    #[tokio::test]
    async fn test_respects_max_retries() {
        let policy = RetryPolicy {
            max_retries: 2,
            base_delay: 0.001,
            max_delay: 1.0,
            backoff_multiplier: 2.0,
            jitter: false,
            on_retry: None,
        };
        let attempt = Arc::new(AtomicU32::new(0));
        let attempt_clone = attempt.clone();

        let result: Result<i32, Error> = with_retry(&policy, || {
            let attempt = attempt_clone.clone();
            async move {
                attempt.fetch_add(1, Ordering::SeqCst);
                Err(Error::from_http_status(
                    500,
                    "Server error".into(),
                    "test",
                    None,
                    None,
                ))
            }
        })
        .await;

        assert!(result.is_err());
        assert_eq!(attempt.load(Ordering::SeqCst), 3); // 1 initial + 2 retries
    }

    #[tokio::test]
    async fn test_retry_after_exceeding_max_delay_raises_immediately() {
        let policy = RetryPolicy {
            max_retries: 3,
            base_delay: 0.001,
            max_delay: 5.0, // max 5 seconds
            backoff_multiplier: 2.0,
            jitter: false,
            on_retry: None,
        };
        let attempt = Arc::new(AtomicU32::new(0));
        let attempt_clone = attempt.clone();

        let result: Result<i32, Error> = with_retry(&policy, || {
            let attempt = attempt_clone.clone();
            async move {
                attempt.fetch_add(1, Ordering::SeqCst);
                // retry_after of 60 seconds > max_delay of 5 seconds
                Err(Error::from_http_status(
                    429,
                    "Rate limited".into(),
                    "test",
                    None,
                    Some(Duration::from_secs(60)),
                ))
            }
        })
        .await;

        let err = result.unwrap_err();
        assert_eq!(err.kind, ErrorKind::RateLimit);
        assert_eq!(attempt.load(Ordering::SeqCst), 1); // No retries — raised immediately
    }

    #[tokio::test]
    async fn test_retry_after_within_max_delay_is_respected() {
        let policy = RetryPolicy {
            max_retries: 2,
            base_delay: 0.001,
            max_delay: 60.0,
            backoff_multiplier: 2.0,
            jitter: false,
            on_retry: None,
        };
        let attempt = Arc::new(AtomicU32::new(0));
        let attempt_clone = attempt.clone();

        let result = with_retry(&policy, || {
            let attempt = attempt_clone.clone();
            async move {
                let n = attempt.fetch_add(1, Ordering::SeqCst);
                if n == 0 {
                    // First attempt: retryable with small retry_after
                    Err(Error::from_http_status(
                        429,
                        "Rate limited".into(),
                        "test",
                        None,
                        Some(Duration::from_millis(1)),
                    ))
                } else {
                    Ok(42)
                }
            }
        })
        .await;

        assert_eq!(result.unwrap(), 42);
        assert_eq!(attempt.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_calculate_delay_exponential_backoff() {
        let policy = RetryPolicy {
            max_retries: 5,
            base_delay: 1.0,
            max_delay: 60.0,
            backoff_multiplier: 2.0,
            jitter: false,
            on_retry: None,
        };

        // attempt 0: 1.0 * 2^0 = 1.0
        let d0 = calculate_delay(&policy, 0, None);
        assert!(
            (d0.as_secs_f64() - 1.0).abs() < 0.01,
            "expected ~1.0, got {}",
            d0.as_secs_f64()
        );

        // attempt 1: 1.0 * 2^1 = 2.0
        let d1 = calculate_delay(&policy, 1, None);
        assert!(
            (d1.as_secs_f64() - 2.0).abs() < 0.01,
            "expected ~2.0, got {}",
            d1.as_secs_f64()
        );

        // attempt 2: 1.0 * 2^2 = 4.0
        let d2 = calculate_delay(&policy, 2, None);
        assert!(
            (d2.as_secs_f64() - 4.0).abs() < 0.01,
            "expected ~4.0, got {}",
            d2.as_secs_f64()
        );
    }

    #[test]
    fn test_calculate_delay_capped_at_max_delay() {
        let policy = RetryPolicy {
            max_retries: 10,
            base_delay: 1.0,
            max_delay: 10.0,
            backoff_multiplier: 2.0,
            jitter: false,
            on_retry: None,
        };

        // attempt 5: 1.0 * 2^5 = 32.0, capped to 10.0
        let d = calculate_delay(&policy, 5, None);
        assert!(
            (d.as_secs_f64() - 10.0).abs() < 0.01,
            "expected ~10.0, got {}",
            d.as_secs_f64()
        );
    }

    #[test]
    fn test_calculate_delay_with_retry_after() {
        let policy = RetryPolicy {
            max_retries: 3,
            base_delay: 1.0,
            max_delay: 60.0,
            backoff_multiplier: 2.0,
            jitter: false,
            on_retry: None,
        };

        // retry_after of 5s should override calculated delay
        let d = calculate_delay(&policy, 0, Some(Duration::from_secs(5)));
        assert!(
            (d.as_secs_f64() - 5.0).abs() < 0.01,
            "expected ~5.0, got {}",
            d.as_secs_f64()
        );
    }

    #[test]
    fn test_calculate_delay_with_jitter() {
        let policy = RetryPolicy {
            max_retries: 3,
            base_delay: 1.0,
            max_delay: 60.0,
            backoff_multiplier: 2.0,
            jitter: true,
            on_retry: None,
        };

        // With jitter, delay should be in range [0.5, 1.5] * base
        for _ in 0..20 {
            let d = calculate_delay(&policy, 0, None);
            assert!(d.as_secs_f64() >= 0.5, "delay too low: {}", d.as_secs_f64());
            assert!(
                d.as_secs_f64() <= 1.5,
                "delay too high: {}",
                d.as_secs_f64()
            );
        }
    }

    #[test]
    fn test_calculate_delay_with_jitter_never_exceeds_max_delay() {
        let policy = RetryPolicy {
            max_retries: 10,
            base_delay: 10.0,
            max_delay: 10.0,
            backoff_multiplier: 2.0,
            jitter: true,
            on_retry: None,
        };
        // Run 100 iterations — with jitter range [0.5, 1.5], the pre-fix code
        // will produce values up to 15.0 (10.0 * 1.5), exceeding max_delay.
        for attempt in 0..100 {
            let delay = calculate_delay(&policy, attempt % 5, None);
            assert!(
                delay <= Duration::from_secs_f64(10.0),
                "Delay {:?} exceeded max_delay of 10.0s on attempt {}",
                delay,
                attempt
            );
        }
    }

    #[tokio::test]
    async fn test_on_retry_callback_called() {
        use std::sync::{Arc, Mutex};
        let calls: Arc<Mutex<Vec<(u32, u64)>>> = Arc::new(Mutex::new(Vec::new()));
        let calls_clone = calls.clone();

        let policy = RetryPolicy {
            max_retries: 2,
            base_delay: 0.01, // very short for test speed
            max_delay: 1.0,
            backoff_multiplier: 1.0,
            jitter: false,
            on_retry: Some(Arc::new(move |_err, attempt, delay| {
                calls_clone
                    .lock()
                    .unwrap()
                    .push((attempt, delay.as_millis() as u64));
            })),
        };

        let counter = Arc::new(std::sync::Mutex::new(0u32));
        let counter_clone = counter.clone();

        let result = with_retry(&policy, || {
            let counter = counter_clone.clone();
            async move {
                let mut c = counter.lock().unwrap();
                *c += 1;
                if *c < 3 {
                    Err(Error::from_http_status(
                        500,
                        "server error".into(),
                        "test",
                        None,
                        None,
                    ))
                } else {
                    Ok("success")
                }
            }
        })
        .await;

        assert!(result.is_ok());
        let recorded = calls.lock().unwrap();
        assert_eq!(recorded.len(), 2); // Called twice (attempts 0 and 1)
        assert_eq!(recorded[0].0, 0);
        assert_eq!(recorded[1].0, 1);
    }
}
