// default_client.rs -- Module-level default client (Layer 3).
//
// Spec reference: S2.5 Module-Level Default Client
// DoD: 8.1.7 -- set_default_client() and implicit lazy initialization from env vars.
//
// Uses arc-swap for lock-free reads with atomic updates.

use std::sync::{Arc, Mutex};

use arc_swap::ArcSwap;
use std::sync::LazyLock;

use unified_llm_types::Error;

use crate::client::Client;

/// The global default client. Wraps `Option<Arc<Client>>`.
static DEFAULT_CLIENT: LazyLock<ArcSwap<Option<Arc<Client>>>> =
    LazyLock::new(|| ArcSwap::from_pointee(None));

/// Guard for the lazy-init path to prevent TOCTOU double-initialization.
static INIT_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

/// Set the module-level default client explicitly.
///
/// Spec S2.5: "Applications can override it: `set_default_client(my_client)`"
pub fn set_default_client(client: Client) {
    DEFAULT_CLIENT.store(Arc::new(Some(Arc::new(client))));
}

/// Get the default client, lazily initializing from environment variables if needed.
///
/// Returns `Arc<Client>` for cheap sharing. Returns `Err(ConfigurationError)` if
/// no client has been set and no API keys are found in the environment.
///
/// Spec S2.5: "This client is lazily initialized from environment variables on first use."
pub fn get_default_client() -> Result<Arc<Client>, Error> {
    // Fast path: already initialized (lock-free read)
    let guard = DEFAULT_CLIENT.load();
    if let Some(ref client) = **guard {
        return Ok(Arc::clone(client));
    }

    // Slow path: acquire lock and double-check to prevent TOCTOU race
    let _lock = INIT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let guard = DEFAULT_CLIENT.load();
    if let Some(ref client) = **guard {
        return Ok(Arc::clone(client));
    }

    // Lazy init from env (under lock)
    let client = Arc::new(Client::from_env()?);
    DEFAULT_CLIENT.store(Arc::new(Some(Arc::clone(&client))));
    Ok(client)
}

/// Reset the default client to None. For testing only.
#[cfg(any(test, feature = "testing"))]
pub fn reset_default_client() {
    DEFAULT_CLIENT.store(Arc::new(None));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::MockProvider;
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_set_and_get_default_client() {
        reset_default_client();

        let mock = MockProvider::new("test");
        let client = Client::builder()
            .provider("test", Box::new(mock))
            .build()
            .unwrap();

        set_default_client(client);
        let retrieved = get_default_client().unwrap();
        assert!(Arc::strong_count(&retrieved) >= 1);
    }

    #[test]
    #[serial]
    fn test_set_default_client_can_be_called_repeatedly() {
        reset_default_client();

        let mock1 = MockProvider::new("provider_a");
        let client1 = Client::builder()
            .provider("provider_a", Box::new(mock1))
            .build()
            .unwrap();
        set_default_client(client1);

        let mock2 = MockProvider::new("provider_b");
        let client2 = Client::builder()
            .provider("provider_b", Box::new(mock2))
            .build()
            .unwrap();
        set_default_client(client2);

        // Should not error -- ArcSwap allows re-setting
        let retrieved = get_default_client().unwrap();
        assert!(Arc::strong_count(&retrieved) >= 1);
    }

    #[test]
    #[serial]
    fn test_get_default_client_no_keys_returns_error() {
        reset_default_client();

        // Safety: Tests run serially via #[serial], no concurrent env access.
        unsafe {
            std::env::remove_var("ANTHROPIC_API_KEY");
            std::env::remove_var("OPENAI_API_KEY");
            std::env::remove_var("GEMINI_API_KEY");
            std::env::remove_var("GOOGLE_API_KEY");
        }

        let result = get_default_client();
        assert!(
            result.is_err(),
            "Should error when no API keys and no client set"
        );
    }

    #[test]
    #[serial]
    fn test_get_default_client_lazy_init_with_key() {
        reset_default_client();

        unsafe {
            std::env::remove_var("OPENAI_API_KEY");
            std::env::remove_var("GEMINI_API_KEY");
            std::env::remove_var("GOOGLE_API_KEY");
            std::env::set_var("ANTHROPIC_API_KEY", "test-key-12345");
        }

        let result = get_default_client();
        assert!(result.is_ok(), "Should lazily init from ANTHROPIC_API_KEY");

        unsafe {
            std::env::remove_var("ANTHROPIC_API_KEY");
        }
    }

    #[test]
    #[serial]
    fn test_reset_default_client_works() {
        reset_default_client();

        let mock = MockProvider::new("test");
        let client = Client::builder()
            .provider("test", Box::new(mock))
            .build()
            .unwrap();
        set_default_client(client);
        assert!(get_default_client().is_ok());

        reset_default_client();

        unsafe {
            std::env::remove_var("ANTHROPIC_API_KEY");
            std::env::remove_var("OPENAI_API_KEY");
            std::env::remove_var("GEMINI_API_KEY");
            std::env::remove_var("GOOGLE_API_KEY");
        }
        assert!(get_default_client().is_err());
    }
}
