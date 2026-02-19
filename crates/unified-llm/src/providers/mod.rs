// Provider adapters — each behind a feature flag.

#[cfg(feature = "anthropic")]
pub mod anthropic;

#[cfg(feature = "openai")]
pub mod openai;

#[cfg(feature = "gemini")]
pub mod gemini;

#[cfg(feature = "openai-compat")]
pub mod openai_compat;
