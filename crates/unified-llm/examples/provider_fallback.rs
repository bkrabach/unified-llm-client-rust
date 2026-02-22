/// Provider fallback pattern (spec Appendix B.5).
///
/// Try the primary provider first; if it fails with a retryable or
/// provider-side error, fall back to a secondary provider.
/// Uses ErrorKind matching to decide when fallback is appropriate.
///
/// Requires at least two API keys (e.g. ANTHROPIC_API_KEY + OPENAI_API_KEY).
use unified_llm::{Client, ErrorKind, Message, Request};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::from_env()?;

    let prompt = vec![Message::user("Explain quicksort in two sentences.")];

    // Try primary provider (Anthropic).
    let primary = Request::default()
        .model("claude-sonnet-4-20250514")
        .provider(Some("anthropic".into()))
        .messages(prompt.clone())
        .max_tokens(256);

    let response = match client.complete(primary).await {
        Ok(resp) => {
            println!("[primary] Anthropic succeeded");
            resp
        }
        Err(e) if should_fallback(&e.kind) => {
            println!(
                "[primary] Anthropic failed ({:?}), falling back to OpenAI",
                e.kind
            );

            // Fall back to secondary provider (OpenAI).
            let fallback = Request::default()
                .model("gpt-4o-mini")
                .provider(Some("openai".into()))
                .messages(prompt)
                .max_tokens(256);

            client.complete(fallback).await?
        }
        Err(e) => return Err(e.into()), // Non-retryable error, propagate.
    };

    println!("Provider: {}", response.provider);
    println!("Response: {}", response.text());
    Ok(())
}

/// Decide whether an error warrants falling back to another provider.
/// We fall back on server errors, rate limits, and timeouts — but NOT
/// on auth errors or bad requests (those would fail everywhere).
fn should_fallback(kind: &ErrorKind) -> bool {
    matches!(
        kind,
        ErrorKind::Server
            | ErrorKind::RateLimit
            | ErrorKind::RequestTimeout
            | ErrorKind::Network
            | ErrorKind::QuotaExceeded
    )
}
