/// Basic text generation across providers.
///
/// Shows `Client::from_env()`, `generate()` with a prompt, switching providers,
/// and printing text, usage, and finish_reason.
///
/// Run: cargo run --example basic_generate
/// Requires: OPENAI_API_KEY (or ANTHROPIC_API_KEY / GEMINI_API_KEY) in env.
use unified_llm::{generate, Client, GenerateOptions};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Auto-detect providers from env vars:
    //   ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY
    let client = Client::from_env()?;

    // --- OpenAI ---
    let result = generate(
        GenerateOptions::new("gpt-4o-mini")
            .prompt("What is the capital of France? Answer in one sentence.")
            .provider("openai"),
        &client,
    )
    .await?;

    println!("=== OpenAI ===");
    println!("Text: {}", result.text);
    println!("Finish reason: {}", result.finish_reason.reason);
    println!(
        "Usage: {} input / {} output tokens",
        result.usage.input_tokens, result.usage.output_tokens
    );

    // --- Anthropic (switch provider + model) ---
    let result = generate(
        GenerateOptions::new("claude-sonnet-4-20250514")
            .prompt("What is the capital of France? Answer in one sentence.")
            .provider("anthropic"),
        &client,
    )
    .await?;

    println!("\n=== Anthropic ===");
    println!("Text: {}", result.text);
    println!("Finish reason: {}", result.finish_reason.reason);
    println!(
        "Usage: {} input / {} output tokens",
        result.usage.input_tokens, result.usage.output_tokens
    );

    // --- Gemini (switch provider + model) ---
    let result = generate(
        GenerateOptions::new("gemini-2.0-flash")
            .prompt("What is the capital of France? Answer in one sentence.")
            .provider("gemini"),
        &client,
    )
    .await?;

    println!("\n=== Gemini ===");
    println!("Text: {}", result.text);
    println!("Finish reason: {}", result.finish_reason.reason);
    println!(
        "Usage: {} input / {} output tokens",
        result.usage.input_tokens, result.usage.output_tokens
    );

    Ok(())
}
