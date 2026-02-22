/// Multi-provider comparison: send the same prompt to all three providers.
///
/// Demonstrates provider-agnostic code — the same Request shape works across
/// OpenAI, Anthropic, and Gemini by changing only the model and provider strings.
/// Also shows the catalog API for looking up model metadata.
///
/// Requires all three API keys: ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY.
use unified_llm::{get_model_info, Client, Message, Request};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::from_env()?;

    // Known-good models for each provider (the catalog also has aliases for discovery)
    let models = [
        ("anthropic", "claude-sonnet-4-20250514"),
        ("openai", "gpt-4o-mini"),
        ("gemini", "gemini-2.0-flash"),
    ];
    let prompt = vec![Message::user(
        "What is the theory of relativity? One sentence.",
    )];

    for (provider, model) in &models {
        println!("--- {provider}: {model} ---");

        // Optional: look up catalog metadata (context window, capabilities)
        if let Some(info) = get_model_info(model) {
            println!(
                "  Catalog: {}k context, tools={}, vision={}",
                info.context_window / 1000,
                info.supports_tools,
                info.supports_vision
            );
        }

        let request = Request::default()
            .model(*model)
            .provider(Some((*provider).into()))
            .messages(prompt.clone())
            .max_tokens(256);

        match client.complete(request).await {
            Ok(resp) => {
                println!("  Response: {}", resp.text());
                println!(
                    "  Tokens: {} in / {} out / {} total",
                    resp.usage.input_tokens, resp.usage.output_tokens, resp.usage.total_tokens
                );
            }
            Err(e) => println!("  Error: {:?}: {}", e.kind, e.message),
        }
        println!();
    }

    Ok(())
}
