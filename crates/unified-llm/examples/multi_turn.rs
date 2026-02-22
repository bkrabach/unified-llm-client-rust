/// Multi-turn conversation using the low-level Client API.
///
/// This is the pattern a coding agent loop would use (spec §2.3):
/// build requests directly, call client.complete(), append the assistant's
/// reply, then send a follow-up with the full conversation history.
///
/// Requires at least one API key env var (ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY).
use unified_llm::{Client, Message, Request};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Auto-detect providers from environment variables.
    let client = Client::from_env()?;

    // Start the conversation history with a user message.
    let mut messages = vec![Message::user(
        "What is the capital of France? Reply in one sentence.",
    )];

    // Turn 1: send the initial request via the low-level API.
    let request = Request::default()
        .model("gpt-4o-mini")
        .messages(messages.clone())
        .max_tokens(256)
        .provider(Some("openai".into()));

    let response = client.complete(request).await?;
    let assistant_text = response.text();
    println!("Turn 1 — Assistant: {assistant_text}");

    // Append the assistant's reply to our conversation history.
    messages.push(Message::assistant(&assistant_text));

    // Turn 2: ask a follow-up that requires the prior context.
    messages.push(Message::user("And what is that city's population?"));

    let request = Request::default()
        .model("gpt-4o-mini")
        .messages(messages.clone())
        .max_tokens(256)
        .provider(Some("openai".into()));

    let response = client.complete(request).await?;
    println!("Turn 2 — Assistant: {}", response.text());
    println!(
        "Total tokens (turn 2): {} in + {} out",
        response.usage.input_tokens, response.usage.output_tokens
    );

    Ok(())
}
