/// Real-time streaming output with typewriter effect.
///
/// Shows `stream()` returning `StreamResult`, iterating events with
/// `StreamExt::next()`, printing TextDelta events as they arrive,
/// and getting the final accumulated response.
///
/// Run: cargo run --example streaming
/// Requires: OPENAI_API_KEY (or ANTHROPIC_API_KEY / GEMINI_API_KEY) in env.
use std::io::Write;

use futures::StreamExt;
use unified_llm::{stream, Client, GenerateOptions, StreamEventType};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::from_env()?;

    let opts = GenerateOptions::new("gpt-4o-mini")
        .prompt("Write a short poem about Rust programming.")
        .provider("openai");

    // stream() returns a StreamResult that implements Stream<Item = Result<StreamEvent, Error>>
    let mut result = stream(opts, &client)?;

    println!("Streaming response:\n");

    // Iterate events as they arrive — real-time typewriter effect
    while let Some(event) = result.next().await {
        let event = event?;
        match event.event_type {
            StreamEventType::TextDelta => {
                // Print each text chunk immediately without a newline
                if let Some(ref delta) = event.delta {
                    print!("{}", delta);
                    std::io::stdout().flush()?; // flush for real-time output
                }
            }
            StreamEventType::Finish => {
                println!("\n"); // blank line after streaming completes
            }
            _ => {} // skip StreamStart, TextStart, TextEnd, etc.
        }
    }

    // After the stream ends, get the full accumulated response
    if let Some(response) = result.response() {
        println!("--- Accumulated Response ---");
        println!("Full text: {}", response.text());
        println!("Finish reason: {}", response.finish_reason.reason);
        println!(
            "Usage: {} input / {} output tokens",
            response.usage.input_tokens, response.usage.output_tokens
        );
    }

    Ok(())
}
