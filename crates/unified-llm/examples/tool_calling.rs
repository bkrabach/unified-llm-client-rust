/// Active tool calling with automatic execution loop.
///
/// Shows defining a `Tool::active()` with an async handler, calling `generate()`
/// with tools and `max_tool_rounds`, and the SDK automatically executing the tool
/// and feeding results back to the model.
///
/// Run: cargo run --example tool_calling
/// Requires: OPENAI_API_KEY (or ANTHROPIC_API_KEY / GEMINI_API_KEY) in env.
use unified_llm::{generate, Client, GenerateOptions, Tool};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::from_env()?;

    // Define an active tool — the SDK will auto-execute it when the model calls it
    let weather_tool = Tool::active(
        "get_weather",
        "Get the current weather for a city",
        serde_json::json!({
            "type": "object",
            "properties": {
                "city": { "type": "string", "description": "The city name" }
            },
            "required": ["city"],
            "additionalProperties": false
        }),
        // Async handler: receives parsed JSON args, returns JSON result
        |args| {
            Box::pin(async move {
                let city = args["city"].as_str().unwrap_or("unknown");
                // In a real app, call an actual weather API here
                Ok(serde_json::json!({
                    "city": city,
                    "temperature_f": 72,
                    "condition": "sunny"
                }))
            })
        },
    );

    // generate() with tools — the SDK handles the tool loop automatically:
    //   1. Model receives prompt → decides to call get_weather
    //   2. SDK executes the handler with the model's arguments
    //   3. SDK feeds tool results back to the model
    //   4. Model produces final text incorporating the weather data
    let result = generate(
        GenerateOptions::new("gpt-4o-mini")
            .prompt("What's the weather like in San Francisco?")
            .provider("openai")
            .tools(vec![weather_tool])
            .max_tool_rounds(1),
        &client,
    )
    .await?;

    println!("Final answer: {}", result.text);
    println!("\nSteps taken: {}", result.steps.len());
    for (i, step) in result.steps.iter().enumerate() {
        println!(
            "  Step {}: finish_reason={}, tool_calls={}",
            i + 1,
            step.finish_reason.reason,
            step.tool_calls.len()
        );
    }
    println!(
        "\nTotal usage: {} input / {} output tokens",
        result.total_usage.input_tokens, result.total_usage.output_tokens
    );

    Ok(())
}
