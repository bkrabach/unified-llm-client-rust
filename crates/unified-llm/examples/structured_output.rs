/// JSON schema-validated structured output.
///
/// Shows `generate_object()` with a JSON schema, getting back a parsed
/// and validated `serde_json::Value`, and accessing structured fields.
///
/// Run: cargo run --example structured_output
/// Requires: OPENAI_API_KEY (or ANTHROPIC_API_KEY / GEMINI_API_KEY) in env.
use unified_llm::{generate_object, Client, GenerateOptions};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::from_env()?;

    // Define the JSON schema the model must conform to
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "name":       { "type": "string" },
            "capital":    { "type": "string" },
            "population": { "type": "integer" },
            "languages":  { "type": "array", "items": { "type": "string" } }
        },
        "required": ["name", "capital", "population", "languages"],
        "additionalProperties": false
    });

    // generate_object() calls generate() internally with response_format=json_schema,
    // then parses and validates the response against the schema.
    let result = generate_object(
        GenerateOptions::new("gpt-4o-mini")
            .prompt("Give me information about Japan.")
            .provider("openai"),
        schema,
        &client,
    )
    .await?;

    // result.output contains the parsed, schema-validated serde_json::Value
    if let Some(output) = &result.output {
        println!("Country:    {}", output["name"]);
        println!("Capital:    {}", output["capital"]);
        println!("Population: {}", output["population"]);
        print!("Languages:  ");
        if let Some(langs) = output["languages"].as_array() {
            let names: Vec<&str> = langs.iter().filter_map(|l| l.as_str()).collect();
            println!("{}", names.join(", "));
        }
    }

    println!(
        "\nUsage: {} input / {} output tokens",
        result.usage.input_tokens, result.usage.output_tokens
    );

    Ok(())
}
