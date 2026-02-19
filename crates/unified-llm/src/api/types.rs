// api/types.rs — Tool, GenerateOptions, and related types for the high-level API (Layer 4).

use std::sync::Arc;
use tokio_util::sync::CancellationToken;
use unified_llm_types::*;

use super::generate_types::StepResult;

/// Handler function type for tool execution.
/// Takes parsed JSON arguments, returns JSON result.
pub type ToolExecuteFn = Arc<
    dyn Fn(serde_json::Value) -> BoxFuture<'static, Result<serde_json::Value, Error>> + Send + Sync,
>;

/// Handler function type for tool execution with context.
/// Takes parsed JSON arguments and a ToolContext, returns JSON result.
pub type ToolExecuteWithContextFn = Arc<
    dyn Fn(serde_json::Value, ToolContext) -> BoxFuture<'static, Result<serde_json::Value, Error>>
        + Send
        + Sync,
>;

/// Repair function type for fixing invalid tool call arguments.
/// Called with (tool_call, validation_error) → corrected arguments or None.
pub type RepairToolCallFn = Arc<
    dyn Fn(&unified_llm_types::content::ToolCallData, &str) -> Option<serde_json::Value>
        + Send
        + Sync,
>;

/// Context provided to tool handlers during execution.
#[derive(Debug, Clone)]
pub struct ToolContext {
    /// The conversation messages up to this point.
    pub messages: Vec<Message>,
    /// The ID of the tool call being executed.
    pub tool_call_id: String,
    /// Optional cancellation token for aborting long-running tools.
    pub abort_signal: Option<CancellationToken>,
}

/// A tool with definition and optional execute handler.
///
/// - **Active tool**: has an `execute` or `execute_with_context` handler.
///   When used with `generate()` or `stream()`, the library automatically
///   executes it and loops until the model produces a final text response.
/// - **Passive tool**: no execute handler. Tool calls are returned to the
///   caller in the response.
pub struct Tool {
    /// The serializable definition sent to the provider.
    pub definition: ToolDefinition,
    /// Optional execute handler. Present = active tool, absent = passive tool.
    pub execute: Option<ToolExecuteFn>,
    /// Optional context-aware execute handler. Takes (args, ToolContext).
    /// If both `execute` and `execute_with_context` are set, context-aware wins.
    pub execute_with_context: Option<ToolExecuteWithContextFn>,
}

impl Tool {
    /// Create an active tool with an execute handler.
    pub fn active(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
        execute: impl Fn(serde_json::Value) -> BoxFuture<'static, Result<serde_json::Value, Error>>
            + Send
            + Sync
            + 'static,
    ) -> Self {
        Self {
            definition: ToolDefinition {
                name: name.into(),
                description: description.into(),
                parameters,
                strict: None,
            },
            execute: Some(Arc::new(execute)),
            execute_with_context: None,
        }
    }

    /// Create an active tool with a context-aware execute handler.
    ///
    /// The handler receives both the parsed arguments and a `ToolContext`
    /// containing the conversation history and tool call metadata.
    pub fn active_with_context(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
        execute: impl Fn(
                serde_json::Value,
                ToolContext,
            ) -> BoxFuture<'static, Result<serde_json::Value, Error>>
            + Send
            + Sync
            + 'static,
    ) -> Self {
        Self {
            definition: ToolDefinition {
                name: name.into(),
                description: description.into(),
                parameters,
                strict: None,
            },
            execute: None,
            execute_with_context: Some(Arc::new(execute)),
        }
    }

    /// Create a passive tool (no execute handler).
    pub fn passive(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
    ) -> Self {
        Self {
            definition: ToolDefinition {
                name: name.into(),
                description: description.into(),
                parameters,
                strict: None,
            },
            execute: None,
            execute_with_context: None,
        }
    }

    /// Create a passive tool from an existing ToolDefinition.
    pub fn new(definition: ToolDefinition) -> Self {
        Self {
            definition,
            execute: None,
            execute_with_context: None,
        }
    }

    /// Create an active tool from an existing ToolDefinition with an execute handler.
    pub fn with_execute(definition: ToolDefinition, execute: ToolExecuteFn) -> Self {
        Self {
            definition,
            execute: Some(execute),
            execute_with_context: None,
        }
    }

    /// Check if this tool has an execute handler (active tool).
    pub fn is_active(&self) -> bool {
        self.execute.is_some() || self.execute_with_context.is_some()
    }

    /// Validate the tool definition (name format + parameter schema).
    ///
    /// This is called automatically by `generate()` and `stream()` before
    /// any API call, but can also be called explicitly for fail-fast validation
    /// at construction time.
    ///
    /// # Errors
    /// Returns `Error` if the tool name or parameters are invalid per spec §5.1.
    pub fn validate(&self) -> Result<(), Error> {
        self.definition.validate()
    }
}

impl std::fmt::Debug for Tool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tool")
            .field("definition", &self.definition)
            .field("execute", &self.execute.as_ref().map(|_| "..."))
            .field(
                "execute_with_context",
                &self.execute_with_context.as_ref().map(|_| "..."),
            )
            .finish()
    }
}

/// Options for the `generate()` and `stream()` functions.
///
/// Use the builder pattern:
/// ```ignore
/// let opts = GenerateOptions::new("claude-sonnet-4")
///     .prompt("Explain photosynthesis")
///     .temperature(0.7)
///     .max_tokens(500);
/// ```
pub struct GenerateOptions {
    /// Model identifier (required).
    pub model: String,
    /// Simple text prompt (mutually exclusive with `messages`).
    pub prompt: Option<String>,
    /// Full message history (mutually exclusive with `prompt`).
    pub messages: Option<Vec<Message>>,
    /// System message, prepended to conversation.
    pub system: Option<String>,
    /// Tools available for the model to call.
    pub tools: Option<Vec<Tool>>,
    /// Tool choice mode (auto, none, required, named).
    pub tool_choice: Option<ToolChoice>,
    /// Max rounds of tool execution (default: 1).
    /// 0 = no automatic execution. N = at most N rounds.
    /// Total LLM calls = at most max_tool_rounds + 1.
    pub max_tool_rounds: u32,
    /// Custom stop condition for tool loops.
    #[allow(clippy::type_complexity)]
    pub stop_when: Option<Arc<dyn Fn(&[StepResult]) -> bool + Send + Sync>>,
    /// Response format (text, json_object, json_schema).
    pub response_format: Option<ResponseFormat>,
    /// Sampling temperature.
    pub temperature: Option<f64>,
    /// Nucleus sampling parameter.
    pub top_p: Option<f64>,
    /// Maximum tokens in the response.
    pub max_tokens: Option<u32>,
    /// Stop sequences.
    pub stop_sequences: Option<Vec<String>>,
    /// Reasoning effort level (for reasoning models).
    pub reasoning_effort: Option<String>,
    /// Provider name (overrides default).
    pub provider: Option<String>,
    /// Provider-specific options passthrough.
    pub provider_options: Option<serde_json::Value>,
    /// Max retry attempts for transient errors (default: 2).
    pub max_retries: u32,
    /// Timeout configuration.
    pub timeout: Option<TimeoutConfig>,
    /// Cancellation signal.
    pub abort_signal: Option<CancellationToken>,
    /// Optional repair function for invalid tool call arguments.
    /// Called when schema validation fails, given (tool_call, error_message).
    /// Returns corrected arguments or None to propagate the error.
    pub repair_tool_call: Option<RepairToolCallFn>,
}

impl GenerateOptions {
    /// Create options with the required model parameter.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            prompt: None,
            messages: None,
            system: None,
            tools: None,
            tool_choice: None,
            max_tool_rounds: 1,
            stop_when: None,
            response_format: None,
            temperature: None,
            top_p: None,
            max_tokens: None,
            stop_sequences: None,
            reasoning_effort: None,
            provider: None,
            provider_options: None,
            max_retries: 2,
            timeout: None,
            abort_signal: None,
            repair_tool_call: None,
        }
    }

    // Builder methods — each consumes and returns self for chaining.

    pub fn prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = Some(prompt.into());
        self
    }

    pub fn messages(mut self, messages: Vec<Message>) -> Self {
        self.messages = Some(messages);
        self
    }

    pub fn system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    pub fn tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools = Some(tools);
        self
    }

    pub fn tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = Some(choice);
        self
    }

    pub fn max_tool_rounds(mut self, rounds: u32) -> Self {
        self.max_tool_rounds = rounds;
        self
    }

    pub fn stop_when(mut self, f: impl Fn(&[StepResult]) -> bool + Send + Sync + 'static) -> Self {
        self.stop_when = Some(Arc::new(f));
        self
    }

    pub fn response_format(mut self, fmt: ResponseFormat) -> Self {
        self.response_format = Some(fmt);
        self
    }

    pub fn temperature(mut self, t: f64) -> Self {
        self.temperature = Some(t);
        self
    }

    pub fn top_p(mut self, p: f64) -> Self {
        self.top_p = Some(p);
        self
    }

    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    pub fn stop_sequences(mut self, seqs: Vec<String>) -> Self {
        self.stop_sequences = Some(seqs);
        self
    }

    pub fn reasoning_effort(mut self, effort: impl Into<String>) -> Self {
        self.reasoning_effort = Some(effort.into());
        self
    }

    pub fn provider(mut self, provider: impl Into<String>) -> Self {
        self.provider = Some(provider.into());
        self
    }

    pub fn provider_options(mut self, opts: serde_json::Value) -> Self {
        self.provider_options = Some(opts);
        self
    }

    pub fn max_retries(mut self, retries: u32) -> Self {
        self.max_retries = retries;
        self
    }

    pub fn timeout(mut self, timeout: impl Into<TimeoutConfig>) -> Self {
        self.timeout = Some(timeout.into());
        self
    }

    pub fn abort_signal(mut self, token: CancellationToken) -> Self {
        self.abort_signal = Some(token);
        self
    }

    pub fn repair_tool_call(
        mut self,
        f: impl Fn(&unified_llm_types::content::ToolCallData, &str) -> Option<serde_json::Value>
            + Send
            + Sync
            + 'static,
    ) -> Self {
        self.repair_tool_call = Some(Arc::new(f));
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_active_has_execute_handler() {
        let tool = Tool::active(
            "echo",
            "Echoes input",
            serde_json::json!({"type": "object", "properties": {"text": {"type": "string"}}}),
            |args| Box::pin(async move { Ok(args) }),
        );
        assert!(tool.is_active());
        assert_eq!(tool.definition.name, "echo");
    }

    #[test]
    fn test_tool_passive_has_no_execute_handler() {
        let tool = Tool::passive(
            "search",
            "Search the web",
            serde_json::json!({"type": "object", "properties": {"query": {"type": "string"}}}),
        );
        assert!(!tool.is_active());
        assert_eq!(tool.definition.name, "search");
    }

    #[test]
    fn test_tool_new_creates_passive_from_definition() {
        let def = ToolDefinition {
            name: "test_fn".into(),
            description: "A test function".into(),
            parameters: serde_json::json!({}),
            strict: None,
        };
        let tool = Tool::new(def);
        assert!(!tool.is_active());
        assert_eq!(tool.definition.name, "test_fn");
    }

    #[test]
    fn test_tool_with_execute_creates_active_from_definition() {
        let def = ToolDefinition {
            name: "active_fn".into(),
            description: "An active function".into(),
            parameters: serde_json::json!({}),
            strict: None,
        };
        let execute: ToolExecuteFn = Arc::new(|args| Box::pin(async move { Ok(args) }));
        let tool = Tool::with_execute(def, execute);
        assert!(tool.is_active());
        assert_eq!(tool.definition.name, "active_fn");
    }

    #[test]
    fn test_tool_debug_redacts_execute() {
        let tool = Tool::active("fn", "desc", serde_json::json!({}), |_| {
            Box::pin(async { Ok(serde_json::json!("ok")) })
        });
        let debug = format!("{:?}", tool);
        assert!(debug.contains("\"...\""));
        assert!(!debug.contains("closure"));
    }

    // === F-3: Tool validation at construction time ===

    #[test]
    fn test_tool_validate_rejects_invalid_name() {
        let tool = Tool::active(
            "1invalid",
            "desc",
            serde_json::json!({"type": "object"}),
            |_args| Box::pin(async { Ok(serde_json::json!("ok")) }),
        );
        assert!(
            tool.validate().is_err(),
            "Tool with invalid name should fail validation"
        );
    }

    #[test]
    fn test_tool_passive_validate_rejects_invalid_name() {
        let tool = Tool::passive("1invalid", "desc", serde_json::json!({"type": "object"}));
        assert!(tool.validate().is_err());
    }

    #[test]
    fn test_tool_validate_rejects_invalid_parameters() {
        let tool = Tool::new(ToolDefinition {
            name: "valid_name".into(),
            description: "desc".into(),
            parameters: serde_json::json!({"type": "string"}), // not "object" — invalid
            strict: None,
        });
        assert!(tool.validate().is_err());
    }

    #[test]
    fn test_tool_validate_accepts_valid_definition() {
        let tool = Tool::active(
            "get_weather",
            "Get weather",
            serde_json::json!({"type": "object", "properties": {}}),
            |_args| Box::pin(async { Ok(serde_json::json!("ok")) }),
        );
        assert!(tool.validate().is_ok());
    }

    #[test]
    fn test_generate_options_defaults() {
        let opts = GenerateOptions::new("claude-sonnet-4");
        assert_eq!(opts.model, "claude-sonnet-4");
        assert_eq!(opts.max_tool_rounds, 1);
        assert_eq!(opts.max_retries, 2);
        assert!(opts.prompt.is_none());
        assert!(opts.messages.is_none());
        assert!(opts.tools.is_none());
        assert!(opts.abort_signal.is_none());
    }

    #[test]
    fn test_generate_options_builder_chaining() {
        let opts = GenerateOptions::new("gpt-4o")
            .prompt("Hello")
            .temperature(0.7)
            .max_tokens(500)
            .max_retries(0)
            .max_tool_rounds(3);
        assert_eq!(opts.prompt, Some("Hello".into()));
        assert_eq!(opts.temperature, Some(0.7));
        assert_eq!(opts.max_tokens, Some(500));
        assert_eq!(opts.max_retries, 0);
        assert_eq!(opts.max_tool_rounds, 3);
    }

    #[test]
    fn test_generate_options_messages_builder() {
        let msgs = vec![Message::user("hi"), Message::assistant("hello")];
        let opts = GenerateOptions::new("model").messages(msgs);
        assert_eq!(opts.messages.unwrap().len(), 2);
    }

    #[test]
    fn test_generate_options_system_builder() {
        let opts = GenerateOptions::new("model").system("You are helpful");
        assert_eq!(opts.system, Some("You are helpful".into()));
    }

    #[test]
    fn test_generate_options_all_builder_methods() {
        let opts = GenerateOptions::new("model")
            .prompt("hi")
            .system("sys")
            .tool_choice(ToolChoice {
                mode: "auto".into(),
                tool_name: None,
            })
            .stop_when(|_steps| false)
            .response_format(ResponseFormat {
                r#type: "json_object".into(),
                json_schema: None,
                strict: false,
            })
            .top_p(0.9)
            .stop_sequences(vec!["STOP".into()])
            .reasoning_effort("high")
            .provider("anthropic")
            .provider_options(serde_json::json!({"anthropic": {}}))
            .timeout(TimeoutConfig::default());
        assert!(opts.prompt.is_some());
        assert!(opts.system.is_some());
        assert!(opts.tool_choice.is_some());
        assert!(opts.stop_when.is_some());
        assert!(opts.response_format.is_some());
        assert_eq!(opts.top_p, Some(0.9));
        assert!(opts.stop_sequences.is_some());
        assert_eq!(opts.reasoning_effort, Some("high".into()));
        assert_eq!(opts.provider, Some("anthropic".into()));
        assert!(opts.provider_options.is_some());
        assert!(opts.timeout.is_some());
    }

    #[test]
    fn test_generate_options_timeout_f64_shorthand() {
        let opts = GenerateOptions::new("gpt-4o").timeout(30.0);
        let tc = opts.timeout.unwrap();
        assert_eq!(tc.total, Some(30.0));
        assert_eq!(tc.per_step, None);
    }
}
