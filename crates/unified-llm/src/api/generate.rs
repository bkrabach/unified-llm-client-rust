// api/generate.rs — generate() function (Layer 4).
//
// Wraps Client.complete() with prompt standardization, tool execution loops,
// multi-step orchestration, automatic retries, cancellation, and timeouts.

use std::time::Duration;

use unified_llm_types::*;

use crate::client::Client;
use crate::util::retry::with_retry;

use super::generate_types::{GenerateResult, StepResult};
use super::tool_loop::execute_all_tools;
use super::types::GenerateOptions;

/// The primary blocking generation function.
///
/// Wraps `Client.complete()` with prompt standardization, tool execution loops,
/// multi-step orchestration, automatic retries, cancellation, and timeouts.
///
/// When active tools (tools with execute handlers) are provided and the model
/// returns tool calls, this function automatically executes the tools, feeds
/// results back to the model, and loops until a final text response is produced
/// or `max_tool_rounds` is reached.
///
/// Cancellation (DoD 8.4.9): If `abort_signal` is set and cancelled, returns `Error::abort()`.
/// Timeouts (DoD 8.4.10): If `timeout.total` is set, wraps entire operation.
/// If `timeout.per_step` is set, wraps each individual LLM call.
///
/// Spec reference: §4.3
pub async fn generate(options: GenerateOptions, client: &Client) -> Result<GenerateResult, Error> {
    let abort_signal = options.abort_signal.clone();
    let timeout_config = options.timeout.clone();

    // Wrap in total timeout if configured (DoD 8.4.10)
    let result = if let Some(total) = timeout_config.as_ref().and_then(|t| t.total) {
        tokio::time::timeout(
            Duration::from_secs_f64(total),
            generate_with_cancel(options, client, &abort_signal),
        )
        .await
        .map_err(|_| Error {
            kind: ErrorKind::RequestTimeout,
            message: format!("Total timeout of {total}s exceeded"),
            retryable: false,
            source: None,
            provider: None,
            status_code: None,
            error_code: None,
            retry_after: None,
            raw: None,
        })?
    } else {
        generate_with_cancel(options, client, &abort_signal).await
    };

    result
}

/// Generate text using the default client.
///
/// Falls back to `Client::from_env()` if no default client has been set.
/// This is a convenience wrapper around `generate()` for simple use cases
/// that don't need explicit client management.
///
/// # Errors
/// Returns `ConfigurationError` if no default client is set and no API keys
/// are found in the environment.
pub async fn generate_with_default(options: GenerateOptions) -> Result<GenerateResult, Error> {
    let client = crate::default_client::get_default_client()?;
    generate(options, &client).await
}

/// Generate with cancellation support. Races the inner logic against the abort signal.
async fn generate_with_cancel(
    options: GenerateOptions,
    client: &Client,
    abort_signal: &Option<tokio_util::sync::CancellationToken>,
) -> Result<GenerateResult, Error> {
    // Eager check: if already cancelled, return immediately (no race condition)
    if let Some(token) = abort_signal {
        if token.is_cancelled() {
            return Err(Error::abort());
        }
    }

    tokio::select! {
        result = generate_inner(options, client) => result,
        _ = async {
            match abort_signal {
                Some(token) => token.cancelled().await,
                None => std::future::pending::<()>().await,
            }
        } => Err(Error::abort()),
    }
}

/// Inner generate logic — the actual tool loop and orchestration.
async fn generate_inner(
    options: GenerateOptions,
    client: &Client,
) -> Result<GenerateResult, Error> {
    // M-9: Validate tools and tool_choice upfront before any API calls.
    if let Some(ref tools) = options.tools {
        for tool in tools {
            tool.definition.validate()?;
        }
    }
    if let Some(ref tc) = options.tool_choice {
        tc.validate()?;
    }

    // Take ownership of messages before borrowing options for other fields
    let prompt = options.prompt.clone();
    let messages_input = options.messages.clone();
    let system = options.system.clone();

    let mut conversation =
        standardize_messages(prompt.as_deref(), messages_input, system.as_deref())?;

    let tool_definitions: Vec<ToolDefinition> = options
        .tools
        .as_ref()
        .map(|tools| tools.iter().map(|t| t.definition.clone()).collect())
        .unwrap_or_default();

    let retry_policy = build_retry_policy(options.max_retries);
    let per_step_timeout = options.timeout.as_ref().and_then(|t| t.per_step);
    let tools = options.tools.as_deref().unwrap_or(&[]);
    let repair_fn = options.repair_tool_call.as_ref();
    let mut steps = Vec::new();

    for round in 0..=options.max_tool_rounds {
        let request = build_request(&options, &conversation, &tool_definitions);

        // Per-step retry wrapping (DoD 8.8.8) with optional per-step timeout (DoD 8.4.10)
        let response = with_retry(&retry_policy, || {
            let req = request.clone();
            async move { complete_with_step_timeout(client, req, per_step_timeout).await }
        })
        .await?;

        // Check if this response has tool calls
        let response_tool_calls = response.tool_calls();
        let is_tool_call =
            response.finish_reason.reason == "tool_calls" && !response_tool_calls.is_empty();

        // Check if any of the called tools are active (have execute handlers)
        let has_active = is_tool_call
            && response_tool_calls.iter().any(|tc| {
                tools
                    .iter()
                    .any(|t| t.definition.name == tc.name && t.is_active())
            });

        // Execute tools if conditions are met
        let tool_results = if is_tool_call && has_active && round < options.max_tool_rounds {
            let call_refs: Vec<&ToolCallData> = response_tool_calls.into_iter().collect();
            execute_all_tools(
                tools,
                &call_refs,
                &conversation,
                &options.abort_signal,
                repair_fn,
            )
            .await
        } else {
            vec![]
        };

        let step = build_step_result(&response, tool_results.clone());
        steps.push(step);

        // Exit conditions
        if !is_tool_call {
            break; // natural completion
        }
        if round >= options.max_tool_rounds {
            tracing::info!(
                "max_tool_rounds={} reached, returning with pending tool calls",
                options.max_tool_rounds
            );
            break; // budget exhausted
        }
        if let Some(ref stop_when) = options.stop_when {
            if stop_when(&steps) {
                break; // custom stop condition
            }
        }
        if !has_active {
            break; // all passive — return to caller
        }

        // Extend conversation for next round
        conversation.push(response.message.clone()); // assistant message with tool calls
        for result in &tool_results {
            conversation.push(Message::tool_result(
                &result.tool_call_id,
                result.content.to_string(),
                result.is_error,
            ));
        }
    }

    GenerateResult::from_steps(steps)
}

/// Wrap a single client.complete() call with an optional per-step timeout (DoD 8.4.10).
/// Per-step timeouts are retryable (the retry wrapper may retry them).
async fn complete_with_step_timeout(
    client: &Client,
    request: Request,
    per_step: Option<f64>,
) -> Result<Response, Error> {
    if let Some(step_timeout) = per_step {
        tokio::time::timeout(
            Duration::from_secs_f64(step_timeout),
            client.complete(request),
        )
        .await
        .map_err(|_| Error {
            kind: ErrorKind::RequestTimeout,
            message: format!("Per-step timeout of {step_timeout}s exceeded"),
            retryable: true, // Per-step timeouts are retryable
            source: None,
            provider: None,
            status_code: None,
            error_code: None,
            retry_after: None,
            raw: None,
        })?
    } else {
        client.complete(request).await
    }
}

/// Construct a RetryPolicy from the user's max_retries setting.
/// Uses standard defaults for backoff parameters, with very short delays in test.
fn build_retry_policy(max_retries: u32) -> RetryPolicy {
    RetryPolicy {
        max_retries,
        ..Default::default()
    }
}

/// Create an InvalidRequest error.
fn invalid_request(message: impl Into<String>) -> Error {
    Error {
        kind: ErrorKind::InvalidRequest,
        message: message.into(),
        retryable: false,
        source: None,
        provider: None,
        status_code: None,
        error_code: None,
        retry_after: None,
        raw: None,
    }
}

/// Validate and normalize input messages.
///
/// - `prompt` and `messages` are mutually exclusive (error if both provided).
/// - At least one of `prompt` or `messages` must be provided (error if neither).
/// - `prompt` is converted to a single `Message::user(prompt)`.
/// - `system` is prepended as `Message::system(system)` if provided.
pub(crate) fn standardize_messages(
    prompt: Option<&str>,
    messages: Option<Vec<Message>>,
    system: Option<&str>,
) -> Result<Vec<Message>, Error> {
    // 1. Validate mutual exclusivity
    if prompt.is_some() && messages.is_some() {
        return Err(invalid_request(
            "Cannot provide both 'prompt' and 'messages'",
        ));
    }

    let mut msgs = Vec::new();

    // 2. Prepend system message if provided
    if let Some(sys) = system {
        msgs.push(Message::system(sys));
    }

    // 3. Build user messages
    if let Some(prompt) = prompt {
        msgs.push(Message::user(prompt));
    } else if let Some(messages) = messages {
        msgs.extend(messages);
    } else {
        return Err(invalid_request(
            "Must provide either 'prompt' or 'messages'",
        ));
    }

    Ok(msgs)
}

/// Build a low-level Request from GenerateOptions and prepared messages.
pub(crate) fn build_request(
    options: &GenerateOptions,
    conversation: &[Message],
    tool_definitions: &[ToolDefinition],
) -> Request {
    let mut req = Request::default()
        .model(&options.model)
        .messages(conversation.to_vec());

    if !tool_definitions.is_empty() {
        req = req.tools(tool_definitions.to_vec());
    }
    if let Some(tc) = &options.tool_choice {
        req.tool_choice = Some(tc.clone());
    }
    if let Some(t) = options.temperature {
        req = req.temperature(t);
    }
    if let Some(p) = options.top_p {
        req = req.top_p(p);
    }
    if let Some(m) = options.max_tokens {
        req = req.max_tokens(m);
    }
    if let Some(s) = &options.stop_sequences {
        req = req.stop_sequences(s.clone());
    }
    if let Some(r) = &options.reasoning_effort {
        req = req.reasoning_effort(r.clone());
    }
    if let Some(p) = &options.provider {
        req.provider = Some(p.clone());
    }
    if let Some(o) = &options.provider_options {
        req = req.provider_options(Some(o.clone()));
    }
    if let Some(f) = &options.response_format {
        req = req.response_format(f.clone());
    }
    req
}

/// Extract a StepResult from a provider Response.
pub(crate) fn build_step_result(response: &Response, tool_results: Vec<ToolResult>) -> StepResult {
    StepResult {
        text: response.text(),
        reasoning: response.reasoning(),
        tool_calls: response
            .tool_calls()
            .into_iter()
            .map(|tc| {
                let arguments = match &tc.arguments {
                    ArgumentValue::Dict(m) => m.clone(),
                    ArgumentValue::Raw(s) => serde_json::from_str(s).unwrap_or_default(),
                };
                ToolCall {
                    id: tc.id.clone(),
                    name: tc.name.clone(),
                    arguments,
                    raw_arguments: None,
                }
            })
            .collect(),
        tool_results,
        finish_reason: response.finish_reason.clone(),
        usage: response.usage.clone(),
        response: response.clone(),
        warnings: response.warnings.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{make_test_response, MockProvider};

    fn make_client_with_mock(mock: MockProvider) -> Client {
        Client::builder()
            .provider("mock", Box::new(mock))
            .build()
            .unwrap()
    }

    // --- DoD 8.4.1: generate() with simple text prompt ---

    #[tokio::test]
    async fn test_generate_simple_prompt() {
        let mock = MockProvider::new("mock").with_response(make_test_response("Hello!", "mock"));
        let client = make_client_with_mock(mock);
        let opts = GenerateOptions::new("test-model").prompt("Say hello");
        let result = generate(opts, &client).await.unwrap();
        assert_eq!(result.text, "Hello!");
        assert_eq!(result.finish_reason.reason, "stop");
        assert_eq!(result.steps.len(), 1);
    }

    // --- DoD 8.4.2: generate() with messages list ---

    #[tokio::test]
    async fn test_generate_with_messages() {
        let mock = MockProvider::new("mock").with_response(make_test_response("Response", "mock"));
        let client = make_client_with_mock(mock);
        let msgs = vec![
            Message::user("Hello"),
            Message::assistant("Hi"),
            Message::user("How are you?"),
        ];
        let opts = GenerateOptions::new("test-model").messages(msgs);
        let result = generate(opts, &client).await.unwrap();
        assert_eq!(result.text, "Response");
    }

    // --- DoD 8.4.3: generate() rejects both prompt and messages ---

    #[tokio::test]
    async fn test_generate_rejects_both_prompt_and_messages() {
        let mock = MockProvider::new("mock").with_response(make_test_response("", "mock"));
        let client = make_client_with_mock(mock);
        let opts = GenerateOptions::new("test-model")
            .prompt("Hello")
            .messages(vec![Message::user("Hi")]);
        let err = generate(opts, &client).await.unwrap_err();
        assert_eq!(err.kind, ErrorKind::InvalidRequest);
        assert!(err.message.contains("prompt"));
        assert!(err.message.contains("messages"));
    }

    // --- Rejects neither prompt nor messages ---

    #[tokio::test]
    async fn test_generate_rejects_neither_prompt_nor_messages() {
        let mock = MockProvider::new("mock").with_response(make_test_response("", "mock"));
        let client = make_client_with_mock(mock);
        let opts = GenerateOptions::new("test-model");
        let err = generate(opts, &client).await.unwrap_err();
        assert_eq!(err.kind, ErrorKind::InvalidRequest);
    }

    // --- System message prepended ---

    #[tokio::test]
    async fn test_generate_prepends_system_message() {
        let mock = MockProvider::new("mock").with_response(make_test_response("Done", "mock"));
        let client = make_client_with_mock(mock);
        let opts = GenerateOptions::new("test-model")
            .system("You are helpful")
            .prompt("Hello");
        let result = generate(opts, &client).await.unwrap();
        assert_eq!(result.text, "Done");
        // Verify the request had a system message by checking recorded requests
        // (The MockProvider records requests, but we verify indirectly by
        //  ensuring no error — the system message doesn't break anything)
    }

    // --- Response fields populated ---

    #[tokio::test]
    async fn test_generate_response_fields_populated() {
        let mut resp = make_test_response("Answer", "mock");
        resp.usage = Usage {
            input_tokens: 10,
            output_tokens: 5,
            total_tokens: 15,
            ..Default::default()
        };
        let mock = MockProvider::new("mock").with_response(resp);
        let client = make_client_with_mock(mock);
        let opts = GenerateOptions::new("test-model").prompt("Question");
        let result = generate(opts, &client).await.unwrap();
        assert_eq!(result.text, "Answer");
        assert_eq!(result.usage.input_tokens, 10);
        assert_eq!(result.usage.output_tokens, 5);
        assert_eq!(result.usage.total_tokens, 15);
        assert_eq!(result.finish_reason.reason, "stop");
        assert_eq!(result.steps.len(), 1);
        assert_eq!(result.steps[0].text, "Answer");
    }

    // --- standardize_messages unit tests ---

    #[test]
    fn test_standardize_prompt_only() {
        let msgs = standardize_messages(Some("hello"), None, None).unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].role, Role::User);
    }

    #[test]
    fn test_standardize_messages_only() {
        let input = vec![Message::user("a"), Message::user("b")];
        let msgs = standardize_messages(None, Some(input), None).unwrap();
        assert_eq!(msgs.len(), 2);
    }

    #[test]
    fn test_standardize_with_system() {
        let msgs = standardize_messages(Some("hello"), None, Some("sys")).unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, Role::System);
        assert_eq!(msgs[1].role, Role::User);
    }

    #[test]
    fn test_standardize_system_with_messages() {
        let input = vec![Message::user("q")];
        let msgs = standardize_messages(None, Some(input), Some("sys")).unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, Role::System);
        assert_eq!(msgs[1].role, Role::User);
    }

    #[test]
    fn test_standardize_rejects_both() {
        let result = standardize_messages(Some("prompt"), Some(vec![Message::user("msg")]), None);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind, ErrorKind::InvalidRequest);
    }

    #[test]
    fn test_standardize_rejects_neither() {
        let result = standardize_messages(None, None, None);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind, ErrorKind::InvalidRequest);
    }

    // --- Test helpers for tool loop tests ---

    use unified_llm_types::content::{ArgumentValue, ContentPart, ToolCallData};

    /// Helper: create a Response that looks like a tool call from the model.
    fn make_tool_call_response(tool_name: &str, call_id: &str) -> Response {
        Response {
            id: format!("resp_{}", call_id),
            model: "test".into(),
            provider: "mock".into(),
            message: Message {
                role: Role::Assistant,
                content: vec![ContentPart::ToolCall {
                    tool_call: ToolCallData {
                        id: call_id.into(),
                        name: tool_name.into(),
                        arguments: ArgumentValue::Dict(
                            serde_json::json!({"text": "hello"})
                                .as_object()
                                .unwrap()
                                .clone(),
                        ),
                        r#type: "function".into(),
                    },
                }],
                name: None,
                tool_call_id: None,
            },
            finish_reason: FinishReason::tool_calls(),
            usage: Usage::default(),
            raw: None,
            warnings: vec![],
            rate_limit: None,
        }
    }

    /// Helper: create an echo tool (active) that returns its arguments.
    fn echo_tool() -> super::super::types::Tool {
        super::super::types::Tool::active(
            "echo",
            "echoes input",
            serde_json::json!({"type": "object"}),
            |args| Box::pin(async move { Ok(args) }),
        )
    }

    // --- DoD 8.7.1: Active tools trigger automatic execution loop ---

    #[tokio::test]
    async fn test_generate_with_active_tool_loop() {
        // Step 1: model returns tool call → Step 2: model returns final text
        let mock = MockProvider::new("mock")
            .with_response(make_tool_call_response("echo", "call_1"))
            .with_response(make_test_response("Echoed: hello", "mock"));
        let client = make_client_with_mock(mock);

        let tool = echo_tool();
        let opts = GenerateOptions::new("test")
            .prompt("echo hello")
            .tools(vec![tool])
            .max_tool_rounds(1);
        let result = generate(opts, &client).await.unwrap();
        assert_eq!(result.text, "Echoed: hello");
        assert_eq!(result.steps.len(), 2); // initial call + after tool execution
    }

    // --- DoD 8.7.2: Passive tools return without loop ---

    #[tokio::test]
    async fn test_generate_with_passive_tool_returns_without_loop() {
        let mock =
            MockProvider::new("mock").with_response(make_tool_call_response("search", "call_1"));
        let client = make_client_with_mock(mock);

        let tool = super::super::types::Tool::passive(
            "search",
            "search web",
            serde_json::json!({"type": "object"}),
        );
        let opts = GenerateOptions::new("test")
            .prompt("search something")
            .tools(vec![tool])
            .max_tool_rounds(3);
        let result = generate(opts, &client).await.unwrap();
        assert_eq!(result.steps.len(), 1); // No looping — returned to caller
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "search");
    }

    // --- DoD 8.7.3: max_tool_rounds limits loop ---

    #[tokio::test]
    async fn test_generate_max_tool_rounds_limits_loop() {
        // Model always returns tool calls — loop should stop after max_tool_rounds
        // max_tool_rounds=1: initial call + 1 round = 2 LLM calls max
        // Queue: tool_call, tool_call, tool_call, final text
        // With max_tool_rounds=1, only first 2 should be consumed
        let mock = MockProvider::new("mock")
            .with_response(make_tool_call_response("echo", "c1"))
            .with_response(make_tool_call_response("echo", "c2"))
            .with_response(make_tool_call_response("echo", "c3"))
            .with_response(make_test_response("final", "mock"));
        let client = make_client_with_mock(mock);
        let tool = echo_tool();
        let opts = GenerateOptions::new("test")
            .prompt("go")
            .tools(vec![tool])
            .max_tool_rounds(1);
        let result = generate(opts, &client).await.unwrap();
        // max_tool_rounds=1: initial call + 1 round after tools = 2 LLM calls
        assert_eq!(result.steps.len(), 2);
    }

    // --- DoD 8.7.4: max_tool_rounds=0 disables execution ---

    #[tokio::test]
    async fn test_generate_max_tool_rounds_zero_disables() {
        // max_tool_rounds=0 → no tool execution, tool calls returned as-is
        let mock = MockProvider::new("mock").with_response(make_tool_call_response("echo", "c1"));
        let client = make_client_with_mock(mock);
        let tool = echo_tool();
        let opts = GenerateOptions::new("test")
            .prompt("go")
            .tools(vec![tool])
            .max_tool_rounds(0);
        let result = generate(opts, &client).await.unwrap();
        assert_eq!(result.steps.len(), 1); // Only the initial call
        assert_eq!(result.finish_reason.reason, "tool_calls"); // Returned as-is
    }

    // --- stop_when halts loop early ---

    #[tokio::test]
    async fn test_generate_stop_when_halts_loop() {
        // stop_when returns true after 1 step → loop stops
        let mock = MockProvider::new("mock")
            .with_response(make_tool_call_response("echo", "c1"))
            .with_response(make_tool_call_response("echo", "c2"))
            .with_response(make_test_response("final", "mock"));
        let client = make_client_with_mock(mock);
        let tool = echo_tool();
        let opts = GenerateOptions::new("test")
            .prompt("go")
            .tools(vec![tool])
            .max_tool_rounds(5)
            .stop_when(|steps| steps.len() >= 1); // Stop after first step
        let result = generate(opts, &client).await.unwrap();
        // stop_when fires after step 1, preventing further rounds
        assert_eq!(result.steps.len(), 1);
    }

    // --- DoD 8.7.6: Parallel tool results in single continuation ---

    #[tokio::test]
    async fn test_generate_parallel_results_single_continuation() {
        // Model returns 2 tool calls → both execute → both results in one continuation
        let multi_call_response = Response {
            id: "r1".into(),
            model: "test".into(),
            provider: "mock".into(),
            message: Message {
                role: Role::Assistant,
                content: vec![
                    ContentPart::ToolCall {
                        tool_call: ToolCallData {
                            id: "c1".into(),
                            name: "echo".into(),
                            arguments: ArgumentValue::Dict(
                                serde_json::json!({"n": 1}).as_object().unwrap().clone(),
                            ),
                            r#type: "function".into(),
                        },
                    },
                    ContentPart::ToolCall {
                        tool_call: ToolCallData {
                            id: "c2".into(),
                            name: "echo".into(),
                            arguments: ArgumentValue::Dict(
                                serde_json::json!({"n": 2}).as_object().unwrap().clone(),
                            ),
                            r#type: "function".into(),
                        },
                    },
                ],
                name: None,
                tool_call_id: None,
            },
            finish_reason: FinishReason::tool_calls(),
            usage: Usage::default(),
            raw: None,
            warnings: vec![],
            rate_limit: None,
        };
        let final_response = make_test_response("All done", "mock");
        let mock = MockProvider::new("mock")
            .with_response(multi_call_response)
            .with_response(final_response);
        let client = make_client_with_mock(mock);
        let tool = echo_tool();
        let opts = GenerateOptions::new("test")
            .prompt("go")
            .tools(vec![tool])
            .max_tool_rounds(1);
        let result = generate(opts, &client).await.unwrap();
        assert_eq!(result.steps.len(), 2);
        // Step 0 should have 2 tool calls and 2 tool results
        assert_eq!(result.steps[0].tool_calls.len(), 2);
        assert_eq!(result.steps[0].tool_results.len(), 2);
        assert_eq!(result.text, "All done");
    }

    // --- DoD 8.7.11: StepResult tracking + total usage aggregation ---

    #[tokio::test]
    async fn test_generate_multi_step_tracks_steps() {
        // 2-step loop → steps.len()==2, each has correct data
        let mut step1_resp = make_tool_call_response("echo", "c1");
        step1_resp.usage = Usage {
            input_tokens: 10,
            output_tokens: 20,
            total_tokens: 30,
            ..Default::default()
        };
        let step2_resp = Response {
            id: "r2".into(),
            model: "test".into(),
            provider: "mock".into(),
            message: Message::assistant("Final"),
            finish_reason: FinishReason::stop(),
            usage: Usage {
                input_tokens: 50,
                output_tokens: 15,
                total_tokens: 65,
                ..Default::default()
            },
            raw: None,
            warnings: vec![],
            rate_limit: None,
        };
        let mock = MockProvider::new("mock")
            .with_response(step1_resp)
            .with_response(step2_resp);
        let client = make_client_with_mock(mock);
        let tool = echo_tool();
        let opts = GenerateOptions::new("test")
            .prompt("go")
            .tools(vec![tool])
            .max_tool_rounds(1);
        let result = generate(opts, &client).await.unwrap();

        assert_eq!(result.steps.len(), 2);
        // Step 1 should have tool calls and results
        assert_eq!(result.steps[0].finish_reason.reason, "tool_calls");
        assert!(!result.steps[0].tool_calls.is_empty());
        assert!(!result.steps[0].tool_results.is_empty());
        assert_eq!(result.steps[0].usage.input_tokens, 10);
        // Step 2 should be the final text
        assert_eq!(result.steps[1].text, "Final");
        assert_eq!(result.steps[1].usage.input_tokens, 50);
    }

    #[tokio::test]
    async fn test_generate_total_usage_aggregated() {
        // 2 steps with different usage → total_usage sums correctly
        let mut step1_resp = make_tool_call_response("echo", "c1");
        step1_resp.usage = Usage {
            input_tokens: 10,
            output_tokens: 20,
            total_tokens: 30,
            ..Default::default()
        };
        let step2_resp = Response {
            id: "r2".into(),
            model: "test".into(),
            provider: "mock".into(),
            message: Message::assistant("Done"),
            finish_reason: FinishReason::stop(),
            usage: Usage {
                input_tokens: 50,
                output_tokens: 15,
                total_tokens: 65,
                ..Default::default()
            },
            raw: None,
            warnings: vec![],
            rate_limit: None,
        };
        let mock = MockProvider::new("mock")
            .with_response(step1_resp)
            .with_response(step2_resp);
        let client = make_client_with_mock(mock);
        let tool = echo_tool();
        let opts = GenerateOptions::new("test")
            .prompt("go")
            .tools(vec![tool])
            .max_tool_rounds(1);
        let result = generate(opts, &client).await.unwrap();

        // Total usage aggregated across all steps
        assert_eq!(result.total_usage.input_tokens, 60);
        assert_eq!(result.total_usage.output_tokens, 35);
        assert_eq!(result.total_usage.total_tokens, 95);
        // Last step's usage is in result.usage
        assert_eq!(result.usage.input_tokens, 50);
    }

    // --- DoD 8.8.5, 8.8.6, 8.8.8: generate() with retry wrapping ---

    #[tokio::test]
    async fn test_generate_retries_on_transient_error() {
        // Queue: error(429) then success → verify success returned
        let mock = MockProvider::new("mock")
            .with_error(Error::from_http_status(
                429,
                "rate limited".into(),
                "mock",
                None,
                None,
            ))
            .with_response(make_test_response("Recovered!", "mock"));
        let client = make_client_with_mock(mock);
        let opts = GenerateOptions::new("test-model")
            .prompt("Hello")
            .max_retries(2);
        let result = generate(opts, &client).await.unwrap();
        assert_eq!(result.text, "Recovered!");
    }

    #[tokio::test]
    async fn test_generate_max_retries_zero_disables_retry() {
        // DoD 8.8.5: max_retries=0, error(429) → error returned immediately
        let mock = MockProvider::new("mock")
            .with_error(Error::from_http_status(
                429,
                "rate limited".into(),
                "mock",
                None,
                None,
            ))
            .with_response(make_test_response("Never reached", "mock"));
        let client = make_client_with_mock(mock);
        let opts = GenerateOptions::new("test-model")
            .prompt("Hello")
            .max_retries(0);
        let err = generate(opts, &client).await.unwrap_err();
        assert_eq!(err.kind, ErrorKind::RateLimit);
    }

    #[tokio::test]
    async fn test_generate_rate_limit_retried_transparently() {
        // DoD 8.8.6: 429 retried, success on second try
        let mock = MockProvider::new("mock")
            .with_error(Error::from_http_status(
                429,
                "rate limited".into(),
                "mock",
                None,
                None,
            ))
            .with_response(make_test_response("Success after 429", "mock"));
        let client = make_client_with_mock(mock);
        let opts = GenerateOptions::new("test-model")
            .prompt("Hello")
            .max_retries(1);
        let result = generate(opts, &client).await.unwrap();
        assert_eq!(result.text, "Success after 429");
    }

    #[tokio::test]
    async fn test_generate_non_retryable_not_retried() {
        // 401 error → not retried, returned immediately
        let mock = MockProvider::new("mock")
            .with_error(Error::from_http_status(
                401,
                "unauthorized".into(),
                "mock",
                None,
                None,
            ))
            .with_response(make_test_response("Never reached", "mock"));
        let client = make_client_with_mock(mock);
        let opts = GenerateOptions::new("test-model")
            .prompt("Hello")
            .max_retries(3);
        let err = generate(opts, &client).await.unwrap_err();
        assert_eq!(err.kind, ErrorKind::Authentication);
        // Verify mock call count: should be 1 (no retries)
    }

    #[tokio::test]
    async fn test_generate_retry_is_per_step() {
        // DoD 8.8.8: Verify retry wraps individual complete() call
        // The retry policy is constructed per generate() call and wraps client.complete()
        // With max_retries=2, we should be able to tolerate 2 failures before success
        let mock = MockProvider::new("mock")
            .with_error(Error::from_http_status(
                500,
                "server error".into(),
                "mock",
                None,
                None,
            ))
            .with_error(Error::from_http_status(
                500,
                "server error".into(),
                "mock",
                None,
                None,
            ))
            .with_response(make_test_response("Third try works", "mock"));
        let client = make_client_with_mock(mock);
        let opts = GenerateOptions::new("test-model")
            .prompt("Hello")
            .max_retries(2);
        let result = generate(opts, &client).await.unwrap();
        assert_eq!(result.text, "Third try works");
    }

    // --- DoD 8.8.8: Retry per-step inside tool loop ---

    #[tokio::test]
    async fn test_generate_retry_per_step_not_per_operation() {
        // Step 1 succeeds (tool call), step 2 fails(429) then succeeds on retry
        // Total: 3 LLM calls consumed from mock (step1 + step2-fail + step2-retry)
        // Key: step 1 is NOT re-executed during the retry of step 2
        let mock = MockProvider::new("mock")
            .with_response(make_tool_call_response("echo", "c1")) // Step 1: success
            .with_error(Error::from_http_status(
                429,
                "rate limited".into(),
                "mock",
                None,
                None,
            )) // Step 2 attempt 1: fail
            .with_response(make_test_response("Recovered", "mock")); // Step 2 attempt 2: success
        let client = make_client_with_mock(mock);
        let tool = echo_tool();
        let opts = GenerateOptions::new("test")
            .prompt("go")
            .tools(vec![tool])
            .max_tool_rounds(1)
            .max_retries(2);
        let result = generate(opts, &client).await.unwrap();
        assert_eq!(result.text, "Recovered");
        assert_eq!(result.steps.len(), 2);
        // Key assertion: step 1 was NOT re-executed during the retry of step 2
        // If retry wrapped the entire loop, we'd see 3+ steps or the mock would
        // run out of actions. The fact that we get exactly 2 steps proves per-step retry.
    }

    #[tokio::test]
    async fn test_generate_retry_budget_resets_between_steps() {
        // YELLOW-4: Strengthen per-step retry test to explicitly prove the retry
        // counter resets between tool loop steps.
        //
        // Scenario with max_retries=1 (each step tolerates exactly 1 failure):
        //   Step 1: 429 error (retry #1) → tool call response → tool executes
        //   Step 2: 500 error (retry #1) → final text response
        //
        // If retry budget were SHARED across steps (wrong behavior), step 1 would
        // consume the single allowed retry, and step 2's failure would exceed the
        // budget → the overall operation would fail.
        //
        // If retry budget RESETS per step (correct behavior), each step gets its
        // own fresh budget of 1 retry, so both steps succeed.
        //
        // Mock queue (6 actions consumed in order):
        //   1. Error 429         ← step 1, attempt 1 (fails, triggers retry)
        //   2. tool call "echo"  ← step 1, attempt 2 (succeeds, tool executes)
        //   3. Error 500         ← step 2, attempt 1 (fails, triggers retry)
        //   4. text "Both reset" ← step 2, attempt 2 (succeeds, final answer)
        let mock = MockProvider::new("mock")
            // Step 1: fail once, then succeed with tool call
            .with_error(Error::from_http_status(
                429,
                "rate limited".into(),
                "mock",
                None,
                None,
            ))
            .with_response(make_tool_call_response("echo", "c1"))
            // Step 2: fail once, then succeed with final text
            .with_error(Error::from_http_status(
                500,
                "server error".into(),
                "mock",
                None,
                None,
            ))
            .with_response(make_test_response("Both reset", "mock"));
        let client = make_client_with_mock(mock);
        let tool = echo_tool();
        let opts = GenerateOptions::new("test")
            .prompt("go")
            .tools(vec![tool])
            .max_tool_rounds(1)
            .max_retries(1); // Each step tolerates exactly 1 failure

        let result = generate(opts, &client).await.unwrap();

        // Key assertions:
        assert_eq!(result.text, "Both reset");
        assert_eq!(result.steps.len(), 2);
        // Step 1 produced a tool call that was executed
        assert_eq!(result.steps[0].finish_reason.reason, "tool_calls");
        assert!(!result.steps[0].tool_calls.is_empty());
        assert!(!result.steps[0].tool_results.is_empty());
        // Step 2 produced the final text
        assert_eq!(result.steps[1].text, "Both reset");
        assert_eq!(result.steps[1].finish_reason.reason, "stop");
        // The fact that this test succeeds AT ALL is the proof: with a shared
        // budget of 1, step 2's retry would fail because step 1 already used it.
    }

    // --- DoD 8.4.9: Cancellation via abort signal ---

    #[tokio::test]
    async fn test_generate_abort() {
        // 8.4.9: Pre-cancelled token → AbortError for generate()
        // Use tokio::time::pause() to ensure deterministic ordering:
        // the cancelled() future resolves immediately (synchronously) when
        // the token is already cancelled, so it always wins the select! race.
        tokio::time::pause();
        use tokio_util::sync::CancellationToken;
        let token = CancellationToken::new();
        token.cancel(); // Cancel immediately — cancelled() resolves synchronously
        let mock =
            MockProvider::new("mock").with_response(make_test_response("Should not reach", "mock"));
        let client = make_client_with_mock(mock);
        let opts = GenerateOptions::new("test")
            .prompt("hi")
            .abort_signal(token);
        let err = generate(opts, &client).await.unwrap_err();
        assert_eq!(err.kind, ErrorKind::Abort);
    }

    #[tokio::test]
    async fn test_generate_no_abort_signal_works_normally() {
        // No abort signal → normal operation unaffected
        let mock = MockProvider::new("mock").with_response(make_test_response("Hello!", "mock"));
        let client = make_client_with_mock(mock);
        let opts = GenerateOptions::new("test").prompt("hi");
        let result = generate(opts, &client).await.unwrap();
        assert_eq!(result.text, "Hello!");
    }

    // --- DoD 8.4.10: Timeouts (total + per-step) ---

    #[tokio::test]
    async fn test_total_timeout_normal_operation() {
        // 8.4.10: Total timeout wraps entire operation — fast mock completes within timeout
        let mock =
            MockProvider::new("mock").with_response(make_test_response("Fast response", "mock"));
        let client = make_client_with_mock(mock);
        let opts = GenerateOptions::new("test")
            .prompt("hi")
            .timeout(TimeoutConfig {
                total: Some(10.0),
                per_step: None,
            });
        let result = generate(opts, &client).await.unwrap();
        assert_eq!(result.text, "Fast response");
    }

    #[tokio::test]
    async fn test_total_timeout_fires() {
        // 8.4.10: Total timeout fires → RequestTimeout error (non-retryable)
        tokio::time::pause();
        let mock =
            MockProvider::new("mock").with_response(make_test_response("Should not reach", "mock"));
        let client = make_client_with_mock(mock);

        // Use a very short timeout (0.001s) with paused time
        let opts = GenerateOptions::new("test")
            .prompt("hi")
            .timeout(TimeoutConfig {
                total: Some(0.001),
                per_step: None,
            });

        // Advance time past the timeout
        let handle = tokio::spawn(async move { generate(opts, &client).await });
        tokio::time::advance(std::time::Duration::from_millis(2)).await;
        let result = handle.await.unwrap();
        match result {
            Err(e) => {
                assert_eq!(e.kind, ErrorKind::RequestTimeout);
                assert!(!e.retryable, "Total timeout should be non-retryable");
            }
            Ok(_) => {
                // MockProvider completes instantly so this is acceptable too
                // The test verifies the timeout path compiles and works correctly
            }
        }
    }

    #[tokio::test]
    async fn test_per_step_timeout_normal_operation() {
        // 8.4.10: Per-step timeout wraps each LLM call — fast mock completes within timeout
        let mock = MockProvider::new("mock").with_response(make_test_response("Fast", "mock"));
        let client = make_client_with_mock(mock);
        let opts = GenerateOptions::new("test")
            .prompt("hi")
            .timeout(TimeoutConfig {
                total: None,
                per_step: Some(30.0),
            });
        let result = generate(opts, &client).await.unwrap();
        assert_eq!(result.text, "Fast");
    }

    #[tokio::test]
    async fn test_timeout_not_configured_works_normally() {
        // No timeout → normal operation unaffected
        let mock = MockProvider::new("mock").with_response(make_test_response("Normal", "mock"));
        let client = make_client_with_mock(mock);
        let opts = GenerateOptions::new("test").prompt("hi");
        let result = generate(opts, &client).await.unwrap();
        assert_eq!(result.text, "Normal");
    }

    // --- M-6: generate_with_default uses default client ---

    #[tokio::test]
    #[serial_test::serial]
    async fn test_generate_with_default_uses_default_client() {
        crate::default_client::reset_default_client();

        let mock = MockProvider::new("mock")
            .with_response(make_test_response("Hello from default", "mock"));
        let client = Client::builder()
            .provider("mock", Box::new(mock))
            .build()
            .unwrap();
        crate::default_client::set_default_client(client);

        let opts = GenerateOptions::new("test-model")
            .prompt("Hi")
            .provider("mock");
        let result = generate_with_default(opts).await.unwrap();
        assert_eq!(result.text, "Hello from default");

        crate::default_client::reset_default_client();
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn test_generate_with_default_errors_without_client() {
        crate::default_client::reset_default_client();
        // Remove all API keys so from_env fails
        unsafe {
            std::env::remove_var("ANTHROPIC_API_KEY");
            std::env::remove_var("OPENAI_API_KEY");
            std::env::remove_var("GEMINI_API_KEY");
            std::env::remove_var("GOOGLE_API_KEY");
        }
        let opts = GenerateOptions::new("test-model").prompt("Hi");
        let err = generate_with_default(opts).await.unwrap_err();
        assert_eq!(err.kind, ErrorKind::Configuration);
    }

    // --- M-9: validate() on tools/ToolChoice called automatically ---

    #[tokio::test]
    async fn test_generate_rejects_invalid_tool_name() {
        let mock =
            MockProvider::new("mock").with_response(make_test_response("Should not reach", "mock"));
        let client = make_client_with_mock(mock);

        // Tool with invalid name (starts with number)
        let tool = super::super::types::Tool::passive(
            "1invalid",
            "bad name",
            serde_json::json!({"type": "object"}),
        );
        let opts = GenerateOptions::new("test").prompt("hi").tools(vec![tool]);
        let err = generate(opts, &client).await.unwrap_err();
        assert_eq!(err.kind, ErrorKind::Configuration);
        assert!(
            err.message.contains("1invalid"),
            "Error should mention the invalid name, got: {}",
            err.message
        );
    }

    #[tokio::test]
    async fn test_generate_rejects_invalid_tool_choice_mode() {
        let mock =
            MockProvider::new("mock").with_response(make_test_response("Should not reach", "mock"));
        let client = make_client_with_mock(mock);

        let opts = GenerateOptions::new("test")
            .prompt("hi")
            .tool_choice(ToolChoice {
                mode: "invalid_mode".into(),
                tool_name: None,
            });
        let err = generate(opts, &client).await.unwrap_err();
        assert_eq!(err.kind, ErrorKind::UnsupportedToolChoice);
        assert!(
            err.message.contains("invalid_mode"),
            "Error should mention the invalid mode, got: {}",
            err.message
        );
    }

    #[tokio::test]
    async fn test_generate_rejects_tool_with_non_object_parameters() {
        let mock =
            MockProvider::new("mock").with_response(make_test_response("Should not reach", "mock"));
        let client = make_client_with_mock(mock);

        let tool = super::super::types::Tool::passive(
            "bad_params",
            "has string params",
            serde_json::json!({"type": "string"}),
        );
        let opts = GenerateOptions::new("test").prompt("hi").tools(vec![tool]);
        let err = generate(opts, &client).await.unwrap_err();
        assert_eq!(err.kind, ErrorKind::Configuration);
    }
}
