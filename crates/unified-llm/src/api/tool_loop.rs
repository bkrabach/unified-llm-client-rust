// api/tool_loop.rs — tool execution utilities (Layer 4).
//
// Provides `execute_all_tools()` for parallel tool execution with:
// - JSON Schema validation of arguments (H-6)
// - Repair pathway for invalid arguments (H-7)
// - Context injection for context-aware tools (H-5)
// - Preserved result ordering for mixed active/passive tools (M-8)

use futures::future::join_all;
use serde_json::json;
use unified_llm_types::content::{ArgumentValue, ToolCallData};
use unified_llm_types::{Message, ToolResult};

use super::types::{RepairToolCallFn, Tool, ToolContext};
use tokio_util::sync::CancellationToken;

/// Parse tool call arguments to a serde_json::Value.
fn parse_arguments(call: &ToolCallData) -> Result<serde_json::Value, String> {
    match &call.arguments {
        ArgumentValue::Dict(map) => Ok(serde_json::Value::Object(map.clone())),
        ArgumentValue::Raw(s) => {
            serde_json::from_str(s).map_err(|e| format!("Failed to parse arguments: {}", e))
        }
    }
}

/// Validate arguments against a tool's parameter schema.
/// Returns Ok(()) if valid, Err(error_message) if invalid.
fn validate_against_schema(
    schema: &serde_json::Value,
    args: &serde_json::Value,
) -> Result<(), String> {
    if jsonschema::is_valid(schema, args) {
        Ok(())
    } else {
        let error_detail = jsonschema::validate(schema, args)
            .err()
            .map(|e| e.to_string())
            .unwrap_or_else(|| "unknown validation error".into());
        Err(format!("Schema validation failed: {}", error_detail))
    }
}

/// Execute all tool calls concurrently with schema validation, repair, and context.
///
/// Spec references: §5.7 (parallel execution), §5.8 (validation)
///
/// Guarantees:
/// 1. All tool calls executed concurrently (join_all)
/// 2. Results preserve ordering (same order as tool_calls) — including passive tools (M-8)
/// 3. Partial failures are graceful (is_error=true, not exception)
/// 4. Unknown tools get error results, not exceptions
/// 5. Arguments are validated against the tool's parameter schema (H-6)
/// 6. Invalid arguments can be repaired via repair_fn (H-7)
/// 7. Context-aware tools receive ToolContext (H-5)
/// 8. Passive tools return a placeholder ToolResult to preserve ordering (M-8)
pub async fn execute_all_tools(
    tools: &[Tool],
    tool_calls: &[&ToolCallData],
    messages: &[Message],
    abort_signal: &Option<CancellationToken>,
    repair_fn: Option<&RepairToolCallFn>,
) -> Vec<ToolResult> {
    let futures: Vec<_> = tool_calls
        .iter()
        .map(|call| {
            let tool = tools.iter().find(|t| t.definition.name == call.name);
            let messages = messages.to_vec();
            let abort_signal = abort_signal.clone();
            async move {
                match tool {
                    Some(t) if t.is_active() => {
                        // Parse arguments
                        let args = match parse_arguments(call) {
                            Ok(v) => v,
                            Err(e) => {
                                return ToolResult {
                                    tool_call_id: call.id.clone(),
                                    content: json!(e),
                                    is_error: true,
                                };
                            }
                        };

                        // H-6: Validate arguments against schema
                        let args = match validate_against_schema(&t.definition.parameters, &args) {
                            Ok(()) => args,
                            Err(validation_error) => {
                                // H-7: Try repair if repair_fn is provided
                                if let Some(repair) = repair_fn {
                                    if let Some(repaired) = repair(call, &validation_error) {
                                        // Re-validate the repaired arguments
                                        match validate_against_schema(
                                            &t.definition.parameters,
                                            &repaired,
                                        ) {
                                            Ok(()) => repaired,
                                            Err(e2) => {
                                                return ToolResult {
                                                    tool_call_id: call.id.clone(),
                                                    content: json!(format!(
                                                        "Repair failed validation: {}",
                                                        e2
                                                    )),
                                                    is_error: true,
                                                };
                                            }
                                        }
                                    } else {
                                        // Repair returned None — propagate original error
                                        return ToolResult {
                                            tool_call_id: call.id.clone(),
                                            content: json!(validation_error),
                                            is_error: true,
                                        };
                                    }
                                } else {
                                    // No repair fn — return validation error
                                    return ToolResult {
                                        tool_call_id: call.id.clone(),
                                        content: json!(validation_error),
                                        is_error: true,
                                    };
                                }
                            }
                        };

                        // H-5: Execute with context if handler supports it, else plain
                        // M-14: Wrap execution in tokio::select! to propagate cancellation
                        let exec_future = if let Some(ref ctx_handler) = t.execute_with_context {
                            let ctx = ToolContext {
                                messages,
                                tool_call_id: call.id.clone(),
                                abort_signal: abort_signal.clone(),
                            };
                            ctx_handler(args, ctx)
                        } else if let Some(ref handler) = t.execute {
                            handler(args)
                        } else {
                            // Shouldn't happen (is_active() checked), but be safe
                            return ToolResult {
                                tool_call_id: call.id.clone(),
                                content: json!("Tool marked active but has no handler"),
                                is_error: true,
                            };
                        };

                        let exec_result = if let Some(ref token) = abort_signal {
                            tokio::select! {
                                result = exec_future => result,
                                _ = token.cancelled() => {
                                    Err(unified_llm_types::Error::abort())
                                }
                            }
                        } else {
                            exec_future.await
                        };

                        match exec_result {
                            Ok(result) => ToolResult {
                                tool_call_id: call.id.clone(),
                                content: result,
                                is_error: false,
                            },
                            Err(e) => ToolResult {
                                tool_call_id: call.id.clone(),
                                content: json!(e.message),
                                is_error: true,
                            },
                        }
                    }
                    Some(_) => {
                        // M-8: Passive tool — return placeholder result to preserve ordering.
                        // The model expects results for all tool calls in order.
                        ToolResult {
                            tool_call_id: call.id.clone(),
                            content: json!("Tool call returned to caller (passive tool)"),
                            is_error: false,
                        }
                    }
                    None => {
                        // Unknown tool
                        ToolResult {
                            tool_call_id: call.id.clone(),
                            content: json!(format!("Unknown tool: {}", call.name)),
                            is_error: true,
                        }
                    }
                }
            }
        })
        .collect();

    // M-8: No .flatten() — all results preserved in order
    join_all(futures).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::types::Tool;
    use unified_llm_types::content::{ArgumentValue, ToolCallData};
    use unified_llm_types::Error;

    fn make_tool_call(id: &str, name: &str, args: serde_json::Value) -> ToolCallData {
        ToolCallData {
            id: id.into(),
            name: name.into(),
            arguments: ArgumentValue::Dict(args.as_object().unwrap().clone()),
            r#type: "function".into(),
        }
    }

    // --- DoD 8.7.5: Parallel execution ---

    #[tokio::test]
    async fn test_execute_all_tools_parallel() {
        use std::sync::atomic::{AtomicU32, Ordering};
        use std::sync::Arc;
        let counter = Arc::new(AtomicU32::new(0));
        let c = counter.clone();
        let tool = Tool::active(
            "inc",
            "increment",
            serde_json::json!({"type": "object"}),
            move |_| {
                let c = c.clone();
                Box::pin(async move {
                    c.fetch_add(1, Ordering::SeqCst);
                    Ok(serde_json::json!("done"))
                })
            },
        );
        let calls = vec![
            make_tool_call("c1", "inc", serde_json::json!({})),
            make_tool_call("c2", "inc", serde_json::json!({})),
        ];
        let call_refs: Vec<&ToolCallData> = calls.iter().collect();
        let results = execute_all_tools(&[tool], &call_refs, &[], &None, None).await;
        assert_eq!(results.len(), 2);
        assert_eq!(counter.load(Ordering::SeqCst), 2);
        // Results preserve ordering
        assert_eq!(results[0].tool_call_id, "c1");
        assert_eq!(results[1].tool_call_id, "c2");
    }

    // --- DoD 8.7.7: Tool execution errors → is_error=true ---

    #[tokio::test]
    async fn test_execute_tool_error_returns_is_error() {
        let tool = Tool::active(
            "fail",
            "fails",
            serde_json::json!({"type": "object"}),
            |_| Box::pin(async { Err(Error::configuration("tool broke")) }),
        );
        let calls = vec![make_tool_call("c1", "fail", serde_json::json!({}))];
        let call_refs: Vec<&ToolCallData> = calls.iter().collect();
        let results = execute_all_tools(&[tool], &call_refs, &[], &None, None).await;
        assert_eq!(results.len(), 1);
        assert!(results[0].is_error);
        assert_eq!(results[0].tool_call_id, "c1");
    }

    // --- DoD 8.7.8: Unknown tool → error result ---

    #[tokio::test]
    async fn test_execute_unknown_tool_returns_error() {
        let calls = vec![make_tool_call("c1", "nonexistent", serde_json::json!({}))];
        let call_refs: Vec<&ToolCallData> = calls.iter().collect();
        let results = execute_all_tools(&[], &call_refs, &[], &None, None).await;
        assert_eq!(results.len(), 1);
        assert!(results[0].is_error);
        assert!(results[0].content.to_string().contains("Unknown tool"));
    }

    // --- DoD 8.7.10: Argument parsing ---

    #[tokio::test]
    async fn test_execute_argument_parsing_dict() {
        // Dict arguments are passed through as serde_json::Value::Object
        let tool = Tool::active(
            "echo",
            "echo",
            serde_json::json!({"type": "object"}),
            |args| Box::pin(async move { Ok(args) }),
        );
        let calls = vec![make_tool_call(
            "c1",
            "echo",
            serde_json::json!({"key": "value"}),
        )];
        let call_refs: Vec<&ToolCallData> = calls.iter().collect();
        let results = execute_all_tools(&[tool], &call_refs, &[], &None, None).await;
        assert!(!results[0].is_error);
        assert_eq!(results[0].content["key"], "value");
    }

    #[tokio::test]
    async fn test_execute_argument_parsing_raw() {
        // Raw string arguments are parsed as JSON before passing to handler
        let tool = Tool::active(
            "echo",
            "echo",
            serde_json::json!({"type": "object"}),
            |args| Box::pin(async move { Ok(args) }),
        );
        let raw_call = ToolCallData {
            id: "c2".into(),
            name: "echo".into(),
            arguments: ArgumentValue::Raw(r#"{"raw_key": "raw_value"}"#.into()),
            r#type: "function".into(),
        };
        let call_refs: Vec<&ToolCallData> = vec![&raw_call];
        let results = execute_all_tools(&[tool], &call_refs, &[], &None, None).await;
        assert!(!results[0].is_error);
        assert_eq!(results[0].content["raw_key"], "raw_value");
    }

    // --- Partial failure: some succeed, some fail ---

    #[tokio::test]
    async fn test_execute_partial_failure() {
        let good_tool = Tool::active(
            "good",
            "works",
            serde_json::json!({"type": "object"}),
            |_| Box::pin(async { Ok(serde_json::json!("success")) }),
        );
        let bad_tool = Tool::active(
            "bad",
            "fails",
            serde_json::json!({"type": "object"}),
            |_| Box::pin(async { Err(Error::configuration("broken")) }),
        );
        let calls = vec![
            make_tool_call("c1", "good", serde_json::json!({})),
            make_tool_call("c2", "bad", serde_json::json!({})),
        ];
        let tools = vec![good_tool, bad_tool];
        let call_refs: Vec<&ToolCallData> = calls.iter().collect();
        let results = execute_all_tools(&tools, &call_refs, &[], &None, None).await;
        assert_eq!(results.len(), 2);
        assert!(!results[0].is_error);
        assert!(results[1].is_error);
    }

    // --- H-6: JSON Schema validation of tool arguments ---

    #[tokio::test]
    async fn test_tool_arguments_validated_against_schema() {
        let tool = Tool::active(
            "get_weather",
            "Get weather",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "city": { "type": "string" }
                },
                "required": ["city"]
            }),
            |args| Box::pin(async move { Ok(args) }),
        );

        // Tool call with invalid arguments (missing required "city")
        let bad_call = ToolCallData {
            id: "call_1".into(),
            name: "get_weather".into(),
            arguments: ArgumentValue::Dict(
                serde_json::json!({"invalid_field": 123})
                    .as_object()
                    .unwrap()
                    .clone(),
            ),
            r#type: "function".into(),
        };

        let call_refs: Vec<&ToolCallData> = vec![&bad_call];
        let results = execute_all_tools(&[tool], &call_refs, &[], &None, None).await;

        assert_eq!(results.len(), 1);
        assert!(
            results[0].is_error,
            "Invalid arguments should produce an error result"
        );
        assert!(
            results[0].content.to_string().contains("city"),
            "Error should mention the missing required field, got: {}",
            results[0].content
        );
    }

    #[tokio::test]
    async fn test_tool_valid_arguments_pass_schema_validation() {
        let tool = Tool::active(
            "get_weather",
            "Get weather",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "city": { "type": "string" }
                },
                "required": ["city"]
            }),
            |args| Box::pin(async move { Ok(args) }),
        );

        let good_call = make_tool_call(
            "call_1",
            "get_weather",
            serde_json::json!({"city": "Paris"}),
        );
        let call_refs: Vec<&ToolCallData> = vec![&good_call];
        let results = execute_all_tools(&[tool], &call_refs, &[], &None, None).await;

        assert_eq!(results.len(), 1);
        assert!(
            !results[0].is_error,
            "Valid arguments should succeed, got: {}",
            results[0].content
        );
        assert_eq!(results[0].content["city"], "Paris");
    }

    // --- H-7: repair_tool_call pathway ---

    #[tokio::test]
    async fn test_repair_tool_call_on_schema_failure() {
        let tool = Tool::active(
            "get_weather",
            "Get weather",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "city": { "type": "string" }
                },
                "required": ["city"]
            }),
            |args| Box::pin(async move { Ok(args) }),
        );

        // Invalid call (missing "city")
        let bad_call = ToolCallData {
            id: "call_1".into(),
            name: "get_weather".into(),
            arguments: ArgumentValue::Dict(
                serde_json::json!({"wrong": "field"})
                    .as_object()
                    .unwrap()
                    .clone(),
            ),
            r#type: "function".into(),
        };

        // Repair function that returns corrected arguments
        let repair_fn: RepairToolCallFn = std::sync::Arc::new(
            |_call: &ToolCallData, _error: &str| -> Option<serde_json::Value> {
                Some(serde_json::json!({"city": "Paris"}))
            },
        );

        let call_refs: Vec<&ToolCallData> = vec![&bad_call];
        let results = execute_all_tools(&[tool], &call_refs, &[], &None, Some(&repair_fn)).await;

        assert_eq!(results.len(), 1);
        assert!(
            !results[0].is_error,
            "Repair should have fixed the arguments, got: {}",
            results[0].content
        );
        assert_eq!(results[0].content["city"], "Paris");
    }

    #[tokio::test]
    async fn test_repair_returns_none_propagates_error() {
        let tool = Tool::active(
            "get_weather",
            "Get weather",
            serde_json::json!({
                "type": "object",
                "properties": { "city": { "type": "string" } },
                "required": ["city"]
            }),
            |args| Box::pin(async move { Ok(args) }),
        );

        let bad_call = ToolCallData {
            id: "call_1".into(),
            name: "get_weather".into(),
            arguments: ArgumentValue::Dict(serde_json::json!({}).as_object().unwrap().clone()),
            r#type: "function".into(),
        };

        // Repair function that returns None (can't fix it)
        let repair_fn: RepairToolCallFn = std::sync::Arc::new(
            |_call: &ToolCallData, _error: &str| -> Option<serde_json::Value> { None },
        );

        let call_refs: Vec<&ToolCallData> = vec![&bad_call];
        let results = execute_all_tools(&[tool], &call_refs, &[], &None, Some(&repair_fn)).await;

        assert_eq!(results.len(), 1);
        assert!(
            results[0].is_error,
            "Repair returned None — error should propagate"
        );
    }

    // --- H-5: Tool context injection ---

    #[tokio::test]
    async fn test_tool_handler_receives_context() {
        use std::sync::{Arc, Mutex};

        let received_context: Arc<Mutex<Option<ToolContext>>> = Arc::new(Mutex::new(None));
        let ctx_clone = received_context.clone();

        let tool = Tool::active_with_context(
            "echo",
            "echoes input",
            serde_json::json!({"type": "object"}),
            move |args, ctx| {
                let ctx_clone = ctx_clone.clone();
                Box::pin(async move {
                    *ctx_clone.lock().unwrap() = Some(ctx);
                    Ok(args)
                })
            },
        );

        let calls = vec![make_tool_call("c1", "echo", serde_json::json!({}))];
        let call_refs: Vec<&ToolCallData> = calls.iter().collect();
        let messages = vec![Message::user("hello")];

        execute_all_tools(&[tool], &call_refs, &messages, &None, None).await;

        let ctx = received_context.lock().unwrap();
        assert!(ctx.is_some(), "Tool handler should receive ToolContext");
        let ctx = ctx.as_ref().unwrap();
        assert_eq!(ctx.tool_call_id, "c1");
        assert_eq!(ctx.messages.len(), 1);
    }

    // --- M-8: Mixed active/passive tool result ordering ---

    #[tokio::test]
    async fn test_mixed_active_passive_preserves_ordering() {
        let active_tool = Tool::active(
            "active",
            "active tool",
            serde_json::json!({"type": "object"}),
            |_| Box::pin(async { Ok(serde_json::json!("active_result")) }),
        );
        let passive_tool = Tool::passive(
            "passive",
            "passive tool",
            serde_json::json!({"type": "object"}),
        );

        // Call order: passive, active, passive
        let calls = vec![
            make_tool_call("c1", "passive", serde_json::json!({})),
            make_tool_call("c2", "active", serde_json::json!({})),
            make_tool_call("c3", "passive", serde_json::json!({})),
        ];
        let tools = vec![active_tool, passive_tool];
        let call_refs: Vec<&ToolCallData> = calls.iter().collect();
        let results = execute_all_tools(&tools, &call_refs, &[], &None, None).await;

        // M-8: All 3 results must be present and in order
        assert_eq!(
            results.len(),
            3,
            "All tool calls must produce results (no .flatten() dropping)"
        );
        assert_eq!(results[0].tool_call_id, "c1");
        assert_eq!(results[1].tool_call_id, "c2");
        assert_eq!(results[2].tool_call_id, "c3");
        // Active tool result should contain the actual result
        assert_eq!(results[1].content, serde_json::json!("active_result"));
        // Passive tool results should not be errors
        assert!(!results[0].is_error);
        assert!(!results[2].is_error);
    }

    // --- M-14: Cancellation propagates through tool execution ---

    #[tokio::test]
    async fn test_cancellation_stops_tool_execution() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        let completed = Arc::new(AtomicBool::new(false));
        let completed_clone = completed.clone();

        let tool = Tool::active(
            "slow",
            "a slow tool",
            serde_json::json!({"type": "object"}),
            move |_| {
                let completed_clone = completed_clone.clone();
                Box::pin(async move {
                    // Simulate a long-running tool
                    tokio::time::sleep(std::time::Duration::from_secs(10)).await;
                    completed_clone.store(true, Ordering::SeqCst);
                    Ok(serde_json::json!("done"))
                })
            },
        );

        let token = CancellationToken::new();
        let abort_signal = Some(token.clone());

        let calls = vec![make_tool_call("c1", "slow", serde_json::json!({}))];
        let call_refs: Vec<&ToolCallData> = calls.iter().collect();

        // Cancel after a short delay
        let token_clone = token.clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            token_clone.cancel();
        });

        let start = std::time::Instant::now();
        let results = execute_all_tools(&[tool], &call_refs, &[], &abort_signal, None).await;
        let elapsed = start.elapsed();

        assert_eq!(results.len(), 1);
        assert!(
            results[0].is_error,
            "Cancelled tool should produce an error"
        );
        assert!(
            elapsed < std::time::Duration::from_secs(2),
            "Should cancel quickly, took {:?}",
            elapsed
        );
        assert!(
            !completed.load(Ordering::SeqCst),
            "Tool should not have completed"
        );
    }

    #[tokio::test]
    async fn test_no_cancellation_token_runs_normally() {
        // Without a cancellation token, tools run to completion
        let tool = Tool::active(
            "fast",
            "a fast tool",
            serde_json::json!({"type": "object"}),
            |_| Box::pin(async { Ok(serde_json::json!("fast_result")) }),
        );

        let calls = vec![make_tool_call("c1", "fast", serde_json::json!({}))];
        let call_refs: Vec<&ToolCallData> = calls.iter().collect();
        let results = execute_all_tools(&[tool], &call_refs, &[], &None, None).await;

        assert_eq!(results.len(), 1);
        assert!(!results[0].is_error);
        assert_eq!(results[0].content, serde_json::json!("fast_result"));
    }

    #[tokio::test]
    async fn test_cancellation_with_context_aware_tool() {
        let tool = Tool::active_with_context(
            "ctx_slow",
            "context-aware slow tool",
            serde_json::json!({"type": "object"}),
            move |_, _ctx| {
                Box::pin(async move {
                    tokio::time::sleep(std::time::Duration::from_secs(10)).await;
                    Ok(serde_json::json!("done"))
                })
            },
        );

        let token = CancellationToken::new();
        let abort_signal = Some(token.clone());

        let calls = vec![make_tool_call("c1", "ctx_slow", serde_json::json!({}))];
        let call_refs: Vec<&ToolCallData> = calls.iter().collect();

        // Cancel immediately
        token.cancel();

        let results = execute_all_tools(&[tool], &call_refs, &[], &abort_signal, None).await;

        assert_eq!(results.len(), 1);
        assert!(
            results[0].is_error,
            "Cancelled context-aware tool should produce an error"
        );
    }
}
