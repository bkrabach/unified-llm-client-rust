// api/generate_types.rs — GenerateResult and StepResult (Layer 4).
//
// These types represent the result of high-level generate() operations.
// They belong in Layer 4 (unified-llm), not Layer 1 (unified-llm-types),
// because they change with the high-level API and should not force semver
// bumps on the types crate used by adapter authors.

use serde::{Deserialize, Serialize};

use unified_llm_types::error::Error;
use unified_llm_types::response::{FinishReason, Response, Usage, Warning};
use unified_llm_types::tool::{ToolCall, ToolResult};

/// Result of a single step in a multi-step generation (one LLM call).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepResult {
    /// Concatenated text output from this step.
    pub text: String,
    /// Reasoning/thinking text from this step, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    /// Tool calls requested by the model in this step.
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
    /// Tool results from executing tool calls in this step.
    #[serde(default)]
    pub tool_results: Vec<ToolResult>,
    /// Why the model stopped generating.
    pub finish_reason: FinishReason,
    /// Token usage for this step.
    pub usage: Usage,
    /// The full provider response for this step.
    pub response: Response,
    /// Any warnings from this step.
    #[serde(default)]
    pub warnings: Vec<Warning>,
}

/// Result of a complete generation operation (possibly multi-step).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateResult {
    /// Final concatenated text output.
    pub text: String,
    /// Final reasoning/thinking text, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    /// Tool calls from the final step.
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
    /// Tool results from the final step.
    #[serde(default)]
    pub tool_results: Vec<ToolResult>,
    /// Finish reason from the final step.
    pub finish_reason: FinishReason,
    /// Usage from the final step.
    pub usage: Usage,
    /// Aggregated usage across all steps.
    pub total_usage: Usage,
    /// All individual step results.
    pub steps: Vec<StepResult>,
    /// The final provider response.
    pub response: Response,
    /// Parsed structured output, if applicable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<serde_json::Value>,
}

impl GenerateResult {
    /// Construct a GenerateResult from a sequence of step results.
    /// Aggregates usage across all steps. Takes the final step's data
    /// as the top-level result.
    ///
    /// Returns an error if `steps` is empty.
    #[allow(clippy::result_large_err)]
    pub fn from_steps(steps: Vec<StepResult>) -> Result<Self, Error> {
        if steps.is_empty() {
            return Err(Error::configuration(
                "from_steps requires at least one step",
            ));
        }

        let total_usage = steps
            .iter()
            .map(|s| s.usage.clone())
            .reduce(|a, b| a + b)
            .unwrap_or_default();

        // Clone from last step before moving steps into the result
        let last = steps.last().unwrap();
        let text = last.text.clone();
        let reasoning = last.reasoning.clone();
        let tool_calls = last.tool_calls.clone();
        let tool_results = last.tool_results.clone();
        let finish_reason = last.finish_reason.clone();
        let usage = last.usage.clone();
        let response = last.response.clone();

        Ok(Self {
            text,
            reasoning,
            tool_calls,
            tool_results,
            finish_reason,
            usage,
            total_usage,
            steps,
            response,
            output: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use unified_llm_types::message::Message;

    fn make_test_response(text: &str) -> Response {
        Response {
            id: "resp_test".into(),
            model: "test".into(),
            provider: "test".into(),
            message: Message::assistant(text),
            finish_reason: FinishReason::stop(),
            usage: Usage::default(),
            raw: None,
            warnings: vec![],
            rate_limit: None,
        }
    }

    fn make_step(input: u32, output: u32) -> StepResult {
        StepResult {
            text: "text".into(),
            reasoning: None,
            tool_calls: vec![],
            tool_results: vec![],
            finish_reason: FinishReason::stop(),
            usage: Usage {
                input_tokens: input,
                output_tokens: output,
                total_tokens: input + output,
                ..Default::default()
            },
            response: make_test_response("text"),
            warnings: vec![],
        }
    }

    #[test]
    fn test_generate_result_from_single_step() {
        let step = StepResult {
            text: "Hello".into(),
            reasoning: None,
            tool_calls: vec![],
            tool_results: vec![],
            finish_reason: FinishReason::stop(),
            usage: Usage {
                input_tokens: 10,
                output_tokens: 5,
                total_tokens: 15,
                ..Default::default()
            },
            response: make_test_response("Hello"),
            warnings: vec![],
        };
        let result = GenerateResult::from_steps(vec![step]).unwrap();
        assert_eq!(result.text, "Hello");
        assert_eq!(result.total_usage.total_tokens, 15);
        assert_eq!(result.steps.len(), 1);
        assert_eq!(result.finish_reason.reason, "stop");
    }

    #[test]
    fn test_generate_result_total_usage_aggregates() {
        let step1 = make_step(10, 5);
        let step2 = make_step(15, 8);
        let result = GenerateResult::from_steps(vec![step1, step2]).unwrap();
        assert_eq!(result.total_usage.input_tokens, 25);
        assert_eq!(result.total_usage.output_tokens, 13);
        assert_eq!(result.total_usage.total_tokens, 38);
    }

    #[test]
    fn test_generate_result_uses_last_step_text() {
        let step1 = StepResult {
            text: "First".into(),
            reasoning: None,
            tool_calls: vec![],
            tool_results: vec![],
            finish_reason: FinishReason::tool_calls(),
            usage: Usage {
                input_tokens: 10,
                output_tokens: 5,
                total_tokens: 15,
                ..Default::default()
            },
            response: make_test_response("First"),
            warnings: vec![],
        };
        let step2 = StepResult {
            text: "Final answer".into(),
            reasoning: None,
            tool_calls: vec![],
            tool_results: vec![],
            finish_reason: FinishReason::stop(),
            usage: Usage {
                input_tokens: 20,
                output_tokens: 10,
                total_tokens: 30,
                ..Default::default()
            },
            response: make_test_response("Final answer"),
            warnings: vec![],
        };
        let result = GenerateResult::from_steps(vec![step1, step2]).unwrap();
        assert_eq!(result.text, "Final answer");
        assert_eq!(result.finish_reason.reason, "stop");
    }

    #[test]
    fn test_generate_result_output_none_by_default() {
        let step = make_step(10, 5);
        let result = GenerateResult::from_steps(vec![step]).unwrap();
        assert!(result.output.is_none());
    }

    #[test]
    fn test_generate_result_from_steps_empty_returns_error() {
        let result = GenerateResult::from_steps(vec![]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.kind, unified_llm_types::error::ErrorKind::Configuration);
        assert!(err.message.contains("from_steps"));
    }
}
