// stream_lifecycle.rs — Stream event ordering validation (M-5).
//
// Provides warnings when stream events arrive in unexpected order,
// enforcing the spec §3.13 contract without hard failures.

use unified_llm_types::StreamEventType;

/// Tracks stream event ordering and warns on violations.
///
/// The spec (§3.13) defines an expected ordering:
/// - `STREAM_START` must come first
/// - `TEXT_START` before `TEXT_DELTA`, `TEXT_END` after deltas
/// - Same for `REASONING_*` and `TOOL_CALL_*` triples
/// - `FINISH` must come last
///
/// This validator logs warnings via `tracing::warn!` when events arrive
/// out of expected order, helping catch adapter bugs without crashing.
#[derive(Debug)]
pub struct StreamLifecycle {
    started: bool,
    finished: bool,
    in_text: bool,
    in_reasoning: bool,
    in_tool_call: bool,
}

impl Default for StreamLifecycle {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamLifecycle {
    pub fn new() -> Self {
        Self {
            started: false,
            finished: false,
            in_text: false,
            in_reasoning: false,
            in_tool_call: false,
        }
    }

    /// Validate and record an event transition. Logs warnings on ordering violations.
    pub fn observe(&mut self, event_type: &StreamEventType) {
        if self.finished && !matches!(event_type, StreamEventType::ProviderEvent) {
            tracing::warn!(
                event = ?event_type,
                "Stream event received after FINISH — violates spec §3.13 ordering"
            );
        }

        match event_type {
            StreamEventType::StreamStart => {
                if self.started {
                    tracing::warn!("Duplicate STREAM_START event");
                }
                self.started = true;
            }
            StreamEventType::TextStart => {
                if !self.started {
                    tracing::warn!("TEXT_START before STREAM_START");
                }
                if self.in_text {
                    tracing::warn!("TEXT_START while already in text block");
                }
                self.in_text = true;
            }
            StreamEventType::TextDelta => {
                if !self.in_text {
                    tracing::warn!("TEXT_DELTA without preceding TEXT_START");
                }
            }
            StreamEventType::TextEnd => {
                if !self.in_text {
                    tracing::warn!("TEXT_END without preceding TEXT_START");
                }
                self.in_text = false;
            }
            StreamEventType::ReasoningStart => {
                if !self.started {
                    tracing::warn!("REASONING_START before STREAM_START");
                }
                if self.in_reasoning {
                    tracing::warn!("REASONING_START while already in reasoning block");
                }
                self.in_reasoning = true;
            }
            StreamEventType::ReasoningDelta => {
                if !self.in_reasoning {
                    tracing::warn!("REASONING_DELTA without preceding REASONING_START");
                }
            }
            StreamEventType::ReasoningEnd => {
                if !self.in_reasoning {
                    tracing::warn!("REASONING_END without preceding REASONING_START");
                }
                self.in_reasoning = false;
            }
            StreamEventType::ToolCallStart => {
                if !self.started {
                    tracing::warn!("TOOL_CALL_START before STREAM_START");
                }
                self.in_tool_call = true;
            }
            StreamEventType::ToolCallDelta => {
                if !self.in_tool_call {
                    tracing::warn!("TOOL_CALL_DELTA without preceding TOOL_CALL_START");
                }
            }
            StreamEventType::ToolCallEnd => {
                if !self.in_tool_call {
                    tracing::warn!("TOOL_CALL_END without preceding TOOL_CALL_START");
                }
                self.in_tool_call = false;
            }
            StreamEventType::Finish => {
                if !self.started {
                    tracing::warn!("FINISH before STREAM_START");
                }
                self.finished = true;
            }
            StreamEventType::StepFinish => {
                // Extension event — allowed between rounds
            }
            StreamEventType::Error | StreamEventType::ProviderEvent => {
                // Can happen in any state
            }
            StreamEventType::Unknown(_) => {
                // Unknown events are always allowed
            }
        }
    }

    /// Whether a FINISH event has been observed.
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Reset for a new tool-loop step.
    pub fn reset(&mut self) {
        self.finished = false;
        self.in_text = false;
        self.in_reasoning = false;
        self.in_tool_call = false;
        // `started` stays true — the stream is still active
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_text_lifecycle() {
        let mut lc = StreamLifecycle::new();
        lc.observe(&StreamEventType::StreamStart);
        lc.observe(&StreamEventType::TextStart);
        lc.observe(&StreamEventType::TextDelta);
        lc.observe(&StreamEventType::TextDelta);
        lc.observe(&StreamEventType::TextEnd);
        lc.observe(&StreamEventType::Finish);
        assert!(lc.is_finished());
    }

    #[test]
    fn test_valid_tool_call_lifecycle() {
        let mut lc = StreamLifecycle::new();
        lc.observe(&StreamEventType::StreamStart);
        lc.observe(&StreamEventType::ToolCallStart);
        lc.observe(&StreamEventType::ToolCallDelta);
        lc.observe(&StreamEventType::ToolCallEnd);
        lc.observe(&StreamEventType::Finish);
        assert!(lc.is_finished());
    }

    #[test]
    fn test_valid_reasoning_lifecycle() {
        let mut lc = StreamLifecycle::new();
        lc.observe(&StreamEventType::StreamStart);
        lc.observe(&StreamEventType::ReasoningStart);
        lc.observe(&StreamEventType::ReasoningDelta);
        lc.observe(&StreamEventType::ReasoningEnd);
        lc.observe(&StreamEventType::TextStart);
        lc.observe(&StreamEventType::TextDelta);
        lc.observe(&StreamEventType::TextEnd);
        lc.observe(&StreamEventType::Finish);
        assert!(lc.is_finished());
    }

    #[test]
    fn test_reset_for_tool_loop() {
        let mut lc = StreamLifecycle::new();
        lc.observe(&StreamEventType::StreamStart);
        lc.observe(&StreamEventType::Finish);
        assert!(lc.is_finished());

        lc.reset();
        assert!(!lc.is_finished());
        // started should still be true (stream is still active)
        // New step can emit text events without another StreamStart
        lc.observe(&StreamEventType::TextStart);
        lc.observe(&StreamEventType::TextDelta);
        lc.observe(&StreamEventType::TextEnd);
        lc.observe(&StreamEventType::Finish);
        assert!(lc.is_finished());
    }

    #[test]
    fn test_unknown_events_allowed() {
        let mut lc = StreamLifecycle::new();
        lc.observe(&StreamEventType::StreamStart);
        lc.observe(&StreamEventType::Unknown("CUSTOM_EVENT".into()));
        lc.observe(&StreamEventType::Finish);
        assert!(lc.is_finished());
    }
}
