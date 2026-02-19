// SSE (Server-Sent Events) parser — state machine for parsing SSE byte streams.

/// A parsed SSE event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseEvent {
    /// The event type (from `event:` field).
    pub event_type: Option<String>,
    /// The data payload (from `data:` field(s), joined with newlines).
    pub data: String,
    /// The event ID (from `id:` field).
    pub id: Option<String>,
    /// The retry interval in milliseconds (from `retry:` field).
    pub retry: Option<u64>,
}

/// Incremental SSE parser that handles partial chunks.
///
/// Feed chunks of text via `feed()` and receive complete events.
/// Handles: `event:`, `data:`, `id:`, `retry:` fields, comment lines
/// (`:` prefix), multi-line data, and blank line boundaries.
pub struct SseParser {
    /// Buffer for incomplete lines spanning chunk boundaries.
    buffer: String,
    /// Current event type being accumulated.
    event_type: Option<String>,
    /// Current data lines being accumulated.
    data_lines: Vec<String>,
    /// Current event ID being accumulated.
    id: Option<String>,
    /// Current retry value being accumulated.
    retry: Option<u64>,
    /// Whether we've seen any field for the current event.
    has_fields: bool,
}

impl Default for SseParser {
    fn default() -> Self {
        Self::new()
    }
}

impl SseParser {
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            event_type: None,
            data_lines: Vec::new(),
            id: None,
            retry: None,
            has_fields: false,
        }
    }

    /// Feed a chunk of text. Returns any complete events parsed from the chunk.
    pub fn feed(&mut self, chunk: &str) -> Vec<SseEvent> {
        self.buffer.push_str(chunk);
        let mut events = Vec::new();

        // Process complete lines from the buffer
        loop {
            // Find the next line ending (\n or \r\n)
            let newline_pos = self.buffer.find('\n');
            match newline_pos {
                Some(pos) => {
                    // Extract the line (strip \r if present)
                    let line_end = if pos > 0 && self.buffer.as_bytes()[pos - 1] == b'\r' {
                        pos - 1
                    } else {
                        pos
                    };
                    let line = self.buffer[..line_end].to_string();
                    self.buffer = self.buffer[pos + 1..].to_string();

                    if line.is_empty() {
                        // Blank line = event boundary
                        if let Some(event) = self.emit_event() {
                            events.push(event);
                        }
                    } else {
                        self.process_line(&line);
                    }
                }
                None => break, // No complete line yet, wait for more data
            }
        }

        events
    }

    fn process_line(&mut self, line: &str) {
        // Comment lines start with ':'
        if line.starts_with(':') {
            return;
        }

        // Split on first ':'
        if let Some(colon_pos) = line.find(':') {
            let field = &line[..colon_pos];
            // Value starts after colon, optionally after a single space
            let value = &line[colon_pos + 1..];
            let value = value.strip_prefix(' ').unwrap_or(value);

            match field {
                "event" => {
                    self.event_type = Some(value.to_string());
                    self.has_fields = true;
                }
                "data" => {
                    self.data_lines.push(value.to_string());
                    self.has_fields = true;
                }
                "id" => {
                    self.id = Some(value.to_string());
                    self.has_fields = true;
                }
                "retry" => {
                    self.retry = value.parse::<u64>().ok();
                    self.has_fields = true;
                }
                _ => {
                    // Unknown field, ignore per SSE spec
                }
            }
        }
        // Lines without ':' are ignored per SSE spec
    }

    fn emit_event(&mut self) -> Option<SseEvent> {
        if !self.has_fields {
            return None;
        }

        let event = SseEvent {
            event_type: self.event_type.take(),
            data: self.data_lines.join("\n"),
            id: self.id.take(),
            retry: self.retry.take(),
        };

        self.data_lines.clear();
        self.has_fields = false;

        Some(event)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_event_data() {
        let mut parser = SseParser::new();
        let events = parser.feed("event: message\ndata: hello\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, Some("message".into()));
        assert_eq!(events[0].data, "hello");
    }

    #[test]
    fn test_data_only_event() {
        let mut parser = SseParser::new();
        let events = parser.feed("data: hello\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, None);
        assert_eq!(events[0].data, "hello");
    }

    #[test]
    fn test_multiline_data() {
        let mut parser = SseParser::new();
        let events = parser.feed("data: line1\ndata: line2\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "line1\nline2");
    }

    #[test]
    fn test_multiple_events() {
        let mut parser = SseParser::new();
        let events = parser.feed("data: first\n\ndata: second\n\n");
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].data, "first");
        assert_eq!(events[1].data, "second");
    }

    #[test]
    fn test_comment_lines_ignored() {
        let mut parser = SseParser::new();
        let events = parser.feed(": this is a comment\ndata: real\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "real");
    }

    #[test]
    fn test_empty_data_lines() {
        let mut parser = SseParser::new();
        let events = parser.feed("data:\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "");
    }

    #[test]
    fn test_retry_field() {
        let mut parser = SseParser::new();
        let events = parser.feed("retry: 3000\ndata: hello\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].retry, Some(3000));
        assert_eq!(events[0].data, "hello");
    }

    #[test]
    fn test_id_field() {
        let mut parser = SseParser::new();
        let events = parser.feed("id: evt_123\ndata: hello\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].id, Some("evt_123".into()));
        assert_eq!(events[0].data, "hello");
    }

    #[test]
    fn test_event_type_field() {
        let mut parser = SseParser::new();
        let events = parser.feed("event: content_block_delta\ndata: {\"text\":\"hi\"}\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, Some("content_block_delta".into()));
        assert_eq!(events[0].data, "{\"text\":\"hi\"}");
    }

    #[test]
    fn test_done_sentinel() {
        let mut parser = SseParser::new();
        let events = parser.feed("data: [DONE]\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "[DONE]");
    }

    #[test]
    fn test_partial_chunks_accumulated() {
        let mut parser = SseParser::new();
        let events1 = parser.feed("data: hel");
        assert!(events1.is_empty()); // No complete event yet
        let events2 = parser.feed("lo\n\n");
        assert_eq!(events2.len(), 1);
        assert_eq!(events2[0].data, "hello");
    }

    #[test]
    fn test_partial_chunks_across_field_boundary() {
        let mut parser = SseParser::new();
        assert!(parser.feed("event: mess").is_empty());
        assert!(parser.feed("age\n").is_empty());
        assert!(parser.feed("data: hi\n").is_empty());
        let events = parser.feed("\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, Some("message".into()));
        assert_eq!(events[0].data, "hi");
    }

    #[test]
    fn test_data_with_space_after_colon() {
        let mut parser = SseParser::new();
        let events = parser.feed("data: hello\n\n");
        assert_eq!(events[0].data, "hello");
    }

    #[test]
    fn test_data_without_space_after_colon() {
        let mut parser = SseParser::new();
        let events = parser.feed("data:hello\n\n");
        assert_eq!(events[0].data, "hello");
    }

    #[test]
    fn test_crlf_line_endings() {
        let mut parser = SseParser::new();
        let events = parser.feed("data: hello\r\n\r\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "hello");
    }

    #[test]
    fn test_empty_event_not_emitted() {
        let mut parser = SseParser::new();
        // Two consecutive blank lines with no data between them
        let events = parser.feed("\n\n");
        assert!(events.is_empty());
    }

    #[test]
    fn test_retry_non_numeric_ignored() {
        let mut parser = SseParser::new();
        let events = parser.feed("retry: abc\ndata: hello\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].retry, None);
    }
}
