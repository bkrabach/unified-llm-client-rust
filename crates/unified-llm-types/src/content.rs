use serde::{Deserialize, Serialize};

/// Discriminator for content part variants.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ContentKind {
    Text,
    Image,
    Audio,
    Document,
    ToolCall,
    ToolResult,
    Thinking,
    RedactedThinking,
    /// Extensibility: unknown content kinds from providers.
    /// Preserves the original kind string.
    Other(String),
}

impl serde::Serialize for ContentKind {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            ContentKind::Text => serializer.serialize_str("text"),
            ContentKind::Image => serializer.serialize_str("image"),
            ContentKind::Audio => serializer.serialize_str("audio"),
            ContentKind::Document => serializer.serialize_str("document"),
            ContentKind::ToolCall => serializer.serialize_str("tool_call"),
            ContentKind::ToolResult => serializer.serialize_str("tool_result"),
            ContentKind::Thinking => serializer.serialize_str("thinking"),
            ContentKind::RedactedThinking => serializer.serialize_str("redacted_thinking"),
            ContentKind::Other(s) => serializer.serialize_str(s),
        }
    }
}

impl<'de> serde::Deserialize<'de> for ContentKind {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Ok(match s.as_str() {
            "text" => ContentKind::Text,
            "image" => ContentKind::Image,
            "audio" => ContentKind::Audio,
            "document" => ContentKind::Document,
            "tool_call" => ContentKind::ToolCall,
            "tool_result" => ContentKind::ToolResult,
            "thinking" => ContentKind::Thinking,
            "redacted_thinking" => ContentKind::RedactedThinking,
            _ => ContentKind::Other(s),
        })
    }
}

/// A single content part within a message. Tagged union.
///
/// Custom `Serialize`/`Deserialize` implementations handle the `"kind"` tag:
/// known variants use standard internally-tagged enum logic; the `Unknown`
/// variant preserves both the original kind string **and** all sibling fields
/// so that unrecognised content types survive a round-trip.
#[derive(Debug, Clone)]
pub enum ContentPart {
    Text {
        text: String,
    },
    Image {
        image: ImageData,
    },
    Audio {
        audio: AudioData,
    },
    Document {
        document: DocumentData,
    },
    ToolCall {
        tool_call: ToolCallData,
    },
    ToolResult {
        tool_result: ToolResultData,
    },
    Thinking {
        thinking: ThinkingData,
    },
    RedactedThinking {
        thinking: ThinkingData,
    },
    /// Extension content with preserved data. NOT deserialized via serde —
    /// constructed manually by provider adapters when they encounter unknown types.
    Extension {
        extension_kind: String,
        data: serde_json::Value,
    },
    /// Extensibility: unknown content kinds are preserved with their original
    /// kind string and all sibling fields via custom serde (F-8).
    Unknown {
        kind: String,
        data: serde_json::Value,
    },
}

// ---------------------------------------------------------------------------
// Private helper enum for derived serde on known variants only.
// ---------------------------------------------------------------------------
#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum KnownContentPart {
    Text { text: String },
    Image { image: ImageData },
    Audio { audio: AudioData },
    Document { document: DocumentData },
    ToolCall { tool_call: ToolCallData },
    ToolResult { tool_result: ToolResultData },
    Thinking { thinking: ThinkingData },
    RedactedThinking { thinking: ThinkingData },
}

/// Known kind strings for dispatch during deserialization.
const KNOWN_KINDS: &[&str] = &[
    "text",
    "image",
    "audio",
    "document",
    "tool_call",
    "tool_result",
    "thinking",
    "redacted_thinking",
];

impl Serialize for ContentPart {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            // Known variants: delegate to the helper enum.
            ContentPart::Text { text } => {
                KnownContentPart::Text { text: text.clone() }.serialize(serializer)
            }
            ContentPart::Image { image } => KnownContentPart::Image {
                image: image.clone(),
            }
            .serialize(serializer),
            ContentPart::Audio { audio } => KnownContentPart::Audio {
                audio: audio.clone(),
            }
            .serialize(serializer),
            ContentPart::Document { document } => KnownContentPart::Document {
                document: document.clone(),
            }
            .serialize(serializer),
            ContentPart::ToolCall { tool_call } => KnownContentPart::ToolCall {
                tool_call: tool_call.clone(),
            }
            .serialize(serializer),
            ContentPart::ToolResult { tool_result } => KnownContentPart::ToolResult {
                tool_result: tool_result.clone(),
            }
            .serialize(serializer),
            ContentPart::Thinking { thinking } => KnownContentPart::Thinking {
                thinking: thinking.clone(),
            }
            .serialize(serializer),
            ContentPart::RedactedThinking { thinking } => KnownContentPart::RedactedThinking {
                thinking: thinking.clone(),
            }
            .serialize(serializer),
            // Extension: merge kind as "extension" + flatten data.
            ContentPart::Extension {
                extension_kind,
                data,
            } => {
                use serde::ser::SerializeMap;
                let mut map = serializer.serialize_map(None)?;
                map.serialize_entry("kind", "extension")?;
                map.serialize_entry("extension_kind", extension_kind)?;
                map.serialize_entry("data", data)?;
                map.end()
            }
            // Unknown: merge kind + flatten remaining data fields.
            ContentPart::Unknown { kind, data } => {
                use serde::ser::SerializeMap;
                // Start with the data object's fields, then inject "kind".
                if let serde_json::Value::Object(obj) = data {
                    let mut map = serializer.serialize_map(Some(obj.len() + 1))?;
                    map.serialize_entry("kind", kind)?;
                    for (k, v) in obj {
                        map.serialize_entry(k, v)?;
                    }
                    map.end()
                } else {
                    let mut map = serializer.serialize_map(Some(2))?;
                    map.serialize_entry("kind", kind)?;
                    map.serialize_entry("data", data)?;
                    map.end()
                }
            }
        }
    }
}

impl<'de> Deserialize<'de> for ContentPart {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // Step 1: Deserialize into a generic JSON map.
        let mut map = serde_json::Map::deserialize(deserializer)?;

        // Step 2: Extract the "kind" discriminator.
        let kind_str = map
            .get("kind")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        // Step 3: Dispatch.
        if KNOWN_KINDS.contains(&kind_str.as_str()) {
            // Re-serialize the map and deserialize through KnownContentPart.
            let value = serde_json::Value::Object(map);
            let known: KnownContentPart =
                serde_json::from_value(value).map_err(serde::de::Error::custom)?;
            Ok(match known {
                KnownContentPart::Text { text } => ContentPart::Text { text },
                KnownContentPart::Image { image } => ContentPart::Image { image },
                KnownContentPart::Audio { audio } => ContentPart::Audio { audio },
                KnownContentPart::Document { document } => ContentPart::Document { document },
                KnownContentPart::ToolCall { tool_call } => ContentPart::ToolCall { tool_call },
                KnownContentPart::ToolResult { tool_result } => {
                    ContentPart::ToolResult { tool_result }
                }
                KnownContentPart::Thinking { thinking } => ContentPart::Thinking { thinking },
                KnownContentPart::RedactedThinking { thinking } => {
                    ContentPart::RedactedThinking { thinking }
                }
            })
        } else if kind_str == "extension" {
            // Extension variant: extract extension_kind and data fields.
            let extension_kind = map
                .remove("extension_kind")
                .and_then(|v| v.as_str().map(|s| s.to_string()))
                .unwrap_or_default();
            let data = map.remove("data").unwrap_or(serde_json::Value::Null);
            Ok(ContentPart::Extension {
                extension_kind,
                data,
            })
        } else {
            // Unknown kind: preserve kind string and all remaining fields.
            map.remove("kind");
            Ok(ContentPart::Unknown {
                kind: kind_str,
                data: serde_json::Value::Object(map),
            })
        }
    }
}

impl ContentPart {
    /// Convenience: create a text content part.
    pub fn text(s: impl Into<String>) -> Self {
        ContentPart::Text { text: s.into() }
    }

    /// Convenience: create an image content part from a URL.
    pub fn image_url(url: impl Into<String>) -> Self {
        ContentPart::Image {
            image: ImageData {
                url: Some(url.into()),
                data: None,
                media_type: None,
                detail: None,
            },
        }
    }

    /// Return the discriminant kind of this content part.
    pub fn kind(&self) -> ContentKind {
        match self {
            ContentPart::Text { .. } => ContentKind::Text,
            ContentPart::Image { .. } => ContentKind::Image,
            ContentPart::Audio { .. } => ContentKind::Audio,
            ContentPart::Document { .. } => ContentKind::Document,
            ContentPart::ToolCall { .. } => ContentKind::ToolCall,
            ContentPart::ToolResult { .. } => ContentKind::ToolResult,
            ContentPart::Thinking { .. } => ContentKind::Thinking,
            ContentPart::RedactedThinking { .. } => ContentKind::RedactedThinking,
            ContentPart::Extension { extension_kind, .. } => {
                ContentKind::Other(extension_kind.clone())
            }
            ContentPart::Unknown { kind, .. } => ContentKind::Other(kind.clone()),
        }
    }

    /// Convenience: create an image content part from bytes.
    pub fn image_bytes(data: Vec<u8>, media_type: impl Into<String>) -> Self {
        ContentPart::Image {
            image: ImageData {
                url: None,
                data: Some(data),
                media_type: Some(media_type.into()),
                detail: None,
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageData {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Vec<u8>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

impl ImageData {
    /// Validate that exactly one of `url` or `data` is set.
    pub fn validate(&self) -> Result<(), String> {
        match (&self.url, &self.data) {
            (Some(_), Some(_)) => Err("ImageData: cannot set both url and data".into()),
            (None, None) => Err("ImageData: must set either url or data".into()),
            _ => Ok(()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioData {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Vec<u8>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
}

impl AudioData {
    /// Validate that exactly one of `url` or `data` is set.
    pub fn validate(&self) -> Result<(), String> {
        match (&self.url, &self.data) {
            (Some(_), Some(_)) => Err("AudioData: cannot set both url and data".into()),
            (None, None) => Err("AudioData: must set either url or data".into()),
            _ => Ok(()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentData {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Vec<u8>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_name: Option<String>,
}

impl DocumentData {
    /// Validate that exactly one of `url` or `data` is set.
    pub fn validate(&self) -> Result<(), String> {
        match (&self.url, &self.data) {
            (Some(_), Some(_)) => Err("DocumentData: cannot set both url and data".into()),
            (None, None) => Err("DocumentData: must set either url or data".into()),
            _ => Ok(()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallData {
    pub id: String,
    pub name: String,
    /// Parsed JSON arguments or raw string.
    pub arguments: ArgumentValue,
    #[serde(default = "default_tool_type")]
    pub r#type: String,
}

fn default_tool_type() -> String {
    "function".to_string()
}

/// Tool call arguments: either parsed dict or raw string.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ArgumentValue {
    Dict(serde_json::Map<String, serde_json::Value>),
    Raw(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResultData {
    pub tool_call_id: String,
    pub content: serde_json::Value,
    #[serde(default)]
    pub is_error: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_data: Option<Vec<u8>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_media_type: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingData {
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
    #[serde(default)]
    pub redacted: bool,
    /// Opaque payload for redacted thinking blocks.
    /// Must be sent back verbatim to the provider for multi-turn continuations.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- ContentKind tests ---

    #[test]
    fn test_content_kind_known_variants() {
        let kind = ContentKind::Text;
        let json = serde_json::to_string(&kind).unwrap();
        assert_eq!(json, "\"text\"");
    }

    #[test]
    fn test_content_kind_all_variants_roundtrip() {
        for (kind, expected) in [
            (ContentKind::Text, "\"text\""),
            (ContentKind::Image, "\"image\""),
            (ContentKind::Audio, "\"audio\""),
            (ContentKind::Document, "\"document\""),
            (ContentKind::ToolCall, "\"tool_call\""),
            (ContentKind::ToolResult, "\"tool_result\""),
            (ContentKind::Thinking, "\"thinking\""),
            (ContentKind::RedactedThinking, "\"redacted_thinking\""),
        ] {
            let json = serde_json::to_string(&kind).unwrap();
            assert_eq!(json, expected);
            let back: ContentKind = serde_json::from_str(&json).unwrap();
            assert_eq!(back, kind);
        }
    }

    #[test]
    fn test_content_kind_unknown_deserializes_to_other() {
        let kind: ContentKind = serde_json::from_str("\"some_future_kind\"").unwrap();
        match kind {
            ContentKind::Other(s) => assert_eq!(s, "some_future_kind"),
            _ => panic!("Expected Other variant with preserved string"),
        }
    }

    #[test]
    fn test_content_kind_other_preserves_string() {
        // Known kinds still work
        let kind: ContentKind = serde_json::from_str("\"text\"").unwrap();
        assert_eq!(kind, ContentKind::Text);

        // Unknown kinds should preserve the original string
        let kind: ContentKind = serde_json::from_str("\"custom_block\"").unwrap();
        match kind {
            ContentKind::Other(s) => assert_eq!(s, "custom_block"),
            _ => panic!("Expected Other(\"custom_block\"), got {:?}", kind),
        }
    }

    #[test]
    fn test_content_kind_other_round_trips() {
        let kind = ContentKind::Other("my_extension".to_string());
        let json = serde_json::to_string(&kind).unwrap();
        assert_eq!(json, "\"my_extension\"");
        let back: ContentKind = serde_json::from_str(&json).unwrap();
        match back {
            ContentKind::Other(s) => assert_eq!(s, "my_extension"),
            _ => panic!("Round-trip failed"),
        }
    }

    // --- ContentPart tests ---

    #[test]
    fn test_content_part_text_roundtrip() {
        let part = ContentPart::text("hello");
        let json = serde_json::to_string(&part).unwrap();
        let back: ContentPart = serde_json::from_str(&json).unwrap();
        assert!(matches!(back, ContentPart::Text { text } if text == "hello"));
    }

    #[test]
    fn test_content_part_text_serde_has_kind_tag() {
        let part = ContentPart::text("hello");
        let json = serde_json::to_string(&part).unwrap();
        assert!(json.contains("\"kind\":\"text\""));
        assert!(json.contains("\"text\":\"hello\""));
    }

    #[test]
    fn test_content_part_image_url_roundtrip() {
        let part = ContentPart::image_url("https://example.com/img.png");
        let json = serde_json::to_string(&part).unwrap();
        let back: ContentPart = serde_json::from_str(&json).unwrap();
        match back {
            ContentPart::Image { image } => {
                assert_eq!(image.url, Some("https://example.com/img.png".to_string()));
            }
            _ => panic!("Expected Image variant"),
        }
    }

    #[test]
    fn test_content_part_image_bytes_roundtrip() {
        let part = ContentPart::image_bytes(vec![1, 2, 3], "image/png");
        let json = serde_json::to_string(&part).unwrap();
        let back: ContentPart = serde_json::from_str(&json).unwrap();
        match back {
            ContentPart::Image { image } => {
                assert_eq!(image.data, Some(vec![1, 2, 3]));
                assert_eq!(image.media_type, Some("image/png".to_string()));
            }
            _ => panic!("Expected Image variant"),
        }
    }

    #[test]
    fn test_content_part_unknown_deserializes_preserving_kind_and_data() {
        // F-8: Unknown kinds now preserve both the kind string and all fields.
        let json = r#"{"kind":"future_content_type","foo":"bar","count":42}"#;
        let part: ContentPart = serde_json::from_str(json).unwrap();
        match &part {
            ContentPart::Unknown { kind, data } => {
                assert_eq!(kind, "future_content_type");
                assert_eq!(data["foo"], "bar");
                assert_eq!(data["count"], 42);
            }
            _ => panic!("Expected Unknown variant, got {:?}", part),
        }
    }

    #[test]
    fn test_content_part_unknown_minimal() {
        // Minimal unknown: only kind, no extra fields
        let json = r#"{"kind":"future_content_type"}"#;
        let part: ContentPart = serde_json::from_str(json).unwrap();
        match &part {
            ContentPart::Unknown { kind, data } => {
                assert_eq!(kind, "future_content_type");
                assert!(data.as_object().unwrap().is_empty());
            }
            _ => panic!("Expected Unknown variant"),
        }
    }

    #[test]
    fn test_content_part_unknown_roundtrip() {
        // F-8: Unknown kinds survive serialize → deserialize round-trip.
        let json = r#"{"kind":"custom_block","payload":"preserved","nested":{"a":1}}"#;
        let part: ContentPart = serde_json::from_str(json).unwrap();
        let serialized = serde_json::to_string(&part).unwrap();
        let back: ContentPart = serde_json::from_str(&serialized).unwrap();
        match &back {
            ContentPart::Unknown { kind, data } => {
                assert_eq!(kind, "custom_block");
                assert_eq!(data["payload"], "preserved");
                assert_eq!(data["nested"]["a"], 1);
            }
            _ => panic!("Expected Unknown variant after round-trip"),
        }
    }

    #[test]
    fn test_content_part_extension_preserves_data() {
        // Extension variant preserves kind and data when constructed manually
        let part = ContentPart::Extension {
            extension_kind: "video".to_string(),
            data: serde_json::json!({"url": "https://example.com/video.mp4", "duration": 120}),
        };
        match &part {
            ContentPart::Extension {
                extension_kind,
                data,
            } => {
                assert_eq!(extension_kind, "video");
                assert_eq!(data["url"], "https://example.com/video.mp4");
                assert_eq!(data["duration"], 120);
            }
            _ => panic!("Expected Extension variant"),
        }
    }

    #[test]
    fn test_content_part_extension_different_from_unknown() {
        // Unknown preserves kind+data from deserialization; Extension is manually constructed
        let unknown = ContentPart::Unknown {
            kind: "custom".to_string(),
            data: serde_json::json!({"payload": "from_serde"}),
        };
        let extension = ContentPart::Extension {
            extension_kind: "custom".to_string(),
            data: serde_json::json!({"payload": "preserved"}),
        };
        assert!(matches!(unknown, ContentPart::Unknown { .. }));
        assert!(matches!(extension, ContentPart::Extension { .. }));
    }

    // --- ImageData tests ---

    #[test]
    fn test_image_data_optional_fields_omitted() {
        let img = ImageData {
            url: Some("https://example.com/img.png".to_string()),
            data: None,
            media_type: None,
            detail: None,
        };
        let json = serde_json::to_string(&img).unwrap();
        assert!(!json.contains("data"));
        assert!(!json.contains("media_type"));
        assert!(!json.contains("detail"));
    }

    #[test]
    fn test_image_data_roundtrip() {
        let img = ImageData {
            url: Some("https://example.com/img.png".to_string()),
            data: Some(vec![0xFF, 0xD8]),
            media_type: Some("image/jpeg".to_string()),
            detail: Some("high".to_string()),
        };
        let json = serde_json::to_string(&img).unwrap();
        let back: ImageData = serde_json::from_str(&json).unwrap();
        assert_eq!(back.url, img.url);
        assert_eq!(back.data, img.data);
        assert_eq!(back.media_type, img.media_type);
        assert_eq!(back.detail, img.detail);
    }

    // --- AudioData tests ---

    #[test]
    fn test_audio_data_roundtrip() {
        let audio = AudioData {
            url: Some("https://example.com/audio.mp3".to_string()),
            data: None,
            media_type: Some("audio/mpeg".to_string()),
        };
        let json = serde_json::to_string(&audio).unwrap();
        let back: AudioData = serde_json::from_str(&json).unwrap();
        assert_eq!(back.url, audio.url);
        assert_eq!(back.media_type, audio.media_type);
    }

    // --- DocumentData tests ---

    #[test]
    fn test_document_data_roundtrip() {
        let doc = DocumentData {
            url: None,
            data: Some(vec![0x25, 0x50, 0x44, 0x46]),
            media_type: Some("application/pdf".to_string()),
            file_name: Some("report.pdf".to_string()),
        };
        let json = serde_json::to_string(&doc).unwrap();
        let back: DocumentData = serde_json::from_str(&json).unwrap();
        assert_eq!(back.data, doc.data);
        assert_eq!(back.file_name, Some("report.pdf".to_string()));
    }

    // --- ToolCallData tests ---

    #[test]
    fn test_tool_call_data_with_dict_arguments() {
        let tc = ToolCallData {
            id: "call_1".to_string(),
            name: "get_weather".to_string(),
            arguments: ArgumentValue::Dict(serde_json::Map::new()),
            r#type: "function".to_string(),
        };
        let json = serde_json::to_string(&tc).unwrap();
        let back: ToolCallData = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, "call_1");
        assert_eq!(back.name, "get_weather");
        assert!(matches!(back.arguments, ArgumentValue::Dict(_)));
    }

    #[test]
    fn test_tool_call_data_default_type() {
        let json = r#"{"id":"call_1","name":"fn","arguments":{}}"#;
        let tc: ToolCallData = serde_json::from_str(json).unwrap();
        assert_eq!(tc.r#type, "function");
    }

    // --- ArgumentValue tests ---

    #[test]
    fn test_argument_value_dict() {
        let val = ArgumentValue::Dict(serde_json::Map::new());
        let json = serde_json::to_string(&val).unwrap();
        assert_eq!(json, "{}");
    }

    #[test]
    fn test_argument_value_raw_string() {
        let val = ArgumentValue::Raw("raw args".to_string());
        let json = serde_json::to_string(&val).unwrap();
        assert_eq!(json, "\"raw args\"");
    }

    #[test]
    fn test_argument_value_dict_with_data() {
        let mut map = serde_json::Map::new();
        map.insert(
            "city".to_string(),
            serde_json::Value::String("SF".to_string()),
        );
        let val = ArgumentValue::Dict(map);
        let json = serde_json::to_string(&val).unwrap();
        let back: ArgumentValue = serde_json::from_str(&json).unwrap();
        match back {
            ArgumentValue::Dict(m) => assert_eq!(m["city"], "SF"),
            _ => panic!("Expected Dict"),
        }
    }

    // --- ToolResultData tests ---

    #[test]
    fn test_tool_result_data_roundtrip() {
        let tr = ToolResultData {
            tool_call_id: "call_1".to_string(),
            content: serde_json::Value::String("sunny, 72F".to_string()),
            is_error: false,
            image_data: None,
            image_media_type: None,
        };
        let json = serde_json::to_string(&tr).unwrap();
        let back: ToolResultData = serde_json::from_str(&json).unwrap();
        assert_eq!(back.tool_call_id, "call_1");
        assert!(!back.is_error);
    }

    #[test]
    fn test_tool_result_data_error_flag() {
        let tr = ToolResultData {
            tool_call_id: "call_2".to_string(),
            content: serde_json::Value::String("network error".to_string()),
            is_error: true,
            image_data: None,
            image_media_type: None,
        };
        let json = serde_json::to_string(&tr).unwrap();
        let back: ToolResultData = serde_json::from_str(&json).unwrap();
        assert!(back.is_error);
    }

    // --- ThinkingData tests ---

    #[test]
    fn test_thinking_data_preserves_signature() {
        let thinking = ThinkingData {
            text: "Let me think...".to_string(),
            signature: Some("sig_abc123".to_string()),
            redacted: false,
            data: None,
        };
        let json = serde_json::to_string(&thinking).unwrap();
        let back: ThinkingData = serde_json::from_str(&json).unwrap();
        assert_eq!(back.signature, Some("sig_abc123".to_string()));
        assert_eq!(back.text, "Let me think...");
        assert!(!back.redacted);
    }

    #[test]
    fn test_thinking_data_redacted() {
        let thinking = ThinkingData {
            text: String::new(),
            signature: Some("sig_opaque".to_string()),
            redacted: true,
            data: None,
        };
        let json = serde_json::to_string(&thinking).unwrap();
        let back: ThinkingData = serde_json::from_str(&json).unwrap();
        assert!(back.redacted);
        assert_eq!(back.signature, Some("sig_opaque".to_string()));
    }

    #[test]
    fn test_thinking_data_no_signature() {
        let thinking = ThinkingData {
            text: "reasoning".to_string(),
            signature: None,
            redacted: false,
            data: None,
        };
        let json = serde_json::to_string(&thinking).unwrap();
        assert!(!json.contains("signature"));
        let back: ThinkingData = serde_json::from_str(&json).unwrap();
        assert_eq!(back.signature, None);
    }

    // --- ContentPart with ToolCall/ToolResult/Thinking ---

    #[test]
    fn test_content_part_tool_call_roundtrip() {
        let part = ContentPart::ToolCall {
            tool_call: ToolCallData {
                id: "call_1".to_string(),
                name: "search".to_string(),
                arguments: ArgumentValue::Dict(serde_json::Map::new()),
                r#type: "function".to_string(),
            },
        };
        let json = serde_json::to_string(&part).unwrap();
        assert!(json.contains("\"kind\":\"tool_call\""));
        let back: ContentPart = serde_json::from_str(&json).unwrap();
        match back {
            ContentPart::ToolCall { tool_call } => {
                assert_eq!(tool_call.name, "search");
            }
            _ => panic!("Expected ToolCall variant"),
        }
    }

    #[test]
    fn test_content_part_thinking_roundtrip() {
        let part = ContentPart::Thinking {
            thinking: ThinkingData {
                text: "Let me reason...".to_string(),
                signature: Some("sig_123".to_string()),
                redacted: false,
                data: None,
            },
        };
        let json = serde_json::to_string(&part).unwrap();
        assert!(json.contains("\"kind\":\"thinking\""));
        let back: ContentPart = serde_json::from_str(&json).unwrap();
        match back {
            ContentPart::Thinking { thinking } => {
                assert_eq!(thinking.text, "Let me reason...");
                assert_eq!(thinking.signature, Some("sig_123".to_string()));
            }
            _ => panic!("Expected Thinking variant"),
        }
    }

    #[test]
    fn test_content_part_redacted_thinking_roundtrip() {
        let part = ContentPart::RedactedThinking {
            thinking: ThinkingData {
                text: String::new(),
                signature: Some("sig_opaque".to_string()),
                redacted: true,
                data: None,
            },
        };
        let json = serde_json::to_string(&part).unwrap();
        assert!(json.contains("\"kind\":\"redacted_thinking\""));
        let back: ContentPart = serde_json::from_str(&json).unwrap();
        match back {
            ContentPart::RedactedThinking { thinking } => {
                assert!(thinking.redacted);
            }
            _ => panic!("Expected RedactedThinking variant"),
        }
    }

    #[test]
    fn test_thinking_data_with_data_field_roundtrip() {
        let thinking = ThinkingData {
            text: String::new(),
            signature: None,
            redacted: true,
            data: Some("opaque_base64_blob_abc123".to_string()),
        };
        let json = serde_json::to_string(&thinking).unwrap();
        assert!(json.contains("opaque_base64_blob_abc123"));
        let back: ThinkingData = serde_json::from_str(&json).unwrap();
        assert_eq!(back.data, Some("opaque_base64_blob_abc123".to_string()));
        assert!(back.redacted);
    }

    // --- FP-09: ImageData validation ---

    #[test]
    fn test_image_data_validate_url_only() {
        let img = ImageData {
            url: Some("https://example.com/img.png".into()),
            data: None,
            media_type: None,
            detail: None,
        };
        assert!(img.validate().is_ok());
    }

    #[test]
    fn test_image_data_validate_data_only() {
        let img = ImageData {
            url: None,
            data: Some(vec![1, 2, 3]),
            media_type: Some("image/png".into()),
            detail: None,
        };
        assert!(img.validate().is_ok());
    }

    #[test]
    fn test_image_data_validate_both_set_error() {
        let img = ImageData {
            url: Some("url".into()),
            data: Some(vec![1]),
            media_type: None,
            detail: None,
        };
        assert!(img.validate().is_err());
    }

    #[test]
    fn test_image_data_validate_neither_set_error() {
        let img = ImageData {
            url: None,
            data: None,
            media_type: None,
            detail: None,
        };
        assert!(img.validate().is_err());
    }

    // --- FP-09: AudioData validation ---

    #[test]
    fn test_audio_data_validate_url_only() {
        let audio = AudioData {
            url: Some("https://example.com/audio.mp3".into()),
            data: None,
            media_type: None,
        };
        assert!(audio.validate().is_ok());
    }

    #[test]
    fn test_audio_data_validate_data_only() {
        let audio = AudioData {
            url: None,
            data: Some(vec![1, 2, 3]),
            media_type: Some("audio/mp3".into()),
        };
        assert!(audio.validate().is_ok());
    }

    #[test]
    fn test_audio_data_validate_both_set_error() {
        let audio = AudioData {
            url: Some("url".into()),
            data: Some(vec![1]),
            media_type: None,
        };
        assert!(audio.validate().is_err());
    }

    #[test]
    fn test_audio_data_validate_neither_set_error() {
        let audio = AudioData {
            url: None,
            data: None,
            media_type: None,
        };
        assert!(audio.validate().is_err());
    }

    // --- FP-09: DocumentData validation ---

    #[test]
    fn test_document_data_validate_url_only() {
        let doc = DocumentData {
            url: Some("https://example.com/doc.pdf".into()),
            data: None,
            media_type: None,
            file_name: None,
        };
        assert!(doc.validate().is_ok());
    }

    #[test]
    fn test_document_data_validate_data_only() {
        let doc = DocumentData {
            url: None,
            data: Some(vec![1, 2, 3]),
            media_type: Some("application/pdf".into()),
            file_name: None,
        };
        assert!(doc.validate().is_ok());
    }

    #[test]
    fn test_document_data_validate_both_set_error() {
        let doc = DocumentData {
            url: Some("url".into()),
            data: Some(vec![1]),
            media_type: None,
            file_name: None,
        };
        assert!(doc.validate().is_err());
    }

    #[test]
    fn test_document_data_validate_neither_set_error() {
        let doc = DocumentData {
            url: None,
            data: None,
            media_type: None,
            file_name: None,
        };
        assert!(doc.validate().is_err());
    }

    // --- FP-03: ContentPart::kind() ---

    #[test]
    fn test_content_part_kind_returns_correct_variant() {
        assert_eq!(ContentPart::text("hello").kind(), ContentKind::Text);
        assert_eq!(ContentPart::image_url("url").kind(), ContentKind::Image);
        assert_eq!(
            ContentPart::Unknown {
                kind: "custom_block".into(),
                data: serde_json::json!({}),
            }
            .kind(),
            ContentKind::Other("custom_block".into())
        );

        let ext = ContentPart::Extension {
            extension_kind: "video".into(),
            data: serde_json::json!({}),
        };
        assert_eq!(ext.kind(), ContentKind::Other("video".into()));
    }

    #[test]
    fn test_thinking_data_data_field_none_omitted() {
        let thinking = ThinkingData {
            text: "reasoning".to_string(),
            signature: Some("sig_abc".to_string()),
            redacted: false,
            data: None,
        };
        let json = serde_json::to_string(&thinking).unwrap();
        assert!(
            !json.contains("\"data\""),
            "data: None should be omitted from serialization"
        );
    }
}
