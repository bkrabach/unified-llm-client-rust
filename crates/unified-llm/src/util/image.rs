// Image encoding utilities for provider adapters.

use std::path::Path;

use base64::Engine;

/// Check if a string looks like a local file path (starts with `/`, `./`, `../`, or `~`).
pub fn is_local_path(s: &str) -> bool {
    s.starts_with('/') || s.starts_with("./") || s.starts_with("../") || s.starts_with('~')
}

/// Infer a MIME type from a file path extension. Defaults to `"image/png"` if unknown.
pub fn infer_mime_type(path: &str) -> &'static str {
    mime_guess::from_path(path)
        .first()
        .map(|m| match m.as_ref() {
            "image/png" => "image/png",
            "image/jpeg" => "image/jpeg",
            "image/gif" => "image/gif",
            "image/webp" => "image/webp",
            _ => "image/png",
        })
        .unwrap_or("image/png")
}

/// Build a data URI from a MIME type and already-base64-encoded data.
/// Returns `data:{mime};base64,{data}`.
pub fn encode_data_uri(mime_type: &str, base64_data: &str) -> String {
    format!("data:{mime_type};base64,{base64_data}")
}

/// Encode raw bytes to a base64 string using the STANDARD alphabet.
pub fn base64_encode(data: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(data)
}

/// Resolve a local file path to its raw bytes and inferred MIME type (sync version).
///
/// Expands `~/` to the user's home directory via `$HOME`.
/// Returns `(file_bytes, mime_type)` on success.
///
/// NOTE: This uses blocking `std::fs::read`. Prefer calling
/// [`pre_resolve_local_images`] from async contexts so that this
/// sync fallback path is never reached in production.
pub fn resolve_local_file(url: &str) -> Result<(Vec<u8>, String), unified_llm_types::Error> {
    let path = expand_tilde(url);

    let data = std::fs::read(&path).map_err(|e| unified_llm_types::Error {
        kind: unified_llm_types::ErrorKind::InvalidRequest,
        message: format!("Failed to read image file '{}': {}", path.display(), e),
        retryable: false,
        source: None,
        provider: None,
        status_code: None,
        error_code: None,
        retry_after: None,
        raw: None,
    })?;

    let mime = infer_mime_type(&path.to_string_lossy());

    Ok((data, mime.to_string()))
}

/// H-4: Non-blocking version of [`resolve_local_file`] for use in async contexts.
///
/// Uses `tokio::fs::read` to avoid blocking the Tokio runtime.
pub async fn resolve_local_file_async(
    url: &str,
) -> Result<(Vec<u8>, String), unified_llm_types::Error> {
    let path = expand_tilde(url);

    let data = tokio::fs::read(&path)
        .await
        .map_err(|e| unified_llm_types::Error {
            kind: unified_llm_types::ErrorKind::InvalidRequest,
            message: format!("Failed to read image file '{}': {}", path.display(), e),
            retryable: false,
            source: None,
            provider: None,
            status_code: None,
            error_code: None,
            retry_after: None,
            raw: None,
        })?;

    let mime = infer_mime_type(&path.to_string_lossy());

    Ok((data, mime.to_string()))
}

/// Expand `~/` prefix to the user's home directory.
fn expand_tilde(url: &str) -> std::path::PathBuf {
    if let Some(rest) = url.strip_prefix("~/") {
        let home = std::env::var("HOME").unwrap_or_default();
        Path::new(&home).join(rest)
    } else {
        Path::new(url).to_path_buf()
    }
}

/// H-4: Pre-resolve all local file image references in messages to avoid
/// blocking I/O during synchronous request translation.
///
/// Call this from async contexts (adapter `do_complete` / `do_stream`)
/// **before** passing the request to the sync `translate_request` functions.
/// After pre-resolution, `image.data` is populated with file bytes so the
/// sync translate path never hits `resolve_local_file`.
pub async fn pre_resolve_local_images(
    messages: &mut [unified_llm_types::Message],
) -> Result<(), unified_llm_types::Error> {
    for msg in messages.iter_mut() {
        for part in msg.content.iter_mut() {
            if let unified_llm_types::ContentPart::Image { image } = part {
                if image.data.is_none() {
                    if let Some(ref url) = image.url {
                        if is_local_path(url) {
                            let (data, mime) = resolve_local_file_async(url).await?;
                            image.data = Some(data);
                            image.media_type = Some(mime);
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_local_path_absolute() {
        assert!(is_local_path("/home/user/image.png"));
    }

    #[test]
    fn test_is_local_path_relative() {
        assert!(is_local_path("./screenshots/test.png"));
    }

    #[test]
    fn test_is_local_path_relative_parent() {
        assert!(is_local_path("../screenshots/test.png"));
    }

    #[test]
    fn test_is_local_path_tilde() {
        assert!(is_local_path("~/photos/cat.jpg"));
    }

    #[test]
    fn test_is_local_path_url() {
        assert!(!is_local_path("https://example.com/image.png"));
    }

    #[test]
    fn test_is_local_path_data_uri() {
        assert!(!is_local_path("data:image/png;base64,abc123"));
    }

    #[test]
    fn test_infer_mime_type_png() {
        assert_eq!(infer_mime_type("image.png"), "image/png");
    }

    #[test]
    fn test_infer_mime_type_jpeg() {
        assert_eq!(infer_mime_type("photo.jpg"), "image/jpeg");
    }

    #[test]
    fn test_infer_mime_type_gif() {
        assert_eq!(infer_mime_type("anim.gif"), "image/gif");
    }

    #[test]
    fn test_infer_mime_type_webp() {
        assert_eq!(infer_mime_type("photo.webp"), "image/webp");
    }

    #[test]
    fn test_infer_mime_type_unknown_defaults_to_png() {
        assert_eq!(infer_mime_type("file.xyz"), "image/png");
    }

    #[test]
    fn test_encode_data_uri() {
        let data = b"fake png data";
        let encoded = base64::engine::general_purpose::STANDARD.encode(data);
        let uri = encode_data_uri("image/png", &encoded);
        assert_eq!(uri, format!("data:image/png;base64,{}", encoded));
    }

    #[test]
    fn test_base64_encode_bytes() {
        let data = vec![0x89, 0x50, 0x4E, 0x47]; // PNG magic bytes
        let result = base64_encode(&data);
        assert_eq!(
            result,
            base64::engine::general_purpose::STANDARD.encode(&data)
        );
    }

    #[test]
    fn test_resolve_local_file_reads_and_infers_mime() {
        // Create a temp file with a .png extension
        let dir = std::env::temp_dir().join("unified_llm_test_image");
        std::fs::create_dir_all(&dir).unwrap();
        let file_path = dir.join("test_image.png");
        let fake_png = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        std::fs::write(&file_path, &fake_png).unwrap();

        let (data, mime) = resolve_local_file(file_path.to_str().unwrap()).unwrap();
        assert_eq!(data, fake_png);
        assert_eq!(mime, "image/png");

        // Cleanup
        let _ = std::fs::remove_file(&file_path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn test_resolve_local_file_jpeg_extension() {
        let dir = std::env::temp_dir().join("unified_llm_test_image_jpg");
        std::fs::create_dir_all(&dir).unwrap();
        let file_path = dir.join("photo.jpg");
        let fake_jpeg = vec![0xFF, 0xD8, 0xFF, 0xE0];
        std::fs::write(&file_path, &fake_jpeg).unwrap();

        let (data, mime) = resolve_local_file(file_path.to_str().unwrap()).unwrap();
        assert_eq!(data, fake_jpeg);
        assert_eq!(mime, "image/jpeg");

        let _ = std::fs::remove_file(&file_path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn test_resolve_local_file_missing_file_returns_error() {
        let result = resolve_local_file("/nonexistent/path/image.png");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.message.contains("Failed to read image file"));
    }

    #[test]
    fn test_resolve_local_file_tilde_expansion() {
        // We can't easily test tilde expansion without knowing HOME,
        // but we can verify it doesn't panic on a tilde path that doesn't exist.
        let result = resolve_local_file("~/nonexistent_test_dir_12345/image.png");
        assert!(result.is_err()); // File doesn't exist, but path was expanded
    }

    // --- H-4: Async resolve tests ---

    #[tokio::test]
    async fn test_resolve_local_file_async_reads_file() {
        let dir = std::env::temp_dir().join("unified_llm_test_image_async");
        std::fs::create_dir_all(&dir).unwrap();
        let file_path = dir.join("test_async.png");
        let fake_png = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        std::fs::write(&file_path, &fake_png).unwrap();

        let (data, mime) = resolve_local_file_async(file_path.to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(data, fake_png);
        assert_eq!(mime, "image/png");

        let _ = std::fs::remove_file(&file_path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[tokio::test]
    async fn test_resolve_local_file_async_missing_file_returns_error() {
        let result = resolve_local_file_async("/nonexistent/path/image.png").await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.message.contains("Failed to read image file"));
    }

    #[tokio::test]
    async fn test_pre_resolve_local_images_populates_data() {
        use unified_llm_types::{ContentPart, ImageData, Message, Role};

        let dir = std::env::temp_dir().join("unified_llm_test_pre_resolve");
        std::fs::create_dir_all(&dir).unwrap();
        let file_path = dir.join("pre_resolve_test.png");
        let fake_png = vec![0x89, 0x50, 0x4E, 0x47];
        std::fs::write(&file_path, &fake_png).unwrap();

        let mut messages = vec![Message {
            role: Role::User,
            content: vec![
                ContentPart::text("describe this"),
                ContentPart::Image {
                    image: ImageData {
                        url: Some(file_path.to_str().unwrap().to_string()),
                        data: None,
                        media_type: None,
                        detail: None,
                    },
                },
            ],
            name: None,
            tool_call_id: None,
        }];

        pre_resolve_local_images(&mut messages).await.unwrap();

        // After pre-resolution, the image data should be populated
        if let ContentPart::Image { image } = &messages[0].content[1] {
            assert!(image.data.is_some(), "image.data should be populated");
            assert_eq!(image.data.as_ref().unwrap(), &fake_png);
            assert_eq!(image.media_type.as_deref(), Some("image/png"));
        } else {
            panic!("Expected Image content part");
        }

        let _ = std::fs::remove_file(&file_path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[tokio::test]
    async fn test_pre_resolve_skips_non_local_urls() {
        use unified_llm_types::{ContentPart, ImageData, Message, Role};

        let mut messages = vec![Message {
            role: Role::User,
            content: vec![ContentPart::Image {
                image: ImageData {
                    url: Some("https://example.com/cat.jpg".to_string()),
                    data: None,
                    media_type: None,
                    detail: None,
                },
            }],
            name: None,
            tool_call_id: None,
        }];

        pre_resolve_local_images(&mut messages).await.unwrap();

        // Non-local URL should not be resolved
        if let ContentPart::Image { image } = &messages[0].content[0] {
            assert!(
                image.data.is_none(),
                "remote URL should not be pre-resolved"
            );
        } else {
            panic!("Expected Image content part");
        }
    }

    #[tokio::test]
    async fn test_pre_resolve_skips_already_populated_data() {
        use unified_llm_types::{ContentPart, ImageData, Message, Role};

        let original_data = vec![1, 2, 3, 4];
        let mut messages = vec![Message {
            role: Role::User,
            content: vec![ContentPart::Image {
                image: ImageData {
                    url: Some("/some/local/path.png".to_string()),
                    data: Some(original_data.clone()),
                    media_type: Some("image/png".to_string()),
                    detail: None,
                },
            }],
            name: None,
            tool_call_id: None,
        }];

        pre_resolve_local_images(&mut messages).await.unwrap();

        // Already-populated data should not be overwritten
        if let ContentPart::Image { image } = &messages[0].content[0] {
            assert_eq!(image.data.as_ref().unwrap(), &original_data);
        } else {
            panic!("Expected Image content part");
        }
    }
}
