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

/// Resolve a local file path to its raw bytes and inferred MIME type.
///
/// Expands `~/` to the user's home directory via `$HOME`.
/// Returns `(file_bytes, mime_type)` on success.
pub fn resolve_local_file(url: &str) -> Result<(Vec<u8>, String), unified_llm_types::Error> {
    let path = if let Some(rest) = url.strip_prefix("~/") {
        let home = std::env::var("HOME").unwrap_or_default();
        Path::new(&home).join(rest)
    } else {
        Path::new(url).to_path_buf()
    };

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
}
