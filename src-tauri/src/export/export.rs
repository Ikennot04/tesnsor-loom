use std::fs;
use std::path::Path;

use serde::Serialize;

/// Saves a portable model to disk in the requested format.
///
/// Supported `format` values (case-insensitive):
/// - `"json"`: pretty JSON, human-readable but large
/// - `"mpk"`: MessagePack, compact binary
///
/// The matching file extension is added to `output_path` automatically.
pub fn export_model<T: Serialize>(
    model: &T,
    format: &str,
    output_path: &str,
) -> Result<(), String> {
    let format = format.to_lowercase();

    let (bytes, ext) = match format.as_str() {
        "json" => (
            serde_json::to_vec_pretty(model)
                .map_err(|e| format!("Failed to serialize model (json): {e}"))?,
            "json",
        ),
        "mpk" | "messagepack" => (
            rmp_serde::to_vec_named(model)
                .map_err(|e| format!("Failed to serialize model (mpk): {e}"))?,
            "mpk",
        ),
        other => {
            return Err(format!(
                "Unsupported export format '{other}'. Use one of: json, mpk."
            ));
        }
    };

    let target = Path::new(output_path).with_extension(ext);

    if let Some(parent) = target.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create output directory: {e}"))?;
        }
    }

    fs::write(&target, bytes).map_err(|e| format!("Failed to write model file: {e}"))
}