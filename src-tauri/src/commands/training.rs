use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::mpsc::channel;
use tauri::{AppHandle, Emitter};
use tensorloom_core::config::TrainConfig;
use tensorloom_core::data::{Environment, Generated};

use crate::export::export_model;

/// Starts a training run on a background thread and returns immediately.
///
/// Progress reaches the UI as events:
/// - `training-progress`: every `TrainEvent` from the engine
/// - `training-complete`: the output path, after the model is exported
/// - `training-error`: an error message
///
/// `config.environment` (the rules.json contents) is optional. When present, the
/// engine validates it, checks the layer sizes against it, and uses its task block.
#[tauri::command]
pub async fn start_training_session(
    app: AppHandle,
    config: TrainConfig,
    data: String,
    export_format: String,
    output_path: String,
) -> Result<String, String> {
    println!(
        "Starting training session: {} epochs, lr {}, environment: {}",
        config.epochs,
        config.lr,
        config
            .environment
            .as_ref()
            .map(|e| e.name.as_str())
            .unwrap_or("none"),
    );
    let (tx, rx) = channel();

    std::thread::spawn(move || {
        // Forward engine events to the UI while training runs.
        let progress_thread = std::thread::spawn({
            let app = app.clone();
            move || {
                while let Ok(event) = rx.recv() {
                    let _ = app.emit("training-progress", event);
                }
            }
        });

        // A panic inside Burn/wgpu would otherwise kill this thread silently and
        // leave the UI stuck on "Training...". Turn it into a normal error.
        let result = catch_unwind(AssertUnwindSafe(|| {
            tensorloom_core::train::execute_training(config, data, tx)
                .map_err(|e| e.to_string())
                .and_then(|model| export_model(&model, &export_format, &output_path))
        }))
        .unwrap_or_else(|panic| {
            let msg = panic
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic".to_string());
            Err(format!("Training crashed: {msg}"))
        });

        // `tx` was moved into the engine and is dropped when it returns (or unwinds),
        // so this ends.
        let _ = progress_thread.join();

        match result {
            Ok(()) => {
                let _ = app.emit("training-complete", output_path);
            }
            Err(e) => {
                let _ = app.emit("training-error", e);
            }
        }
    });

    Ok("Training initiated".to_string())
}

/// Parses + validates a rules.json payload. Returns (input columns, output columns).
#[tauri::command]
pub fn validate_environment(json: String) -> Result<(usize, usize), String> {
    Ok(Environment::from_json(&json)?.io_shape())
}

/// Generates a training CSV from a rules.json payload.
/// Runs on a blocking thread because puzzle generation is CPU-heavy.
#[tauri::command]
pub async fn generate_dataset(json: String, rows: usize, seed: u64) -> Result<Generated, String> {
    tauri::async_runtime::spawn_blocking(move || {
        Environment::from_json(&json)?.generate(rows, seed)
    })
    .await
    .map_err(|e| e.to_string())?
}