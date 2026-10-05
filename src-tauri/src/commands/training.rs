use std::sync::mpsc::channel;
use tauri::{AppHandle, Emitter};
use tensorloom_core::config::TrainConfig;

use crate::export::export_model;

/// Starts a training run on a background thread and returns immediately.
///
/// Progress reaches the UI as events:
/// - `training-progress`: every `TrainEvent` from the engine
/// - `training-complete`: the output path, after the model is exported
/// - `training-error`: an error message
#[tauri::command]
pub async fn start_training_session(
    app: AppHandle,
    config: TrainConfig,
    data: String,
    export_format: String,
    output_path: String,
) -> Result<String, String> {
    println!("Starting training session with config: {:?}", config);
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

        let result = tensorloom_core::train::execute_training(config, data, tx)
            .map_err(|e| e.to_string())
            .and_then(|model| export_model(&model, &export_format, &output_path));

        // `tx` was moved into the engine and is dropped when it returns, so this ends.
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