// // Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
// #[tauri::command]
// fn greet(name: &str) -> String {
//     format!("Hello, {}! You've been greeted from Rust!", name)
// }

// #[cfg_attr(mobile, tauri::mobile_entry_point)]
// pub fn run() {
//     tauri::Builder::default()
//         .plugin(tauri_plugin_opener::init())
//         .invoke_handler(tauri::generate_handler![greet])
//         .run(tauri::generate_context!())
//         .expect("error while running tauri application");
// }

// // src-tauri/src/lib.rs
// #[tauri::command]
// fn system_info() -> tensorloom_core::SystemInfo {
//     tensorloom_core::system_info()
// }

// #[tauri::command]
// fn run_job(args: tensorloom_core::RunArgs) -> Result<String, String> {
//     tensorloom_core::run_job(args)
// }
use std::sync::mpsc::channel;
use tauri::{AppHandle, Emitter};
use tensorloom_core::config::TrainConfig;
use tensorloom_core::model::PortableModel;

fn build_onnx_bytes(_model: &PortableModel) -> Vec<u8> {
    // TODO: build a real ModelProto via prost
    vec![0x4f, 0x4e, 0x4e, 0x58]
}

fn export_model(
    model: &PortableModel,
    format: &str,
    path: &str,
) -> Result<(), String> {
    let bytes = match format {
        "json" => serde_json::to_string_pretty(model)
            .map_err(|e| e.to_string())?
            .into_bytes(),
        "bin" => bincode::serialize(model).map_err(|e| e.to_string())?,
        "onnx" => build_onnx_bytes(model),
        other => return Err(format!("Unknown format: {other}")),
    };
    std::fs::write(path, bytes).map_err(|e| e.to_string())
}

#[tauri::command]
async fn start_training_session(
    app: AppHandle,
    config: TrainConfig,
    data: String,
    export_format: String,
    output_path: String,
) -> Result<String, String> {
    println!("Starting training session with config: {:?}", config);
    let (tx, rx) = channel();

    std::thread::spawn(move || {
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

        let _ = progress_thread.join();

        match result {
            Ok(()) => { let _ = app.emit("training-complete", output_path); }
            Err(e) => { let _ = app.emit("training-error", e); }
        }
    });

    Ok("Training initiated".to_string())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![start_training_session])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}