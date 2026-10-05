mod commands;
mod export;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            commands::training::start_training_session,
            commands::hardware::detect_hardware,
            commands::model_commands::load_model,
            commands::model_commands::predict_rows,
            commands::model_commands::evaluate_model,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}