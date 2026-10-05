// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![greet])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

// src-tauri/src/lib.rs
#[tauri::command]
fn system_info() -> tensorloom_core::SystemInfo {
    tensorloom_core::system_info()
}

#[tauri::command]
fn run_job(args: tensorloom_core::RunArgs) -> Result<String, String> {
    tensorloom_core::run_job(args)
}

#[tauri::command]
fn greet(name: String) -> String {
    tensorloom_core::greet(&name)
}
