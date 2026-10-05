use tensorloom_core::hardware::{detect_available_hardware, HardwareInventory};

/// Lists the GPUs the graphics stack can see. Async so the probe never blocks the UI.
#[tauri::command]
pub async fn detect_hardware() -> Result<HardwareInventory, String> {
    Ok(detect_available_hardware())
}