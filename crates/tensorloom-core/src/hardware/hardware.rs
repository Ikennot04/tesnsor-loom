use serde::{Deserialize, Serialize};
use burn_wgpu::WgpuDevice;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DetectedGpu {
    pub id: usize,
    pub name: String,
    pub device_type: String, // "Discrete", "Integrated", "CPU"
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HardwareInventory {
    pub gpus: Vec<DetectedGpu>,
    pub fallback_cpu: bool,
}

/// Polls system graphic stacks to find available accelerators (Metal, Vulkan, DX12)
pub fn detect_available_hardware() -> HardwareInventory {
    let mut detected_gpus = Vec::new();
    
    // We poll indexes sequentially to inventory what the wgpu stack has initialized
    for index in 0..4 {
        // Test for separate high-powered cards (NVIDIA RTX, AMD Radeon)
        let discrete = WgpuDevice::DiscreteGpu(index);
        if is_device_accessible(&discrete) {
            detected_gpus.push(DetectedGpu {
                id: index,
                name: format!("Discrete GPU Acceleration Block #{}", index),
                device_type: "Discrete".to_string(),
            });
            continue;
        }

        // Test for Unified/Integrated processors (Apple Silicon M5 SoC Unified Memory)
        let integrated = WgpuDevice::IntegratedGpu(index);
        if is_device_accessible(&integrated) {
            let label = if cfg!(target_os = "macos") {
                format!("Apple Silicon Graphics Engine M-Series Core #{}", index)
            } else {
                format!("Integrated System GPU Core #{}", index)
            };
            
            detected_gpus.push(DetectedGpu {
                id: index,
                name: label,
                device_type: "Integrated".to_string(),
            });
        }
    }

    HardwareInventory {
        gpus: detected_gpus,
        fallback_cpu: true,
    }
}

/// Internal handshake helper that runs a safe probe instantiation check 
fn is_device_accessible(device: &WgpuDevice) -> bool {
    // Attempt to register a dummy operation space to verify driver integrity
    std::panic::catch_unwind(|| {
        let _ = burn_wgpu::init_device(device);
    }).is_ok()
}
