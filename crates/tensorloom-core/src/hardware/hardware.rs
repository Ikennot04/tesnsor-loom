use serde::{Deserialize, Serialize};
use wgpu::{Backends, DeviceType, Instance, InstanceDescriptor};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DetectedGpu {
    /// Index to use with burn's WgpuDevice::DiscreteGpu(id) / IntegratedGpu(id)
    pub id: usize,
    pub name: String,
    pub device_type: String, // "Discrete", "Integrated", "Virtual"
    pub backend: String,     // "Metal", "Vulkan", "Dx12", ...
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HardwareInventory {
    pub gpus: Vec<DetectedGpu>,
    pub fallback_cpu: bool,
}

/// Enumerates adapters via wgpu (Metal, Vulkan, DX12) without initializing any Burn device.
pub fn detect_available_hardware() -> HardwareInventory {
    let mut desc = InstanceDescriptor::new_without_display_handle();
    desc.backends = Backends::PRIMARY;
    let instance = Instance::new(desc);

    let mut gpus = Vec::new();
    let (mut discrete_idx, mut integrated_idx, mut virtual_idx) = (0usize, 0usize, 0usize);

    let adapters = pollster::block_on(instance.enumerate_adapters(Backends::PRIMARY));

    for adapter in adapters {
        let info = adapter.get_info();

        // Burn indexes each device kind separately, so keep a counter per kind
        let (kind, id) = match info.device_type {
            DeviceType::DiscreteGpu => {
                discrete_idx += 1;
                ("Discrete", discrete_idx - 1)
            }
            DeviceType::IntegratedGpu => {
                integrated_idx += 1;
                ("Integrated", integrated_idx - 1)
            }
            DeviceType::VirtualGpu => {
                virtual_idx += 1;
                ("Virtual", virtual_idx - 1)
            }
            _ => continue, // skip CPU/software adapters; CPU is the fallback below
        };

        gpus.push(DetectedGpu {
            id,
            name: info.name,
            device_type: kind.to_string(),
            backend: format!("{:?}", info.backend),
        });
    }

    HardwareInventory {
        gpus,
        fallback_cpu: true,
    }
}