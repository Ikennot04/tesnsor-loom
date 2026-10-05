use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LayerConfig {
    pub in_features: usize,
    pub out_features: usize,
    pub activation: String, // "relu", "sigmoid", etc.
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HardwareTarget {
    pub target_mode: String, // "GPU_DISCRETE", "GPU_INTEGRATED", or "CPU"
    pub device_index: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TrainConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub lr: f64,
    pub layers: crate::config::LayerConfig, // Structural layers profile
    pub hardware: HardwareTarget,           // The dynamic configuration payload
}