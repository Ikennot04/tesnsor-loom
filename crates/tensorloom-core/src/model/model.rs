use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PrimitiveLayer {
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
    pub shape: (usize, usize),
    pub activation: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PortableModel {
    pub layers: Vec<PrimitiveLayer>,
    pub metadata: String,
}
