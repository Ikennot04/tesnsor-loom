// Adjust this path to wherever tensorloom-core exposes model.rs.
// If `model` re-exports them: tensorloom_core::model::{PortableModel, PrimitiveLayer}
// Otherwise:                  tensorloom_core::model::model::{PortableModel, PrimitiveLayer}
use tensorloom_core::model::{PortableModel, PrimitiveLayer};

use super::onnx::{Activation, DenseLayerSpec, OnnxExportable};

fn parse_activation(name: &str) -> Result<Activation, String> {
    match name.trim().to_lowercase().as_str() {
        "" | "none" | "linear" | "identity" => Ok(Activation::None),
        "relu" => Ok(Activation::Relu),
        "sigmoid" => Ok(Activation::Sigmoid),
        "tanh" => Ok(Activation::Tanh),
        "softmax" => Ok(Activation::Softmax),
        other => Err(format!("ONNX export: unsupported activation '{other}'")),
    }
}

fn layer_spec(l: &PrimitiveLayer) -> Result<DenseLayerSpec<'_>, String> {
    let (a, b) = l.shape;

    // `shape` is assumed to be (in, out), Burn's Linear layout. The bias length
    // tells us when it is actually (out, in). For square layers this can't be
    // distinguished, so (in, out) wins.
    let (in_features, out_features, transposed) = if l.bias.len() == b {
        (a, b, false)
    } else if l.bias.len() == a {
        (b, a, true)
    } else {
        return Err(format!(
            "ONNX export: bias length {} matches neither dimension of shape {:?}",
            l.bias.len(),
            l.shape
        ));
    };

    Ok(DenseLayerSpec {
        in_features,
        out_features,
        weights: &l.weights,
        bias: &l.bias,
        activation: parse_activation(&l.activation)?,
        transposed,
    })
}

impl OnnxExportable for PortableModel {
    fn onnx_layers(&self) -> Result<Vec<DenseLayerSpec<'_>>, String> {
        self.layers.iter().map(layer_spec).collect()
    }
}
