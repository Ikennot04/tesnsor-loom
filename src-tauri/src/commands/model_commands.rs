use serde::Serialize;
use tensorloom_core::model::{evaluate, Evaluation, PortableModel};

#[derive(Serialize)]
pub struct LayerSummary {
    pub inputs: usize,
    pub outputs: usize,
    pub activation: String,
}

#[derive(Serialize)]
pub struct ModelSummary {
    pub path: String,
    pub metadata: String,
    pub input_size: usize,
    pub output_size: usize,
    pub parameters: usize,
    pub layers: Vec<LayerSummary>,
}

/// Read and validate an exported model (.json or .bin).
fn read_model(path: &str) -> Result<PortableModel, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("Cannot read {path}: {e}"))?;
    let ext = path.rsplit('.').next().map(|e| e.to_ascii_lowercase());

    let model: PortableModel = match ext.as_deref() {
        Some("json") => serde_json::from_slice(&bytes).map_err(|e| format!("Invalid JSON model: {e}"))?,
        Some("bin") => bincode::deserialize(&bytes).map_err(|e| format!("Invalid binary model: {e}"))?,
        Some("onnx") => {
            return Err("The ONNX export is only a placeholder, so it can't be loaded".to_string())
        }
        _ => return Err("Unsupported file type. Choose a .json or .bin model.".to_string()),
    };

    model.validate()?;
    Ok(model)
}

#[tauri::command]
pub fn load_model(path: String) -> Result<ModelSummary, String> {
    let model = read_model(&path)?;
    Ok(ModelSummary {
        input_size: model.input_size().unwrap_or(0),
        output_size: model.output_size().unwrap_or(0),
        parameters: model.param_count(),
        layers: model
            .layers
            .iter()
            .map(|l| LayerSummary {
                inputs: l.shape.0,
                outputs: l.shape.1,
                activation: l.activation.clone(),
            })
            .collect(),
        metadata: model.metadata,
        path,
    })
}

/// Run a batch of rows through the model. The file is re-read each call: it is tiny,
/// and it keeps the backend stateless.
#[tauri::command]
pub fn predict_rows(path: String, rows: Vec<Vec<f32>>) -> Result<Vec<Vec<f32>>, String> {
    let model = read_model(&path)?;
    rows.iter().map(|row| model.predict(row)).collect()
}

#[tauri::command]
pub fn evaluate_model(path: String, data: String) -> Result<Evaluation, String> {
    let model = read_model(&path)?;
    evaluate(&model, &data)
}