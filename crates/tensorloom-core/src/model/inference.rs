// Pure-Rust forward pass over an exported PortableModel. No Burn needed, so it is
// cheap to call and works the same on every platform.

use super::PortableModel;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Evaluation {
    pub rows: usize,
    pub mse: f32,
    /// 1.0 = perfect, 0.0 = no better than predicting the average, < 0 = worse than that.
    pub r2: f32,
    pub mean_target: f32,
}

fn activate(name: &str, v: f32) -> Result<f32, String> {
    match name {
        "relu" => Ok(v.max(0.0)),
        "none" | "linear" | "" => Ok(v),
        "sigmoid" => Ok(1.0 / (1.0 + (-v).exp())),
        "tanh" => Ok(v.tanh()),
        other => Err(format!("unknown activation '{other}'")),
    }
}

impl PortableModel {
    pub fn input_size(&self) -> Option<usize> {
        self.layers.first().map(|l| l.shape.0)
    }

    pub fn output_size(&self) -> Option<usize> {
        self.layers.last().map(|l| l.shape.1)
    }

    pub fn param_count(&self) -> usize {
        self.layers.iter().map(|l| l.weights.len() + l.bias.len()).sum()
    }

    /// Check that shapes, buffers and activations are consistent before running anything.
    pub fn validate(&self) -> Result<(), String> {
        if self.layers.is_empty() {
            return Err("model has no layers".to_string());
        }
        for (k, layer) in self.layers.iter().enumerate() {
            let (n_in, n_out) = layer.shape;
            if layer.weights.len() != n_in * n_out {
                return Err(format!(
                    "layer {k}: expected {} weights for shape ({n_in}, {n_out}), found {}",
                    n_in * n_out,
                    layer.weights.len()
                ));
            }
            if layer.bias.len() != n_out {
                return Err(format!(
                    "layer {k}: expected {n_out} biases, found {}",
                    layer.bias.len()
                ));
            }
            activate(&layer.activation, 0.0).map_err(|e| format!("layer {k}: {e}"))?;
            if let Some(next) = self.layers.get(k + 1) {
                if next.shape.0 != n_out {
                    return Err(format!(
                        "layer {k} outputs {n_out} values but layer {} expects {}",
                        k + 1,
                        next.shape.0
                    ));
                }
            }
        }
        Ok(())
    }

    /// Run one input row through the network.
    pub fn predict(&self, input: &[f32]) -> Result<Vec<f32>, String> {
        self.validate()?;
        self.predict_unchecked(input)
    }

    /// Same as `predict`, but assumes `validate()` already passed (used for big batches).
    fn predict_unchecked(&self, input: &[f32]) -> Result<Vec<f32>, String> {
        let expected = self.layers[0].shape.0;
        if input.len() != expected {
            return Err(format!("expected {expected} input values, got {}", input.len()));
        }
        if let Some(bad) = input.iter().find(|v| !v.is_finite()) {
            return Err(format!("input contains a non-finite value ({bad})"));
        }

        let mut current = input.to_vec();
        for (k, layer) in self.layers.iter().enumerate() {
            let (n_in, n_out) = layer.shape;
            let mut next = Vec::with_capacity(n_out);
            for j in 0..n_out {
                // Weights are row-major [n_in, n_out]: weight from input i to output j.
                let mut sum = layer.bias[j];
                for i in 0..n_in {
                    sum += current[i] * layer.weights[i * n_out + j];
                }
                next.push(activate(&layer.activation, sum).map_err(|e| format!("layer {k}: {e}"))?);
            }
            current = next;
        }
        Ok(current)
    }
}

/// Parse `feature_1,...,feature_n,target` rows. A non-numeric first row is a header.
fn parse_rows(raw: &str, in_features: usize) -> Result<(Vec<Vec<f32>>, Vec<f32>), String> {
    let mut rows = Vec::new();
    let mut targets = Vec::new();

    for (line_no, line) in raw.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let parsed: Result<Vec<f32>, _> = line.split(',').map(|c| c.trim().parse::<f32>()).collect();
        let values = match parsed {
            Ok(v) => v,
            Err(_) if rows.is_empty() => continue, // header
            Err(e) => return Err(format!("CSV line {}: {}", line_no + 1, e)),
        };
        if values.len() != in_features + 1 {
            return Err(format!(
                "CSV line {}: expected {} columns ({} features + 1 target), found {}",
                line_no + 1,
                in_features + 1,
                in_features,
                values.len()
            ));
        }
        rows.push(values[..in_features].to_vec());
        targets.push(values[in_features]);
    }

    if rows.is_empty() {
        return Err("CSV contained no data rows".to_string());
    }
    Ok((rows, targets))
}

/// Score a model on a CSV that includes the target column (single-output models only).
pub fn evaluate(model: &PortableModel, raw_csv: &str) -> Result<Evaluation, String> {
    model.validate()?;
    let n_in = model.input_size().ok_or("model has no layers")?;
    if model.output_size() != Some(1) {
        return Err("evaluate currently supports models with exactly one output".to_string());
    }

    let (rows, targets) = parse_rows(raw_csv, n_in)?;
    let n = rows.len() as f32;

    let mean_target = targets.iter().sum::<f32>() / n;
    let mut sq_err = 0.0f32;
    let mut sq_var = 0.0f32;
    for (row, &t) in rows.iter().zip(&targets) {
        let p = model.predict_unchecked(row)?[0];
        sq_err += (p - t) * (p - t);
        sq_var += (t - mean_target) * (t - mean_target);
    }

    let mse = sq_err / n;
    let var = sq_var / n;
    let r2 = if var > f32::EPSILON { 1.0 - mse / var } else { 0.0 };

    Ok(Evaluation { rows: rows.len(), mse, r2, mean_target })
}