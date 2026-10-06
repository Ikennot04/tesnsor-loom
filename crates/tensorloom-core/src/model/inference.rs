// Pure-Rust forward pass over an exported PortableModel. No Burn needed, so it is
// cheap to call and works the same on every platform.

use super::PortableModel;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OutputEvaluation {
    pub index: usize,
    pub mse: f32,
    /// 1.0 = perfect, 0.0 = no better than predicting the average, < 0 = worse than that.
    pub r2: f32,
    pub mean_target: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Evaluation {
    pub rows: usize,
    /// Average of the per-output MSE values.
    pub mse: f32,
    /// Average of the per-output R² values (1.0 = perfect, 0.0 = no better than the mean, < 0 = worse).
    pub r2: f32,
    /// Average of the per-output target means.
    pub mean_target: f32,
    /// One entry per model output, in output order.
    #[serde(default)]
    pub per_output: Vec<OutputEvaluation>,
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

/// Parse `feature_1,...,feature_n,target_1,...,target_m` rows. A non-numeric first row is a header.
fn parse_rows(
    raw: &str,
    in_features: usize,
    out_features: usize,
) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>), String> {
    let expected_cols = in_features + out_features;
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
        if values.len() != expected_cols {
            return Err(format!(
                "CSV line {}: expected {} columns ({} features + {} targets), found {}",
                line_no + 1,
                expected_cols,
                in_features,
                out_features,
                values.len()
            ));
        }
        rows.push(values[..in_features].to_vec());
        targets.push(values[in_features..].to_vec());
    }

    if rows.is_empty() {
        return Err("CSV contained no data rows".to_string());
    }
    Ok((rows, targets))
}

/// Score a model on a CSV that includes the target column(s), one per model output.
pub fn evaluate(model: &PortableModel, raw_csv: &str) -> Result<Evaluation, String> {
    model.validate()?;
    let n_in = model.input_size().ok_or("model has no layers")?;
    let n_out = model.output_size().ok_or("model has no layers")?;

    let (rows, targets) = parse_rows(raw_csv, n_in, n_out)?;
    let n = rows.len() as f32;

    // Per-output target means.
    let mut means = vec![0.0f32; n_out];
    for t in &targets {
        for (m, v) in means.iter_mut().zip(t) {
            *m += v;
        }
    }
    for m in &mut means {
        *m /= n;
    }

    let mut sq_err = vec![0.0f32; n_out];
    let mut sq_var = vec![0.0f32; n_out];
    for (row, t) in rows.iter().zip(&targets) {
        let pred = model.predict_unchecked(row)?;
        for j in 0..n_out {
            let e = pred[j] - t[j];
            let d = t[j] - means[j];
            sq_err[j] += e * e;
            sq_var[j] += d * d;
        }
    }

    let per_output: Vec<OutputEvaluation> = (0..n_out)
        .map(|j| {
            let mse = sq_err[j] / n;
            let var = sq_var[j] / n;
            let r2 = if var > f32::EPSILON { 1.0 - mse / var } else { 0.0 };
            OutputEvaluation { index: j, mse, r2, mean_target: means[j] }
        })
        .collect();

    let k = n_out as f32;
    Ok(Evaluation {
        rows: rows.len(),
        mse: per_output.iter().map(|o| o.mse).sum::<f32>() / k,
        r2: per_output.iter().map(|o| o.r2).sum::<f32>() / k,
        mean_target: per_output.iter().map(|o| o.mean_target).sum::<f32>() / k,
        per_output,
    })
}
