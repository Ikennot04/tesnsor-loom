use crate::config::TrainConfig;
use crate::events::{EdgeBlock, NetworkSnapshot, TrainEvent};
use crate::model::{PortableModel, PrimitiveLayer};
use std::sync::mpsc::Sender;

// Burn ML core imports
use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use burn::module::{AutodiffModule, Module};
use burn::nn::{Linear, LinearConfig, Relu};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Tensor, TensorData};

/// Core neural network topology.
#[derive(Module, Debug)]
pub struct TensorLoomNetwork<B: Backend> {
    pub input_layer: Linear<B>,
    pub output_layer: Linear<B>,
    pub activation: Relu,
}

impl<B: Backend> TensorLoomNetwork<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        self.forward_with_hidden(input).1
    }

    /// Returns (hidden activations, predictions) so the trainer can report node activity.
    pub fn forward_with_hidden(&self, input: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let hidden = self.activation.forward(self.input_layer.forward(input));
        let output = self.output_layer.forward(hidden.clone());
        (hidden, output)
    }
}

/// Parsed dataset: row-major features plus one target per row.
struct Dataset {
    features: Vec<f32>,
    targets: Vec<f32>,
    rows: usize,
}

/// Parse CSV where each row is `feature_1,...,feature_n,target`.
/// A non-numeric first row is treated as a header and skipped.
fn parse_csv(raw: &str, in_features: usize) -> Result<Dataset, String> {
    let mut features = Vec::new();
    let mut targets = Vec::new();
    let mut rows = 0usize;

    for (line_no, line) in raw.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let cells: Vec<&str> = line.split(',').map(|c| c.trim()).collect();
        let parsed: Result<Vec<f32>, _> = cells.iter().map(|c| c.parse::<f32>()).collect();

        let values = match parsed {
            Ok(v) => v,
            Err(_) if rows == 0 => continue, // header row
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

        features.extend_from_slice(&values[..in_features]);
        targets.push(values[in_features]);
        rows += 1;
    }

    if rows == 0 {
        return Err("CSV contained no data rows".to_string());
    }
    Ok(Dataset { features, targets, rows })
}

/// Dynamic pipeline routing gateway.
pub fn execute_training(
    config: TrainConfig,
    raw_csv_data: String,
    tx: Sender<TrainEvent>,
) -> Result<PortableModel, String> {
    match config.hardware.target_mode.as_str() {
        "GPU_DISCRETE" => {
            let device = WgpuDevice::DiscreteGpu(config.hardware.device_index);
            run_burn_loop::<Autodiff<Wgpu>>(config, raw_csv_data, tx, device)
        }
        "GPU_INTEGRATED" => {
            let device = WgpuDevice::IntegratedGpu(config.hardware.device_index);
            run_burn_loop::<Autodiff<Wgpu>>(config, raw_csv_data, tx, device)
        }
        _ => {
            let device = NdArrayDevice::Cpu;
            run_burn_loop::<Autodiff<NdArray>>(config, raw_csv_data, tx, device)
        }
    }
}

fn tensor_to_vec<B: Backend, const D: usize>(t: Tensor<B, D>) -> Result<Vec<f32>, String> {
    t.into_data()
        .to_vec::<f32>()
        .map_err(|e| format!("tensor conversion failed: {:?}", e))
}

// ---------------------------------------------------------------------------
// Snapshot (what the UI draws): every node and every weight, no sampling
// ---------------------------------------------------------------------------

fn build_snapshot<B: AutodiffBackend>(
    epoch: usize,
    model: &TensorLoomNetwork<B>,
    input_mean: &[f32],
    hidden_mean: &[f32],
    output_mean: &[f32],
) -> Result<NetworkSnapshot, String> {
    let (n_in, n_hid, n_out) = (input_mean.len(), hidden_mean.len(), output_mean.len());

    // Detach from autodiff so we can read the raw weights.
    let inner = model.valid();
    // Burn Linear weights are row-major [d_input, d_output], which matches
    // the frontend's `weights[r * cols + c]` indexing.
    let w1 = tensor_to_vec(inner.input_layer.weight.val())?;
    let w2 = tensor_to_vec(inner.output_layer.weight.val())?;

    Ok(NetworkSnapshot {
        epoch,
        layer_sizes: vec![n_in, n_hid, n_out],
        shown_sizes: vec![n_in, n_hid, n_out], // everything is shown now
        node_activity: vec![
            input_mean.to_vec(),
            hidden_mean.to_vec(),
            output_mean.to_vec(),
        ],
        edges: vec![
            EdgeBlock { rows: n_in, cols: n_hid, weights: w1 },
            EdgeBlock { rows: n_hid, cols: n_out, weights: w2 },
        ],
    })
}

/// Generic training loop.
fn run_burn_loop<B>(
    config: TrainConfig,
    raw_data: String,
    tx: Sender<TrainEvent>,
    device: B::Device,
) -> Result<PortableModel, String>
where
    B: AutodiffBackend,
{
    let input_dim = config.layers.in_features;
    let hidden_dim = config.layers.out_features;
    let output_dim = 1;

    let data = parse_csv(&raw_data, input_dim)?;
    let n = data.rows;

    let _ = tx.send(TrainEvent::Log {
        level: "info".into(),
        message: format!("Loaded {} rows, {} features", n, input_dim),
    });

    let mut model: TensorLoomNetwork<B> = TensorLoomNetwork {
        input_layer: LinearConfig::new(input_dim, hidden_dim).init(&device),
        output_layer: LinearConfig::new(hidden_dim, output_dim).init(&device),
        activation: Relu::new(),
    };

    let mut optimizer = AdamConfig::new().init::<B, TensorLoomNetwork<B>>();

    let x_train = Tensor::<B, 2>::from_data(TensorData::new(data.features, [n, input_dim]), &device);
    let y_targets = Tensor::<B, 2>::from_data(TensorData::new(data.targets, [n, output_dim]), &device);

    // Variance of the targets, used to turn MSE into an R^2-style score.
    let target_var = y_targets
        .clone()
        .sub(y_targets.clone().mean().expand([n, output_dim]))
        .powf_scalar(2.0)
        .mean();
    let target_var_f = tensor_to_vec(target_var)?[0].max(f32::EPSILON);

    // Input "activity" never changes, so compute it once.
    let input_mean = tensor_to_vec(x_train.clone().mean_dim(0))?;

    let lr = config.lr as f64;

    // Throttle: at most ~20 snapshots per run, plus first and last epoch.
    // Snapshots now carry every weight, so they are much larger than before.
    let snapshot_every = (config.epochs / 20).max(1);

    for epoch in 1..=config.epochs {
        let _ = tx.send(TrainEvent::EpochStarted { epoch });

        let (hidden, predictions) = model.forward_with_hidden(x_train.clone());

        let emit_snapshot = epoch == 1 || epoch == config.epochs || epoch % snapshot_every == 0;
        let activity = if emit_snapshot {
            Some((
                tensor_to_vec(hidden.clone().detach().mean_dim(0))?,
                tensor_to_vec(predictions.clone().detach().mean_dim(0))?,
            ))
        } else {
            None
        };

        let loss = predictions.sub(y_targets.clone()).powf_scalar(2.0).mean();
        let loss_f = tensor_to_vec(loss.clone())?[0];

        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(lr, model, grads);

        if let Some((hidden_mean, output_mean)) = activity {
            let snapshot = build_snapshot(epoch, &model, &input_mean, &hidden_mean, &output_mean)?;
            let _ = tx.send(TrainEvent::Network(snapshot));
        }

        // R^2 clamped to [0, 1]; meaningful for regression, unlike 1 - MSE.
        let score = (1.0 - loss_f / target_var_f).clamp(0.0, 1.0);

        let _ = tx.send(TrainEvent::BatchCompleted {
            loss: loss_f,
            accuracy: score,
            progress: epoch as f32 / config.epochs as f32,
        });

        let _ = tx.send(TrainEvent::EpochCompleted {
            epoch,
            avg_loss: loss_f,
            val_accuracy: score, // no validation split yet: this is the training score
        });
    }

    // Drop autodiff tracking and export both layers.
    let inner = model.valid();

    let hidden_bias = match &inner.input_layer.bias {
        Some(b) => tensor_to_vec(b.val())?,
        None => vec![0.0; hidden_dim],
    };
    let out_bias = match &inner.output_layer.bias {
        Some(b) => tensor_to_vec(b.val())?,
        None => vec![0.0; output_dim],
    };

    let exportable_layers = vec![
        PrimitiveLayer {
            weights: tensor_to_vec(inner.input_layer.weight.val())?,
            bias: hidden_bias,
            shape: (input_dim, hidden_dim),
            activation: "relu".to_string(),
        },
        PrimitiveLayer {
            weights: tensor_to_vec(inner.output_layer.weight.val())?,
            bias: out_bias,
            shape: (hidden_dim, output_dim),
            activation: "none".to_string(),
        },
    ];

    let _ = tx.send(TrainEvent::TrainingFinished { success: true });

    Ok(PortableModel {
        layers: exportable_layers,
        metadata: "TensorLoom Engine v1.0 Cross-Platform".to_string(),
    })
}