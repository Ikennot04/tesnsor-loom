use crate::config::TrainConfig;
use crate::events::{EdgeBlock, NetworkSnapshot, TrainEvent};
use crate::model::{PortableModel, PrimitiveLayer};
use std::sync::mpsc::Sender;

// Burn ML core imports
use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use burn::module::{AutodiffModule, Module};
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::activation;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Tensor, TensorData};

/// Activation applied after a layer. Kept outside the Burn module so any
/// number of layers can each pick their own.
#[derive(Clone, Copy, Debug)]
pub enum Act {
    Linear,
    Relu,
    Tanh,
    Sigmoid,
}

impl Act {
    fn parse(name: &str) -> Result<Self, String> {
        match name.to_lowercase().as_str() {
            "none" | "linear" => Ok(Act::Linear),
            "relu" => Ok(Act::Relu),
            "tanh" => Ok(Act::Tanh),
            "sigmoid" => Ok(Act::Sigmoid),
            other => Err(format!(
                "Unknown activation '{other}'. Use one of: none, relu, tanh, sigmoid."
            )),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Act::Linear => "none",
            Act::Relu => "relu",
            Act::Tanh => "tanh",
            Act::Sigmoid => "sigmoid",
        }
    }

    fn apply<B: Backend>(self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        match self {
            Act::Linear => x,
            Act::Relu => activation::relu(x),
            Act::Tanh => activation::tanh(x),
            Act::Sigmoid => activation::sigmoid(x),
        }
    }
}

/// Core neural network: any number of dense layers.
#[derive(Module, Debug)]
pub struct TensorLoomNetwork<B: Backend> {
    pub layers: Vec<Linear<B>>,
}

impl<B: Backend> TensorLoomNetwork<B> {
    /// Returns the post-activation output of every layer; the last one is the prediction.
    pub fn forward_all(&self, input: Tensor<B, 2>, acts: &[Act]) -> Vec<Tensor<B, 2>> {
        let mut outputs = Vec::with_capacity(self.layers.len());
        let mut x = input;
        for (layer, act) in self.layers.iter().zip(acts) {
            x = act.apply(layer.forward(x));
            outputs.push(x.clone());
        }
        outputs
    }
}

/// Parsed dataset: row-major features and targets.
struct Dataset {
    features: Vec<f32>,
    targets: Vec<f32>,
    rows: usize,
}

/// Parse CSV where each row is `feature_1,...,feature_n,target_1,...,target_m`.
/// A non-numeric first row is treated as a header and skipped.
fn parse_csv(raw: &str, in_features: usize, out_features: usize) -> Result<Dataset, String> {
    let expected = in_features + out_features;
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

        if values.len() != expected {
            return Err(format!(
                "CSV line {}: expected {} columns ({} features + {} targets), found {}",
                line_no + 1,
                expected,
                in_features,
                out_features,
                values.len()
            ));
        }

        features.extend_from_slice(&values[..in_features]);
        targets.extend_from_slice(&values[in_features..]);
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

/// `sizes` = [input, hidden..., output]; `layer_means` has one activity vector per entry in `sizes`.
fn build_snapshot<B: AutodiffBackend>(
    epoch: usize,
    model: &TensorLoomNetwork<B>,
    sizes: &[usize],
    layer_means: Vec<Vec<f32>>,
) -> Result<NetworkSnapshot, String> {
    // Detach from autodiff so we can read the raw weights.
    let inner = model.valid();

    // Burn Linear weights are row-major [d_input, d_output], which matches
    // the frontend's `weights[r * cols + c]` indexing.
    let mut edges = Vec::with_capacity(inner.layers.len());
    for (i, layer) in inner.layers.iter().enumerate() {
        edges.push(EdgeBlock {
            rows: sizes[i],
            cols: sizes[i + 1],
            weights: tensor_to_vec(layer.weight.val())?,
        });
    }

    Ok(NetworkSnapshot {
        epoch,
        layer_sizes: sizes.to_vec(),
        shown_sizes: sizes.to_vec(), // everything is shown
        node_activity: layer_means,
        edges,
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
    // ---- Topology: sizes = [input, hidden..., output], one activation per layer ----
    let input_dim = config.layers.in_features;
    let output_dim = config.layers.out_features;
    if input_dim == 0 || output_dim == 0 {
        return Err("Input features and output units must be at least 1".to_string());
    }

    let mut sizes = vec![input_dim];
    let mut acts: Vec<Act> = Vec::new();
    for (i, h) in config.layers.hidden_layers.iter().enumerate() {
        if h.units == 0 {
            return Err(format!("Hidden layer {} must have at least 1 unit", i + 1));
        }
        sizes.push(h.units);
        acts.push(Act::parse(&h.activation)?);
    }
    sizes.push(output_dim);
    acts.push(Act::parse(&config.layers.output_activation)?);

    let data = parse_csv(&raw_data, input_dim, output_dim)?;
    let n = data.rows;

    let _ = tx.send(TrainEvent::Log {
        level: "info".into(),
        message: format!(
            "Loaded {} rows. Network: {}",
            n,
            sizes
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join(" -> ")
        ),
    });

    let mut model: TensorLoomNetwork<B> = TensorLoomNetwork {
        layers: sizes
            .windows(2)
            .map(|w| LinearConfig::new(w[0], w[1]).init(&device))
            .collect(),
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
    let snapshot_every = (config.epochs / 20).max(1);

    for epoch in 1..=config.epochs {
        let _ = tx.send(TrainEvent::EpochStarted { epoch });

        let outputs = model.forward_all(x_train.clone(), &acts);
        let predictions = outputs.last().expect("network has at least one layer").clone();

        let emit_snapshot = epoch == 1 || epoch == config.epochs || epoch % snapshot_every == 0;
        let layer_means = if emit_snapshot {
            let mut means = vec![input_mean.clone()];
            for out in &outputs {
                means.push(tensor_to_vec(out.clone().detach().mean_dim(0))?);
            }
            Some(means)
        } else {
            None
        };

        let loss = predictions.sub(y_targets.clone()).powf_scalar(2.0).mean();
        let loss_f = tensor_to_vec(loss.clone())?[0];

        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(lr, model, grads);

        if let Some(means) = layer_means {
            let snapshot = build_snapshot(epoch, &model, &sizes, means)?;
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

    // Drop autodiff tracking and export every layer.
    let inner = model.valid();
    let mut exportable_layers = Vec::with_capacity(inner.layers.len());
    for (i, layer) in inner.layers.iter().enumerate() {
        let (n_in, n_out) = (sizes[i], sizes[i + 1]);
        let bias = match &layer.bias {
            Some(b) => tensor_to_vec(b.val())?,
            None => vec![0.0; n_out],
        };
        exportable_layers.push(PrimitiveLayer {
            weights: tensor_to_vec(layer.weight.val())?,
            bias,
            shape: (n_in, n_out),
            activation: acts[i].name().to_string(),
        });
    }

    let _ = tx.send(TrainEvent::TrainingFinished { success: true });

    Ok(PortableModel {
        layers: exportable_layers,
        metadata: "TensorLoom Engine v1.0 Cross-Platform".to_string(),
    })
}