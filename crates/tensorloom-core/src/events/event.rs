use serde::{Deserialize, Serialize};

/// Weights between two adjacent layers, row-major: rows = "from" nodes, cols = "to" nodes.
/// Only the nodes that are shown in the UI are included (see `NetworkSnapshot::shown_sizes`).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EdgeBlock {
    pub rows: usize,
    pub cols: usize,
    pub weights: Vec<f32>,
}

/// A downsampled picture of the network at one point in training.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NetworkSnapshot {
    pub epoch: usize,
    /// Real layer widths, e.g. [3, 16, 1].
    pub layer_sizes: Vec<usize>,
    /// Widths actually sent (capped so the payload stays small), e.g. [3, 12, 1].
    pub shown_sizes: Vec<usize>,
    /// Mean activation of each shown node, one Vec per layer.
    pub node_activity: Vec<Vec<f32>>,
    /// One block per pair of adjacent layers (len = layers - 1).
    pub edges: Vec<EdgeBlock>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type", content = "payload")]
pub enum TrainEvent {
    EpochStarted {
        epoch: usize,
    },
    BatchCompleted {
        loss: f32,
        accuracy: f32,
        progress: f32,
    },
    EpochCompleted {
        epoch: usize,
        avg_loss: f32,
        val_accuracy: f32,
    },
    TrainingFinished {
        success: bool,
    },

    // ---- new in this update ----
    /// Live picture of the network (weights + node activity).
    Network(NetworkSnapshot),
    /// Free-form message from the backend, shown in the UI log.
    Log {
        level: String,
        message: String,
    },
}