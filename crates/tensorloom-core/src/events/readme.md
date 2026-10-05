# Network Snapshots and Training Events

This module defines the data structures used to send **live neural network information** from the training backend to the UI.

It provides:

* Neural network weight information
* Node activation/activity information
* Downsampled network snapshots
* Training progress events
* Epoch and batch statistics
* Backend log messages
* Training completion status

All structures implement Serde serialization and deserialization, allowing them to be transmitted between the backend and frontend.

```rust
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EdgeBlock {
    pub rows: usize,
    pub cols: usize,
    pub weights: Vec<f32>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NetworkSnapshot {
    pub epoch: usize,
    pub layer_sizes: Vec<usize>,
    pub shown_sizes: Vec<usize>,
    pub node_activity: Vec<Vec<f32>>,
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

    Network(NetworkSnapshot),

    Log {
        level: String,
        message: String,
    },
}
```

## `EdgeBlock`

`EdgeBlock` represents the **weights connecting two adjacent neural network layers**.

The weights are stored in row-major order:

```text
rows = nodes in the source/from layer
cols = nodes in the destination/to layer
```

For example, if a network contains:

```text
Input Layer       Hidden Layer
    3       →          4
```

The corresponding weight matrix is:

```text
        Hidden
        0  1  2  3
      ┌─────────────
Input 0│ w w w w
     1│ w w w w
     2│ w w w w
```

Therefore:

```text
rows = 3
cols = 4
weights.len() = 12
```

### Structure

| Field     | Type       | Description                    |
| --------- | ---------- | ------------------------------ |
| `rows`    | `usize`    | Number of source/from nodes    |
| `cols`    | `usize`    | Number of destination/to nodes |
| `weights` | `Vec<f32>` | Flattened weight matrix        |

### Example

```rust
let edges = EdgeBlock {
    rows: 3,
    cols: 4,
    weights: vec![
        0.12, -0.31, 0.45, 0.08,
        0.21,  0.17, -0.22, 0.39,
        0.05,  0.28,  0.11, -0.14,
    ],
};
```

The number of weights should satisfy:

```text
weights.len() = rows × cols
```

---

# `NetworkSnapshot`

`NetworkSnapshot` represents a **downsampled view of the neural network at a specific point during training**.

It contains:

* Current epoch
* Actual network layer sizes
* Layer sizes sent to the UI
* Node activation values
* Connection weights

The snapshot is intentionally downsampled so large neural networks do not generate unnecessarily large UI payloads.

### Structure

| Field           | Type             | Description                                     |
| --------------- | ---------------- | ----------------------------------------------- |
| `epoch`         | `usize`          | Training epoch associated with the snapshot     |
| `layer_sizes`   | `Vec<usize>`     | Actual size of every network layer              |
| `shown_sizes`   | `Vec<usize>`     | Number of nodes actually sent to the UI         |
| `node_activity` | `Vec<Vec<f32>>`  | Mean activation of each displayed node          |
| `edges`         | `Vec<EdgeBlock>` | Weight blocks between adjacent displayed layers |

### Example

For a network:

```text
Input → Hidden → Output
  3       16       1
```

The actual layer sizes are:

```rust
layer_sizes: vec![3, 16, 1]
```

If the UI limits the hidden layer to 12 nodes:

```rust
shown_sizes: vec![3, 12, 1]
```

This means the backend knows that the real network contains 16 hidden nodes, but only 12 are transmitted for visualization.

### Example

```rust
let snapshot = NetworkSnapshot {
    epoch: 10,

    layer_sizes: vec![3, 16, 1],

    shown_sizes: vec![3, 12, 1],

    node_activity: vec![
        vec![0.42, 0.71, 0.18],
        vec![
            0.12, 0.83, 0.44, 0.61,
            0.37, 0.52, 0.29, 0.91,
            0.33, 0.68, 0.47, 0.55,
        ],
        vec![0.76],
    ],

    edges: vec![
        EdgeBlock {
            rows: 3,
            cols: 12,
            weights: vec![/* 36 weights */],
        },

        EdgeBlock {
            rows: 12,
            cols: 1,
            weights: vec![/* 12 weights */],
        },
    ],
};
```

## Network Snapshot Structure

```text
NetworkSnapshot
│
├── epoch
│
├── layer_sizes
│   ├── Input
│   ├── Hidden
│   └── Output
│
├── shown_sizes
│   ├── Displayed Input
│   ├── Displayed Hidden
│   └── Displayed Output
│
├── node_activity
│   ├── Input node activity
│   ├── Hidden node activity
│   └── Output node activity
│
└── edges
    ├── Input → Hidden weights
    └── Hidden → Output weights
```

---

# `TrainEvent`

`TrainEvent` is an enum containing all events that can be emitted by the training backend.

The frontend can use these events to update:

* Training progress
* Loss graphs
* Accuracy graphs
* Epoch information
* Neural network visualization
* Training logs
* Completion state

The enum uses Serde's internally tagged representation:

```rust
#[serde(tag = "type", content = "payload")]
```

This produces serialized messages containing a `type` and corresponding `payload`.

---

## `EpochStarted`

Sent when a new training epoch begins.

```rust
TrainEvent::EpochStarted {
    epoch: 1,
}
```

Conceptually:

```json
{
  "type": "EpochStarted",
  "payload": {
    "epoch": 1
  }
}
```

| Field   | Type    | Description            |
| ------- | ------- | ---------------------- |
| `epoch` | `usize` | Epoch that has started |

---

# `BatchCompleted`

Sent after a training batch has completed.

```rust
TrainEvent::BatchCompleted {
    loss: 0.245,
    accuracy: 0.91,
    progress: 0.45,
}
```

| Field      | Type  | Description               |
| ---------- | ----- | ------------------------- |
| `loss`     | `f32` | Current training loss     |
| `accuracy` | `f32` | Current training accuracy |
| `progress` | `f32` | Overall training progress |

The UI can use `progress` to update a progress bar.

For example:

```text
progress = 0.45

█████████░░░░░░░░░░░ 45%
```

---

# `EpochCompleted`

Sent when an entire training epoch has finished.

```rust
TrainEvent::EpochCompleted {
    epoch: 10,
    avg_loss: 0.183,
    val_accuracy: 0.94,
}
```

| Field          | Type    | Description           |
| -------------- | ------- | --------------------- |
| `epoch`        | `usize` | Completed epoch       |
| `avg_loss`     | `f32`   | Average training loss |
| `val_accuracy` | `f32`   | Validation accuracy   |

This event is useful for updating training graphs.

---

# `TrainingFinished`

Sent when the training process has finished.

```rust
TrainEvent::TrainingFinished {
    success: true,
}
```

| Field     | Type   | Description                             |
| --------- | ------ | --------------------------------------- |
| `success` | `bool` | Whether training completed successfully |

Example:

```text
success = true
    ↓
Training completed successfully
```

or:

```text
success = false
    ↓
Training failed or was interrupted
```

---

# `Network`

The `Network` event sends a live `NetworkSnapshot` to the UI.

```rust
TrainEvent::Network(snapshot)
```

This allows the frontend to visualize the neural network while training is occurring.

The event contains:

```text
TrainEvent
    │
    └── Network
          │
          └── NetworkSnapshot
                ├── layer_sizes
                ├── shown_sizes
                ├── node_activity
                └── edges
```

For example:

```rust
let event = TrainEvent::Network(NetworkSnapshot {
    epoch: 5,
    layer_sizes: vec![3, 16, 1],
    shown_sizes: vec![3, 12, 1],
    node_activity: vec![
        vec![0.2, 0.7, 0.4],
        vec![0.1; 12],
        vec![0.8],
    ],
    edges: vec![],
});
```

The UI can use this event to update a live visualization such as:

```text
     INPUT          HIDDEN          OUTPUT

       ○ ──────── ○
       ○ ──────── ○ ──────── ○
       ○ ──────── ○
                 ...
                 ○
```

Node activity can control the visual representation of each node, while edge weights can be used to represent connection strength.

---

# `Log`

`Log` provides a free-form message from the backend.

```rust
TrainEvent::Log {
    level: "INFO".to_string(),
    message: "Training started".to_string(),
}
```

| Field     | Type     | Description                 |
| --------- | -------- | --------------------------- |
| `level`   | `String` | Log severity/category       |
| `message` | `String` | Message displayed in the UI |

Example log levels:

```text
TRACE
DEBUG
INFO
WARN
ERROR
```

Example:

```json
{
  "type": "Log",
  "payload": {
    "level": "INFO",
    "message": "Epoch 10 completed"
  }
}
```

---

# Event Flow

A typical training session can produce events in the following order:

```text
Training Starts
      │
      ▼
EpochStarted
      │
      ▼
BatchCompleted
      │
      ├──────► Network
      │
      ▼
BatchCompleted
      │
      ├──────► Network
      │
      ▼
EpochCompleted
      │
      ▼
EpochStarted
      │
     ...
      │
      ▼
TrainingFinished
```

Backend logs can occur at any point:

```text
              ┌── Log
              │
              ├── Network
              │
Training ─────┼── BatchCompleted
              │
              └── EpochCompleted
```

---

# Data Relationship

The relationship between the structures is:

```text
TrainEvent
│
├── EpochStarted
│
├── BatchCompleted
│
├── EpochCompleted
│
├── TrainingFinished
│
├── Network
│     │
│     └── NetworkSnapshot
│           │
│           ├── layer_sizes
│           ├── shown_sizes
│           ├── node_activity
│           └── EdgeBlock
│                 └── weights
│
└── Log
```

This design separates **training state**, **network visualization data**, and **backend messages** while allowing all of them to be transmitted through a single `TrainEvent` interface.
