# Configuration Structures

This module defines the configuration structures used to describe neural network layers, hardware targets, and training parameters.

All configuration structures implement:

* `Serialize` — Allows the configuration to be serialized.
* `Deserialize` — Allows the configuration to be loaded from serialized data.
* `Debug` — Enables debugging output.
* `Clone` — Allows configurations to be duplicated.

```rust
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LayerConfig {
    pub in_features: usize,
    pub out_features: usize,
    pub activation: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HardwareTarget {
    pub target_mode: String,
    pub device_index: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TrainConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub lr: f64,
    pub layers: crate::config::LayerConfig,
    pub hardware: HardwareTarget,
}
```

## `LayerConfig`

`LayerConfig` describes the structure of a neural network layer.

| Field          | Type     | Description                           |
| -------------- | -------- | ------------------------------------- |
| `in_features`  | `usize`  | Number of input features/neurons      |
| `out_features` | `usize`  | Number of output features/neurons     |
| `activation`   | `String` | Activation function used by the layer |

### Example

```rust
let layer = LayerConfig {
    in_features: 784,
    out_features: 128,
    activation: "relu".to_string(),
};
```

Common activation functions may include:

```text
relu
sigmoid
tanh
softmax
linear
```

---

## `HardwareTarget`

`HardwareTarget` specifies which hardware device should be used for computation.

| Field          | Type     | Description                           |
| -------------- | -------- | ------------------------------------- |
| `target_mode`  | `String` | Hardware backend/mode                 |
| `device_index` | `usize`  | Index of the selected hardware device |

Supported target modes:

```text
GPU_DISCRETE
GPU_INTEGRATED
CPU
```

### Example

```rust
let hardware = HardwareTarget {
    target_mode: "GPU_DISCRETE".to_string(),
    device_index: 0,
};
```

For a CPU target:

```rust
let hardware = HardwareTarget {
    target_mode: "CPU".to_string(),
    device_index: 0,
};
```

---

## `TrainConfig`

`TrainConfig` contains the parameters required for model training.

| Field        | Type             | Description                           |
| ------------ | ---------------- | ------------------------------------- |
| `epochs`     | `usize`          | Number of complete training passes    |
| `batch_size` | `usize`          | Number of samples processed per batch |
| `lr`         | `f64`            | Learning rate                         |
| `layers`     | `LayerConfig`    | Neural network layer configuration    |
| `hardware`   | `HardwareTarget` | Hardware execution configuration      |

### Example

```rust
let config = TrainConfig {
    epochs: 100,
    batch_size: 32,
    lr: 0.001,

    layers: LayerConfig {
        in_features: 784,
        out_features: 128,
        activation: "relu".to_string(),
    },

    hardware: HardwareTarget {
        target_mode: "GPU_DISCRETE".to_string(),
        device_index: 0,
    },
};
```

## Configuration Flow

The structures can be viewed as a hierarchy:

```text
TrainConfig
├── epochs
├── batch_size
├── lr
├── layers
│   ├── in_features
│   ├── out_features
│   └── activation
│
└── hardware
    ├── target_mode
    └── device_index
```

This allows the training system to receive both **model architecture information** and **hardware execution preferences** through a single `TrainConfig` object.

## Serde Support

Because the structures implement `Serialize` and `Deserialize`, they can be converted to and from formats supported by Serde, such as JSON.

For example:

```rust
let config = TrainConfig {
    epochs: 50,
    batch_size: 64,
    lr: 0.001,

    layers: LayerConfig {
        in_features: 784,
        out_features: 256,
        activation: "relu".to_string(),
    },

    hardware: HardwareTarget {
        target_mode: "GPU_DISCRETE".to_string(),
        device_index: 0,
    },
};

let json = serde_json::to_string_pretty(&config)?;
println!("{}", json);
```

A serialized configuration could look like:

```json
{
  "epochs": 50,
  "batch_size": 64,
  "lr": 0.001,
  "layers": {
    "in_features": 784,
    "out_features": 256,
    "activation": "relu"
  },
  "hardware": {
    "target_mode": "GPU_DISCRETE",
    "device_index": 0
  }
}
```

## Dependencies

Add Serde and Serde JSON to `Cargo.toml`:

```toml
[dependencies]
serde = { version = "1", features = ["derive"] }
serde_json = "1"
```

## Module Organization

A typical project structure could be:

```text
src/
├── config/
│   ├── mod.rs
│   └── training.rs
│
├── training/
│   └── ...
│
├── model/
│   └── ...
│
└── main.rs
```

The configuration module can then be imported with:

```rust
use crate::config::{HardwareTarget, LayerConfig, TrainConfig};
```

> **Note:** If `TrainConfig` is intended to support multiple neural-network layers, `layers` would normally be better represented as `Vec<LayerConfig>` rather than a single `LayerConfig`.
>
> ```rust
> pub layers: Vec<LayerConfig>,
> ```
>
> This would allow configurations such as `784 → 256 → 128 → 10` instead of describing only one layer.
