# Portable Model Format

This module defines a lightweight and serializable representation of a trained neural network.

The portable model format is designed to store the essential information required to reconstruct a neural network without depending directly on a specific training backend or hardware accelerator.

It contains:

* Layer weights
* Layer biases
* Layer dimensions
* Activation functions
* Model metadata

```rust
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
```

# `PrimitiveLayer`

`PrimitiveLayer` represents one neural network layer using primitive numerical data.

Unlike a backend-specific neural network layer, this structure contains only the fundamental information required to describe the layer.

### Fields

| Field        | Type             | Description                            |
| ------------ | ---------------- | -------------------------------------- |
| `weights`    | `Vec<f32>`       | Flattened neural network weight values |
| `bias`       | `Vec<f32>`       | Bias value for each output node        |
| `shape`      | `(usize, usize)` | Matrix dimensions of the layer         |
| `activation` | `String`         | Activation function used by the layer  |

---

## Weights

The `weights` field contains the numerical parameters connecting the input nodes to the output nodes.

For a layer with:

```text
Input  = 3
Output = 4
```

the weight matrix has:

```text
3 × 4 = 12
```

weight values.

The values are stored in a flat vector:

```rust
let weights = vec![
    0.12, -0.31, 0.45, 0.08,
    0.21,  0.17, -0.22, 0.39,
    0.05,  0.28, 0.11, -0.14,
];
```

The expected number of weights is:

```text
weights.len() = shape.0 × shape.1
```

---

# Bias

The `bias` vector contains the bias value associated with each output node.

For example, a layer with four output nodes would contain four bias values:

```rust
let bias = vec![
    0.10,
    -0.05,
    0.23,
    0.01,
];
```

Therefore:

```text
bias.len() = number of output nodes
```

---

# Shape

The `shape` field describes the dimensions of the weight matrix:

```rust
pub shape: (usize, usize)
```

For example:

```rust
shape: (3, 4)
```

represents:

```text
3 input nodes
      ↓
4 output nodes
```

Conceptually:

```text
        Output
        0  1  2  3
      ┌─────────────
Input 0│ W W W W
      1│ W W W W
      2│ W W W W
```

The shape therefore determines how the flattened `weights` vector should be interpreted.

---

# Activation

The `activation` field identifies the activation function applied by the layer.

Example:

```rust
activation: "relu".to_string()
```

Common activation values include:

```text
relu
sigmoid
tanh
softmax
linear
```

The actual activation functions supported by the training/inference engine should be documented and validated by the corresponding model implementation.

---

# Example `PrimitiveLayer`

```rust
let layer = PrimitiveLayer {
    weights: vec![
        0.12, -0.31, 0.45, 0.08,
        0.21,  0.17, -0.22, 0.39,
        0.05,  0.28, 0.11, -0.14,
    ],

    bias: vec![
        0.10,
        -0.05,
        0.23,
        0.01,
    ],

    shape: (3, 4),

    activation: "relu".to_string(),
};
```

This describes a layer with:

```text
3 input features
      │
      ▼
┌─────────────┐
│ 4 neurons   │
│ ReLU        │
└─────────────┘
```

---

# `PortableModel`

`PortableModel` represents an entire neural network composed of multiple `PrimitiveLayer` structures.

```rust
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PortableModel {
    pub layers: Vec<PrimitiveLayer>,
    pub metadata: String,
}
```

### Fields

| Field      | Type                  | Description                                 |
| ---------- | --------------------- | ------------------------------------------- |
| `layers`   | `Vec<PrimitiveLayer>` | Ordered collection of neural network layers |
| `metadata` | `String`              | Additional model information                |

The layers are stored in their execution order.

For example:

```text
Input
  │
  ▼
Layer 0
  │
  ▼
Layer 1
  │
  ▼
Layer 2
  │
  ▼
Output
```

---

# Example Portable Model

A simple neural network can be represented as:

```rust
let model = PortableModel {
    layers: vec![
        PrimitiveLayer {
            weights: vec![/* layer 1 weights */],
            bias: vec![/* layer 1 bias */],
            shape: (3, 16),
            activation: "relu".to_string(),
        },

        PrimitiveLayer {
            weights: vec![/* layer 2 weights */],
            bias: vec![/* layer 2 bias */],
            shape: (16, 1),
            activation: "sigmoid".to_string(),
        },
    ],

    metadata: "Example binary classification model".to_string(),
};
```

The resulting architecture is:

```text
Input
  │
  │ 3 features
  ▼
┌──────────────┐
│ 16 neurons   │
│ ReLU         │
└──────────────┘
  │
  │ 16 features
  ▼
┌──────────────┐
│ 1 neuron     │
│ Sigmoid      │
└──────────────┘
  │
  ▼
Output
```

---

# Model Structure

The relationship between the structures is:

```text
PortableModel
│
├── metadata
│
└── layers
    │
    ├── PrimitiveLayer
    │   ├── weights
    │   ├── bias
    │   ├── shape
    │   └── activation
    │
    ├── PrimitiveLayer
    │   ├── weights
    │   ├── bias
    │   ├── shape
    │   └── activation
    │
    └── ...
```

---

# Serialization

Both structures implement Serde:

```rust
Serialize
Deserialize
Debug
Clone
```

This allows the model to be serialized into formats such as JSON or binary formats supported by the project.

For example:

```rust
let json = serde_json::to_string_pretty(&model)?;
println!("{}", json);
```

A serialized model may look like:

```json
{
  "layers": [
    {
      "weights": [
        0.12,
        -0.31,
        0.45
      ],
      "bias": [
        0.10
      ],
      "shape": [3, 1],
      "activation": "sigmoid"
    }
  ],
  "metadata": "Example model"
}
```

---

# Backend Independence

The primary purpose of `PortableModel` is to provide a representation that is not tightly coupled to a particular neural-network backend.

Conceptually:

```text
                 Training Backend
                       │
                       ▼
                Trained Network
                       │
                       ▼
               PortableModel
                       │
          ┌────────────┼────────────┐
          │            │            │
          ▼            ▼            ▼
       Storage      Inference       UI
          │            │            │
          ▼            ▼            ▼
       File/DB      Model Engine   Visualization
```

This makes it possible to transfer the learned parameters between different parts of the application.

---

# Model Validation

When loading a `PortableModel`, the implementation should verify that the stored dimensions are internally consistent.

For every layer:

```text
weights.len() == shape.0 × shape.1
```

and:

```text
bias.len() == shape.1
```

assuming `shape` represents:

```text
(input_features, output_features)
```

For example:

```text
shape = (16, 4)

Expected weights = 16 × 4 = 64
Expected biases  = 4
```

This prevents malformed model data from being used during inference.

---

# Model Portability Flow

A typical model lifecycle is:

```text
Training
   │
   ▼
Neural Network
   │
   ├── Weights
   ├── Biases
   ├── Shapes
   └── Activations
   │
   ▼
PortableModel
   │
   ▼
Serialization
   │
   ▼
Model File / Storage
   │
   ▼
Deserialization
   │
   ▼
PortableModel
   │
   ▼
Inference / Visualization
```

The portable representation therefore acts as a boundary between the **training implementation** and the **model storage/inference layer**.
