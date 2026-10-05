# Hardware Detection and GPU Inventory

This module is responsible for detecting available hardware acceleration devices through the `burn_wgpu` backend.

It provides:

* GPU discovery
* Discrete GPU detection
* Integrated GPU detection
* Apple Silicon GPU identification
* CPU fallback support
* Safe device accessibility probing
* A serializable hardware inventory for use by the frontend

The module uses the `wgpu` backend through Burn to provide hardware abstraction across supported graphics APIs such as:

```text
Metal
Vulkan
DirectX 12
```

## Data Structures

```rust
use serde::{Deserialize, Serialize};
use burn_wgpu::WgpuDevice;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DetectedGpu {
    pub id: usize,
    pub name: String,
    pub device_type: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HardwareInventory {
    pub gpus: Vec<DetectedGpu>,
    pub fallback_cpu: bool,
}
```

---

# `DetectedGpu`

`DetectedGpu` represents a hardware accelerator detected by the system.

### Fields

| Field         | Type     | Description                             |
| ------------- | -------- | --------------------------------------- |
| `id`          | `usize`  | Identifier/index of the detected device |
| `name`        | `String` | Human-readable device name              |
| `device_type` | `String` | Hardware category                       |

Supported device types are:

```text
Discrete
Integrated
CPU
```

### Example

```rust
let gpu = DetectedGpu {
    id: 0,
    name: "Discrete GPU Acceleration Block #0".to_string(),
    device_type: "Discrete".to_string(),
};
```

An integrated GPU on Apple Silicon may be represented as:

```rust
let gpu = DetectedGpu {
    id: 0,
    name: "Apple Silicon Graphics Engine M-Series Core #0".to_string(),
    device_type: "Integrated".to_string(),
};
```

---

# `HardwareInventory`

`HardwareInventory` contains all hardware accelerators detected by the application.

```rust
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HardwareInventory {
    pub gpus: Vec<DetectedGpu>,
    pub fallback_cpu: bool,
}
```

### Fields

| Field          | Type               | Description                                                |
| -------------- | ------------------ | ---------------------------------------------------------- |
| `gpus`         | `Vec<DetectedGpu>` | List of detected GPU devices                               |
| `fallback_cpu` | `bool`             | Indicates whether CPU execution is available as a fallback |

### Example

```rust
let inventory = HardwareInventory {
    gpus: vec![
        DetectedGpu {
            id: 0,
            name: "Apple Silicon Graphics Engine M-Series Core #0".to_string(),
            device_type: "Integrated".to_string(),
        },
    ],
    fallback_cpu: true,
};
```

Conceptually:

```text
HardwareInventory
│
├── gpus
│   ├── GPU #0
│   │   ├── id
│   │   ├── name
│   │   └── device_type
│   │
│   └── GPU #1
│       ├── id
│       ├── name
│       └── device_type
│
└── fallback_cpu
```

---

# `detect_available_hardware`

```rust
pub fn detect_available_hardware() -> HardwareInventory
```

This function scans for available GPU acceleration devices and returns a `HardwareInventory`.

The implementation probes device indexes sequentially:

```rust
for index in 0..4 {
    // Probe hardware...
}
```

Currently, indexes `0` through `3` are tested.

The detection process checks for:

1. Discrete GPUs
2. Integrated GPUs
3. CPU fallback

---

## Hardware Detection Flow

```text
detect_available_hardware()
          │
          ▼
    Device Index 0
          │
          ├── Discrete GPU?
          │       │
          │      YES ──────► Add to inventory
          │
          ▼
    Integrated GPU?
          │
         YES
          │
          ▼
    Add to inventory
          │
          ▼
    Device Index 1
          │
         ...
          │
          ▼
    Device Index 3
          │
          ▼
    Enable CPU fallback
          │
          ▼
 HardwareInventory
```

---

# Discrete GPU Detection

The module first tests for discrete GPUs:

```rust
let discrete = WgpuDevice::DiscreteGpu(index);

if is_device_accessible(&discrete) {
    detected_gpus.push(DetectedGpu {
        id: index,
        name: format!(
            "Discrete GPU Acceleration Block #{}",
            index
        ),
        device_type: "Discrete".to_string(),
    });

    continue;
}
```

Discrete GPUs generally refer to dedicated graphics processors such as:

```text
NVIDIA RTX
AMD Radeon
Other dedicated GPUs
```

If a discrete GPU is successfully detected, it is added to the inventory and the integrated GPU probe for that index is skipped.

---

# Integrated GPU Detection

If no accessible discrete GPU is found for the current index, the module checks for an integrated GPU:

```rust
let integrated = WgpuDevice::IntegratedGpu(index);

if is_device_accessible(&integrated) {
    // Register integrated GPU
}
```

This is particularly relevant for systems using unified-memory architectures.

On macOS, the implementation provides an Apple Silicon-specific display name:

```rust
let label = if cfg!(target_os = "macos") {
    format!(
        "Apple Silicon Graphics Engine M-Series Core #{}",
        index
    )
} else {
    format!(
        "Integrated System GPU Core #{}",
        index
    )
};
```

Therefore, the same hardware inventory structure can be used across different operating systems.

---

# Platform Support

The hardware detection layer is designed around the graphics abstraction provided by `burn_wgpu`.

Conceptually:

```text
                 Leo Training Engine
                         │
                         ▼
                    burn_wgpu
                         │
              ┌──────────┼──────────┐
              │          │          │
            Metal     Vulkan     DX12
              │          │          │
              ▼          ▼          ▼
           macOS      Linux      Windows
```

This allows the application to use a common hardware abstraction instead of implementing completely separate GPU detection systems for each operating system.

---

# `is_device_accessible`

```rust
fn is_device_accessible(device: &WgpuDevice) -> bool
```

This internal helper determines whether a requested WGPU device can be initialized successfully.

```rust
fn is_device_accessible(device: &WgpuDevice) -> bool {
    std::panic::catch_unwind(|| {
        let _ = burn_wgpu::init_device(device);
    })
    .is_ok()
}
```

The probe is wrapped in:

```rust
std::panic::catch_unwind(...)
```

This prevents a failed device initialization from bringing down the entire application.

The function returns:

```text
true
```

when the device initialization succeeds without a panic.

Otherwise:

```text
false
```

is returned.

---

# Safe Hardware Probing

The probe can be represented as:

```text
                 WgpuDevice
                     │
                     ▼
            init_device(device)
                     │
              ┌──────┴──────┐
              │             │
          Successful      Panic
              │             │
              ▼             ▼
            true          false
```

This is useful because unavailable or improperly initialized graphics devices should not terminate the training application during hardware discovery.

---

# CPU Fallback

The returned inventory always enables CPU fallback:

```rust
HardwareInventory {
    gpus: detected_gpus,
    fallback_cpu: true,
}
```

This provides a fallback execution path when:

* No compatible GPU is available.
* GPU initialization fails.
* The user explicitly selects CPU execution.
* GPU acceleration is unavailable on the current platform.

Conceptually:

```text
             Hardware Selection
                     │
          ┌──────────┴──────────┐
          │                     │
      GPU Available        No GPU Available
          │                     │
          ▼                     ▼
    GPU Acceleration        CPU Fallback
```

---

# Example Usage

```rust
let inventory = detect_available_hardware();

for gpu in &inventory.gpus {
    println!(
        "{}: {} ({})",
        gpu.id,
        gpu.name,
        gpu.device_type
    );
}

if inventory.fallback_cpu {
    println!("CPU fallback is available");
}
```

Possible output:

```text
0: Apple Silicon Graphics Engine M-Series Core #0 (Integrated)
CPU fallback is available
```

Or on a system with a dedicated GPU:

```text
0: Discrete GPU Acceleration Block #0 (Discrete)
CPU fallback is available
```

---

# Relationship With Training Configuration

The hardware inventory can be used together with the training configuration to allow the user to select the execution device.

```text
HardwareInventory
       │
       │ detected devices
       ▼
   UI Hardware Selector
       │
       │ selected device
       ▼
HardwareTarget
       │
       ▼
   TrainConfig
       │
       ▼
 Training Engine
```

For example:

```rust
let hardware = HardwareTarget {
    target_mode: "GPU_DISCRETE".to_string(),
    device_index: 0,
};
```

The selected target can then be included in the training configuration.

---

# Serialization

Both `DetectedGpu` and `HardwareInventory` implement:

```rust
Serialize
Deserialize
Debug
Clone
```

This allows the hardware inventory to be:

* Sent to the frontend
* Stored in configuration
* Serialized to JSON
* Logged for debugging
* Cloned when passing configuration between components

Example JSON:

```json
{
  "gpus": [
    {
      "id": 0,
      "name": "Apple Silicon Graphics Engine M-Series Core #0",
      "device_type": "Integrated"
    }
  ],
  "fallback_cpu": true
}
```

---

# Module Responsibility

The hardware detection module should focus on **discovering and describing available hardware**.

It should not be responsible for:

* Training the neural network
* Selecting model architecture
* Managing epochs
* Computing loss
* Rendering the UI

Those responsibilities can remain in their respective modules.

```text
Hardware Module
      │
      ├── Detect GPUs
      ├── Validate devices
      ├── Build inventory
      └── Provide CPU fallback
              │
              ▼
        Training System
```

This separation keeps hardware discovery independent from the training and visualization layers.
