# TensorLoom

A desktop app for training small neural networks on your own CSV data, watching the training happen live, and testing the exported model.

- **Frontend:** React + TypeScript (Vite), in `src/`
- **API layer:** Tauri 2 commands and events, in `src-tauri/src/lib.rs`
- **Training engine:** Rust + [Burn](https://burn.dev), in `crates/`

You load a CSV, pick a network size and hardware target, and press **Start training**. The app streams loss, score and a drawing of the network's weights while it trains. When it finishes, the model is saved to a file you choose. You can then load that file in the app to predict single rows or score a whole CSV.

---

## Project structure

```
tensorloom/
├── src/                        Frontend (React + TypeScript)
│   ├── api/                    Calls into Rust (invoke / listen). No UI code.
│   │   ├── types.ts            TS types that mirror the Rust structs and events
│   │   ├── training.ts         start_training_session + event subscriptions
│   │   ├── hardware.ts         detect_hardware
│   │   ├── model.ts            load_model, predict_rows, evaluate_model, file picker
│   │   └── index.ts
│   ├── components/             UI only
│   │   ├── DataSection.tsx
│   │   ├── NetworkSection.tsx
│   │   ├── TrainingSection.tsx
│   │   ├── HardwareSection.tsx
│   │   ├── ExportSection.tsx
│   │   ├── ProgressPanel.tsx
│   │   ├── NetworkVisualizer.tsx   Live network drawing
│   │   ├── LossChart.tsx
│   │   └── ModelTester.tsx         Load a model, predict, score
│   ├── hooks/useTrainingSession.ts State + event handling for a run
│   ├── utils/csv.ts
│   └── App.tsx
│
├── src-tauri/                  Tauri app (the API between UI and engine)
│   ├── src/
│   │   ├── lib.rs              Tauri commands and the event bridge
│   │   └── model_commands.rs   Model loading / prediction commands
│   ├── capabilities/           Permissions the frontend is allowed to use
│   ├── tauri.conf.json
│   └── Cargo.toml
│
├── crates/
│   ├── tensorloom-core/        The AI training engine (library)
│   │   └── src/
│   │       ├── config/         TrainConfig, LayerConfig, HardwareTarget
│   │       ├── data/           Data handling
│   │       ├── events/         TrainEvent (what the engine reports)
│   │       ├── hardware/       GPU / CPU detection
│   │       ├── model/          PortableModel + inference (predict, evaluate)
│   │       ├── train/          The Burn training loop
│   │       └── lib.rs
│   └── server/                 Separate crate (describe its role here)
│
├── Cargo.toml                  Workspace manifest
├── package.json
└── vite.config.ts
```

---

## How it fits together

```
 React UI  ──invoke──▶  src-tauri/lib.rs  ──calls──▶  tensorloom-core
    ▲                         │                            │
    │                         │                       TrainEvent
    └────── emit events ◀─────┴────── mpsc channel ◀───────┘
```

1. The UI calls the Tauri command `start_training_session` with the config, the CSV text, an export format and an output path.
2. `lib.rs` starts a background thread, so the window stays responsive, and runs `tensorloom_core::train::execute_training`.
3. The engine sends `TrainEvent`s through a channel. A second thread forwards each one to the UI as a Tauri event named `training-progress`.
4. When training ends, `lib.rs` exports the model to the chosen file and emits `training-complete` (or `training-error`).

### Tauri commands (`src-tauri/src/lib.rs`)

| Command | Purpose |
| --- | --- |
| `start_training_session` | Train a model in the background and export it |
| `detect_hardware` | List the GPUs the graphics stack can see (optional; register it if you want the "Detect hardware" button) |
| `load_model` | Read and validate an exported `.json` / `.bin` model |
| `predict_rows` | Run rows through a saved model |
| `evaluate_model` | Score a saved model on a CSV (MSE and R²) |

### Events sent to the frontend

| Event | Payload |
| --- | --- |
| `training-progress` | A `TrainEvent` (see below) |
| `training-complete` | The output path |
| `training-error` | An error message |

`TrainEvent` variants: `EpochStarted`, `BatchCompleted`, `EpochCompleted`, `TrainingFinished`, `Network` (a downsampled snapshot of weights and node activity for the live drawing) and `Log`.

**Adding a new backend update:**
1. Add a variant to `TrainEvent` in `crates/tensorloom-core/src/events/`.
2. Send it from the training loop: `tx.send(TrainEvent::MyEvent { ... })`.
3. Add the matching type to `src/api/types.ts`.
4. Add a `case` for it in `src/hooks/useTrainingSession.ts`.

`lib.rs` already forwards every `TrainEvent`, so it needs no change.

---

## Setting up the environment

### 1. Install the prerequisites

**All platforms**

- **Rust** (1.85 or newer, because the crates use the 2024 edition). Install with [rustup](https://rustup.rs):
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  rustup update
  rustc --version
  ```
- **Node.js** 18 or newer, from [nodejs.org](https://nodejs.org) or a version manager such as `nvm`:
  ```bash
  node --version
  npm --version
  ```

**macOS**

```bash
xcode-select --install
```

This installs the Command Line Tools that Tauri and the Rust linker need. Metal, used for GPU training, is built into macOS.

**Windows**

- Install the [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and select **Desktop development with C++**.
- Install the [WebView2 runtime](https://developer.microsoft.com/microsoft-edge/webview2/) if it isn't already on your machine (it ships with Windows 11).

**Linux (Debian / Ubuntu)**

```bash
sudo apt update
sudo apt install libwebkit2gtk-4.1-dev build-essential curl wget file \
  libxdo-dev libssl-dev libayatana-appindicator3-dev librsvg2-dev
```

Other distributions: see the [Tauri prerequisites guide](https://tauri.app/start/prerequisites/). GPU training needs working Vulkan drivers.

### 2. Install the project dependencies

```bash
git clone <your-repo-url> tensorloom
cd tensorloom
npm install
```

The first Rust build downloads and compiles Burn and its dependencies, so expect it to take several minutes.

### 3. Required Tauri plugins

The app uses the **dialog** plugin for the Save and Open file pickers, and the **shell** plugin from the template.

```bash
npm run tauri add dialog
```

This installs `@tauri-apps/plugin-dialog`, adds `tauri-plugin-dialog` to `src-tauri/Cargo.toml` and registers it. If you set it up by hand, check all three of these:

1. `npm i @tauri-apps/plugin-dialog`
2. In `src-tauri`: `cargo add tauri-plugin-dialog`, and add `.plugin(tauri_plugin_dialog::init())` to the builder in `lib.rs`
3. In `src-tauri/capabilities/default.json`:
   ```json
   "permissions": ["core:default", "dialog:default"]
   ```

`src-tauri/Cargo.toml` also needs `serde`, `serde_json` and `bincode` (for exporting and loading models), plus the `tensorloom-core` path dependency.

### 4. Engine dependencies (`crates/tensorloom-core/Cargo.toml`)

```toml
[dependencies]
sysinfo = "0.39.6"
serde = { version = "1", features = ["derive"] }
burn = { version = "0.21.0", features = ["wgpu", "ndarray", "autodiff", "train"] }
thiserror = "2.0.21"
```

Do not add `burn-wgpu` separately. The `wgpu` feature of `burn` already includes it, and a second copy can cause version-mismatch type errors. Import the device type as `burn::backend::wgpu::WgpuDevice`.

---

## Running the app

```bash
npm run tauri dev
```

This starts Vite and opens the **desktop window**. Always use that window. Opening `http://localhost:1420` in Chrome or Safari shows the same page without the Tauri bridge, and every call to Rust fails with `Cannot read properties of undefined (reading 'invoke')`.

Changes to the frontend hot-reload. Changes to Rust code rebuild and restart the app.

### Build a release app

```bash
npm run tauri build
```

The installer or app bundle ends up in `src-tauri/target/release/bundle/`.

### Run only the engine tests / checks

```bash
cargo check -p tensorloom-core
cargo test  -p tensorloom-core
```

---

## Using the app

1. **Data:** load a CSV file or paste CSV text.
   - Each row is `feature_1,...,feature_n,target`, so the **last column is the target**.
   - A header row is detected and skipped.
   - All values must be numbers. The model has one output (regression).
2. **Network:** set **Input features** to the number of feature columns (it fills in when you load a file) and choose the number of hidden units.
3. **Training:** set epochs and learning rate.
4. **Hardware:** choose CPU, a discrete GPU or an integrated GPU, and the device index. Use **Detect hardware** to list GPUs. Start with CPU to confirm everything works. On Apple Silicon use the integrated GPU.
5. **Export:** choose a format, then **Browse...** to pick where the model is saved.
6. Press **Start training**. Watch the progress bar, the live network drawing, the loss chart and the epoch table.
7. **Test a trained model:** use **Open model...** to load a saved model, enter feature values to get a prediction, or score it on a CSV that includes the target column.

### Reading the results

- **Score (R²):** 1.0 is a perfect fit, 0 means no better than always guessing the average, and a negative value is worse than that. A negative score usually means the model needs a higher learning rate or more epochs.
- **Live network drawing:** edge thickness is the weight size, blue is a positive weight and red a negative one, and node brightness is the mean activation. Layers wider than 12 nodes are sampled and labeled "showing 12 of N".

### Tuning tips

| Goal | Try |
| --- | --- |
| Loss barely moves | Raise the learning rate (for example `0.01` to `0.05`) or add epochs |
| Loss is unstable or goes to NaN | Lower the learning rate |
| Run is too short or too long | Rescale epochs. Total time is epochs times time per epoch |
| UI feels slow | Use a smaller CSV (the whole file is sent to Rust as one string) |

---

## Model files

| Format | Notes |
| --- | --- |
| **JSON** | Human-readable. Layers of `weights`, `bias`, `shape` and `activation` |
| **Binary (bincode)** | Much smaller, same content |
| **ONNX** | Placeholder only: it writes a few bytes and is not a usable model |

For a layer with `shape: [n_in, n_out]`, `weights` is row-major with `n_in * n_out` values, and each output is
`activation( Σ input[i] · weights[i * n_out + j] + bias[j] )`.

The current engine trains a fixed two-layer network (input, relu hidden layer, one linear output).

---

## Known limitations

- Training is full-batch: one gradient step per epoch. The `batch_size` setting is sent but not used yet.
- The activation is fixed to `relu`. The activation dropdown is for future use.
- Scoring after training uses the training data, because there is no validation split yet. Test on a CSV the model has not seen.
- ONNX export is a placeholder.
- Only single-output regression is supported.

---

## Troubleshooting

| Problem | Cause and fix |
| --- | --- |
| `Cannot read properties of undefined (reading 'invoke')` or `'transformCallback'` | The page is running in a normal browser tab. Use the window opened by `npm run tauri dev`. In its dev tools, `window.__TAURI_INTERNALS__` should be an object |
| `dialog.save not allowed. Plugin not found` | The dialog plugin isn't registered in Rust. See "Required Tauri plugins" and restart `tauri dev` |
| `Cannot find module '@tauri-apps/plugin-dialog'` | Run `npm i @tauri-apps/plugin-dialog`, then restart the TypeScript server in your editor |
| `dialog.save not allowed` (without "Plugin not found") | Add `"dialog:default"` to `src-tauri/capabilities/default.json` |
| CSV error like `expected 4 columns` | **Input features** doesn't match the CSV. It should be the number of columns minus one |
| GPU run fails or panics | The machine may have no GPU of that type. Switch the target to CPU or the other GPU type |
| Rust change has no effect | Rust code needs a rebuild. Stop and restart `npm run tauri dev` |

---

## License

See [LICENSE](./LICENSE).