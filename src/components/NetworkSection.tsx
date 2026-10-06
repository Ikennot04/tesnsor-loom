import type { HiddenLayerConfig, LayerConfig } from "../types/type";

interface Props {
  value: LayerConfig;
  onChange: (value: LayerConfig) => void;
}

const MAX_HIDDEN_LAYERS = 8;
const HIDDEN_ACTIVATIONS = ["relu", "tanh", "sigmoid"];
const OUTPUT_ACTIVATIONS = ["none", "relu", "tanh", "sigmoid"];

export function NetworkSection({ value, onChange }: Props) {
  const setHidden = (index: number, patch: Partial<HiddenLayerConfig>) => {
    const hidden_layers = value.hidden_layers.map((layer, i) =>
      i === index ? { ...layer, ...patch } : layer,
    );
    onChange({ ...value, hidden_layers });
  };

  const addHidden = () => {
    if (value.hidden_layers.length >= MAX_HIDDEN_LAYERS) return;
    const last = value.hidden_layers[value.hidden_layers.length - 1];
    onChange({
      ...value,
      hidden_layers: [
        ...value.hidden_layers,
        { units: last?.units ?? 16, activation: last?.activation ?? "relu" },
      ],
    });
  };

  const removeHidden = (index: number) => {
    onChange({
      ...value,
      hidden_layers: value.hidden_layers.filter((_, i) => i !== index),
    });
  };

  return (
    <section className="neo-raised mx-auto mt-6 w-1/3 min-w-[280px] max-w-full p-6 sm:p-8">
      <header className="mb-6">
        <h2 className="text-xl font-bold tracking-tight text-primary">2. Network</h2>
        <p className="mt-1 text-sm text-secondary">
          Configure input, hidden, and output layers.
        </p>
      </header>

      <div className="space-y-2">
        <label
          htmlFor="in-features"
          className="block text-sm font-medium text-secondary"
        >
          Input features
        </label>
        <input
          id="in-features"
          type="number"
          min={1}
          step={1}
          required
          value={value.in_features}
          onChange={(e) =>
            onChange({ ...value, in_features: Number(e.target.value) })
          }
          className="neo-field"
        />
        <p className="text-xs text-secondary">
          Must equal the number of CSV feature columns.
        </p>
      </div>

      <div className="mt-6">
        <div className="mb-3 flex flex-wrap items-center gap-2">
          <h3 className="text-sm font-semibold text-primary">Hidden layers</h3>
          <span className="neo-chip">{value.hidden_layers.length}</span>
        </div>

        {value.hidden_layers.length === 0 && (
          <p className="mb-3 text-sm text-secondary">
            No hidden layers: the network is a single linear layer from input to
            output.
          </p>
        )}

        <div className="space-y-3">
          {value.hidden_layers.map((layer, i) => (
            <div
              key={i}
              className="neo-inset space-y-3 px-4 py-4"
            >
              <p className="text-sm font-semibold text-primary">
                Hidden {i + 1}
              </p>
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                <div className="space-y-1.5">
                  <label
                    htmlFor={`hidden-units-${i}`}
                    className="block text-xs font-medium text-secondary"
                  >
                    Units
                  </label>
                  <input
                    id={`hidden-units-${i}`}
                    type="number"
                    min={1}
                    step={1}
                    required
                    value={layer.units}
                    onChange={(e) =>
                      setHidden(i, { units: Number(e.target.value) })
                    }
                    className="neo-field"
                  />
                </div>
                <div className="space-y-1.5">
                  <label
                    htmlFor={`hidden-act-${i}`}
                    className="block text-xs font-medium text-secondary"
                  >
                    Activation
                  </label>
                  <select
                    id={`hidden-act-${i}`}
                    value={layer.activation}
                    onChange={(e) =>
                      setHidden(i, { activation: e.target.value })
                    }
                    className="neo-field"
                  >
                    {HIDDEN_ACTIVATIONS.map((a) => (
                      <option key={a} value={a}>
                        {a}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
              <button
                type="button"
                className="neo-btn-ghost"
                onClick={() => removeHidden(i)}
              >
                Remove
              </button>
            </div>
          ))}
        </div>

        <div className="mt-4">
          <button
            type="button"
            className="neo-btn"
            onClick={addHidden}
            disabled={value.hidden_layers.length >= MAX_HIDDEN_LAYERS}
          >
            + Add hidden layer
          </button>
        </div>
      </div>

      <div className="mt-8 space-y-4">
        <h3 className="text-sm font-semibold text-primary">Output layer</h3>

        <div className="space-y-2">
          <label
            htmlFor="out-features"
            className="block text-sm font-medium text-secondary"
          >
            Output units
          </label>
          <input
            id="out-features"
            type="number"
            min={1}
            step={1}
            required
            value={value.out_features}
            onChange={(e) =>
              onChange({ ...value, out_features: Number(e.target.value) })
            }
            className="neo-field"
          />
          <p className="text-xs text-secondary">
            The last this-many CSV columns are the targets.
          </p>
        </div>

        <div className="space-y-2">
          <label
            htmlFor="out-activation"
            className="block text-sm font-medium text-secondary"
          >
            Output activation
          </label>
          <select
            id="out-activation"
            value={value.output_activation}
            onChange={(e) =>
              onChange({ ...value, output_activation: e.target.value })
            }
            className="neo-field"
          >
            {OUTPUT_ACTIVATIONS.map((a) => (
              <option key={a} value={a}>
                {a}
              </option>
            ))}
          </select>
          <p className="text-xs text-secondary">Use none for regression.</p>
        </div>
      </div>
    </section>
  );
}
