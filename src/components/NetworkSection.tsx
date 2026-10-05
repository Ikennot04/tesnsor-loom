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
    <fieldset>
      <legend>2. Network</legend>

      <p>
        <label>
          Input features:{" "}
          <input
            type="number"
            min={1}
            step={1}
            required
            value={value.in_features}
            onChange={(e) => onChange({ ...value, in_features: Number(e.target.value) })}
          />
        </label>{" "}
        (must equal the number of CSV feature columns)
      </p>

      <h4>Hidden layers ({value.hidden_layers.length})</h4>
      {value.hidden_layers.length === 0 && (
        <p>No hidden layers: the network is a single linear layer from input to output.</p>
      )}
      {value.hidden_layers.map((layer, i) => (
        <p key={i}>
          <strong>Hidden {i + 1}</strong>{" "}
          <label>
            Units:{" "}
            <input
              type="number"
              min={1}
              step={1}
              required
              value={layer.units}
              onChange={(e) => setHidden(i, { units: Number(e.target.value) })}
            />
          </label>{" "}
          <label>
            Activation:{" "}
            <select
              value={layer.activation}
              onChange={(e) => setHidden(i, { activation: e.target.value })}
            >
              {HIDDEN_ACTIVATIONS.map((a) => (
                <option key={a} value={a}>
                  {a}
                </option>
              ))}
            </select>
          </label>{" "}
          <button type="button" onClick={() => removeHidden(i)}>
            Remove
          </button>
        </p>
      ))}
      <p>
        <button
          type="button"
          onClick={addHidden}
          disabled={value.hidden_layers.length >= MAX_HIDDEN_LAYERS}
        >
          + Add hidden layer
        </button>
      </p>

      <h4>Output layer</h4>
      <p>
        <label>
          Output units:{" "}
          <input
            type="number"
            min={1}
            step={1}
            required
            value={value.out_features}
            onChange={(e) => onChange({ ...value, out_features: Number(e.target.value) })}
          />
        </label>{" "}
        (the last this-many CSV columns are the targets)
      </p>
      <p>
        <label>
          Output activation:{" "}
          <select
            value={value.output_activation}
            onChange={(e) => onChange({ ...value, output_activation: e.target.value })}
          >
            {OUTPUT_ACTIVATIONS.map((a) => (
              <option key={a} value={a}>
                {a}
              </option>
            ))}
          </select>
        </label>{" "}
        (use none for regression)
      </p>
    </fieldset>
  );
}