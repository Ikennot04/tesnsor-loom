import type { LayerConfig } from "../types/type";

interface Props {
  value: LayerConfig;
  onChange: (value: LayerConfig) => void;
}

export function NetworkSection({ value, onChange }: Props) {
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
      <p>
        <label>
          Hidden units:{" "}
          <input
            type="number"
            min={1}
            step={1}
            required
            value={value.out_features}
            onChange={(e) => onChange({ ...value, out_features: Number(e.target.value) })}
          />
        </label>
      </p>
      <p>
        <label>
          Activation:{" "}
          <select
            value={value.activation}
            onChange={(e) => onChange({ ...value, activation: e.target.value })}
          >
            <option value="relu">relu</option>
          </select>
        </label>{" "}
        (the core currently hardcodes relu)
      </p>
    </fieldset>
  );
}