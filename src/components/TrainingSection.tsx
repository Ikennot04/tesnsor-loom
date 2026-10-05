import type { TrainConfig } from "../types/type";

export type TrainingParams = Pick<TrainConfig, "epochs" | "batch_size" | "lr">;

interface Props {
  value: TrainingParams;
  onChange: (value: TrainingParams) => void;
}

export function TrainingSection({ value, onChange }: Props) {
  return (
    <fieldset>
      <legend>3. Training</legend>
      <p>
        <label>
          Epochs:{" "}
          <input
            type="number"
            min={1}
            step={1}
            required
            value={value.epochs}
            onChange={(e) => onChange({ ...value, epochs: Number(e.target.value) })}
          />
        </label>
      </p>
      <p>
        <label>
          Batch size:{" "}
          <input
            type="number"
            min={1}
            step={1}
            required
            value={value.batch_size}
            onChange={(e) => onChange({ ...value, batch_size: Number(e.target.value) })}
          />
        </label>{" "}
        (not used by the core yet: it runs full-batch)
      </p>
      <p>
        <label>
          Learning rate:{" "}
          <input
            type="number"
            min={0}
            step="any"
            required
            value={value.lr}
            onChange={(e) => onChange({ ...value, lr: Number(e.target.value) })}
          />
        </label>
      </p>
    </fieldset>
  );
}