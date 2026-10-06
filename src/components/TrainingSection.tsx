import type { TrainConfig } from "../types/type";

export type TrainingParams = Pick<TrainConfig, "epochs" | "batch_size" | "lr">;

interface Props {
  value: TrainingParams;
  onChange: (value: TrainingParams) => void;
}

export function TrainingSection({ value, onChange }: Props) {
  return (
    <section className="neo-raised h-full w-full p-6 sm:p-8">
      <header className="mb-6">
        <h2 className="text-xl font-bold tracking-tight text-primary">
          3. Training
        </h2>
        <p className="mt-1 text-sm text-secondary">
          Set epochs, batch size, and learning rate.
        </p>
      </header>

      <div className="space-y-5">
        <div className="space-y-2">
          <label
            htmlFor="epochs"
            className="block text-sm font-medium text-secondary"
          >
            Epochs
          </label>
          <input
            id="epochs"
            type="number"
            min={1}
            step={1}
            required
            value={value.epochs}
            onChange={(e) =>
              onChange({ ...value, epochs: Number(e.target.value) })
            }
            className="neo-field"
          />
        </div>

        <div className="space-y-2">
          <label
            htmlFor="batch-size"
            className="block text-sm font-medium text-secondary"
          >
            Batch size
          </label>
          <input
            id="batch-size"
            type="number"
            min={1}
            step={1}
            required
            value={value.batch_size}
            onChange={(e) =>
              onChange({ ...value, batch_size: Number(e.target.value) })
            }
            className="neo-field"
          />
          <p className="text-xs text-secondary">
            Not used by the core yet: it runs full-batch.
          </p>
        </div>

        <div className="space-y-2">
          <label
            htmlFor="learning-rate"
            className="block text-sm font-medium text-secondary"
          >
            Learning rate
          </label>
          <input
            id="learning-rate"
            type="number"
            min={0}
            step="any"
            required
            value={value.lr}
            onChange={(e) => onChange({ ...value, lr: Number(e.target.value) })}
            className="neo-field"
          />
        </div>
      </div>
    </section>
  );
}
