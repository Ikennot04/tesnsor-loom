import { useState } from "react";
import { detectHardware } from "../api/hardware";
import type { DetectedGpu, HardwareTarget, TargetMode } from "../types/type";

interface Props {
  value: HardwareTarget;
  onChange: (value: HardwareTarget) => void;
}

export function HardwareSection({ value, onChange }: Props) {
  const [gpus, setGpus] = useState<DetectedGpu[]>([]);
  const [detectStatus, setDetectStatus] = useState("");

  const handleDetect = async () => {
    setDetectStatus("Detecting...");
    setGpus([]);
    try {
      const inventory = await detectHardware();
      setGpus(inventory.gpus);
      setDetectStatus(`Found ${inventory.gpus.length} GPU(s).`);
    } catch (err) {
      setDetectStatus(`Detection unavailable: ${String(err)}`);
    }
  };

  const useGpu = (gpu: DetectedGpu) =>
    onChange({
      target_mode: gpu.device_type === "Discrete" ? "GPU_DISCRETE" : "GPU_INTEGRATED",
      device_index: gpu.id,
    });

  return (
    <section className="neo-raised h-full w-full p-6 sm:p-8">
      <header className="mb-6">
        <h2 className="text-xl font-bold tracking-tight text-primary">4. Hardware</h2>
        <p className="mt-1 text-sm text-secondary">
          Detect GPUs and choose where training runs.
        </p>
      </header>

      <div className="neo-inset flex flex-wrap items-center justify-between gap-3 px-4 py-4">
        <span className="text-sm font-medium text-secondary">Hardware scan</span>
        <button type="button" className="neo-btn" onClick={handleDetect}>
          Detect hardware
        </button>
      </div>

      {detectStatus && (
        <div className="mt-3">
          <span className="neo-chip">{detectStatus}</span>
        </div>
      )}

      <div className="mt-6 space-y-5">
        <div className="space-y-2">
          <label
            htmlFor="target-mode"
            className="block text-sm font-medium text-secondary"
          >
            Target
          </label>
          <select
            id="target-mode"
            value={value.target_mode}
            onChange={(e) =>
              onChange({ ...value, target_mode: e.target.value as TargetMode })
            }
            className="neo-field"
          >
            <option value="CPU">CPU</option>
            <option value="GPU_DISCRETE">Discrete GPU</option>
            <option value="GPU_INTEGRATED">Integrated GPU</option>
          </select>
        </div>

        <div className="space-y-2">
          <label
            htmlFor="device-index"
            className="block text-sm font-medium text-secondary"
          >
            Device index
          </label>
          <input
            id="device-index"
            type="number"
            min={0}
            step={1}
            value={value.device_index}
            onChange={(e) =>
              onChange({ ...value, device_index: Number(e.target.value) })
            }
            className="neo-field"
          />
        </div>
      </div>

      {gpus.length > 0 && (
        <div className="mt-6">
          <h3 className="mb-3 text-sm font-semibold text-primary">
            Detected GPUs
          </h3>
          <ul className="space-y-3">
            {gpus.map((gpu) => (
              <li
                key={`${gpu.device_type}-${gpu.id}`}
                className="neo-inset flex flex-wrap items-center justify-between gap-3 px-4 py-3"
              >
                <span className="text-sm text-secondary">
                  #{gpu.id} {gpu.name} ({gpu.device_type})
                </span>
                <button
                  type="button"
                  className="neo-btn-ghost"
                  onClick={() => useGpu(gpu)}
                >
                  Use
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}
