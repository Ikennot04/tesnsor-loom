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
    <fieldset>
      <legend>4. Hardware</legend>
      <p>
        <button type="button" onClick={handleDetect}>
          Detect hardware
        </button>{" "}
        <span>{detectStatus}</span>
      </p>
      <p>
        <label>
          Target:{" "}
          <select
            value={value.target_mode}
            onChange={(e) => onChange({ ...value, target_mode: e.target.value as TargetMode })}
          >
            <option value="CPU">CPU</option>
            <option value="GPU_DISCRETE">Discrete GPU</option>
            <option value="GPU_INTEGRATED">Integrated GPU</option>
          </select>
        </label>
      </p>
      <p>
        <label>
          Device index:{" "}
          <input
            type="number"
            min={0}
            step={1}
            value={value.device_index}
            onChange={(e) => onChange({ ...value, device_index: Number(e.target.value) })}
          />
        </label>
      </p>
      <ul>
        {gpus.map((gpu) => (
          <li key={`${gpu.device_type}-${gpu.id}`}>
            #{gpu.id} {gpu.name} ({gpu.device_type}){" "}
            <button type="button" onClick={() => useGpu(gpu)}>
              Use
            </button>
          </li>
        ))}
      </ul>
    </fieldset>
  );
}