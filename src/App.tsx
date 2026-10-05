import { useState, type FormEvent } from "react";
import type { ExportFormat, HardwareTarget, LayerConfig } from "./types/type";
import { DataSection } from "./components/DataSection";
import { NetworkSection } from "./components/NetworkSection";
import { TrainingSection, type TrainingParams } from "./components/TrainingSection";
import { HardwareSection } from "./components/HardwareSection";
import { ExportSection } from "./components/ExportSection";
import { ProgressPanel } from "./components/ProgressPanel";
import { useTrainingSession } from "./hooks/useTrainingSession";
import { summarizeCsv } from "./utils/csv";

export default function App() {
  const [csv, setCsv] = useState("");
  const [layers, setLayers] = useState<LayerConfig>({
    in_features: 3,
    out_features: 16,
    activation: "relu",
  });
  const [params, setParams] = useState<TrainingParams>({
    epochs: 100,
    batch_size: 32,
    lr: 0.001,
  });
  const [hardware, setHardware] = useState<HardwareTarget>({
    target_mode: "CPU",
    device_index: 0,
  });
  const [exportFormat, setExportFormat] = useState<ExportFormat>("json");
  const [outputPath, setOutputPath] = useState("");
  const [formError, setFormError] = useState("");

  const session = useTrainingSession();

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (session.running) return;

    const summary = summarizeCsv(csv);
    if (summary.rows === 0) {
      setFormError("Please load or paste CSV data first.");
      return;
    }
    if (summary.columns !== layers.in_features + 1) {
      setFormError(
        `CSV has ${summary.columns} columns but the network expects ` +
          `${layers.in_features} features + 1 target (${layers.in_features + 1}).`,
      );
      return;
    }

    setFormError("");
    void session.start({
      config: { ...params, layers, hardware },
      data: csv.trim(),
      exportFormat,
      outputPath: outputPath.trim(),
    });
  };

  return (
    <main>
      <h1>TensorLoom</h1>

      <form onSubmit={handleSubmit}>
        <DataSection
          csv={csv}
          onChange={setCsv}
          onFeatureCountSuggested={(n) => setLayers((prev: any) => ({ ...prev, in_features: n }))}
        />
        <NetworkSection value={layers} onChange={setLayers} />
        <TrainingSection value={params} onChange={setParams} />
        <HardwareSection value={hardware} onChange={setHardware} />
        <ExportSection
          format={exportFormat}
          outputPath={outputPath}
          onFormatChange={setExportFormat}
          onOutputPathChange={setOutputPath}
        />

        {formError && <p role="alert">{formError}</p>}

        <p>
          <button type="submit" disabled={session.running}>
            {session.running ? "Training..." : "Start training"}
          </button>
        </p>
      </form>

      <ProgressPanel
        progress={session.progress}
        status={session.status}
        epochs={session.epochs}
        log={session.log}
        snapshot={session.snapshot}
      />
    </main>
  );
}