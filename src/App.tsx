import { useState, type FormEvent } from "react";
import type { ExportFormat, HardwareTarget, LayerConfig } from "./types/type";
import { DataSection } from "./components/DataSection";
import { NetworkSection } from "./components/NetworkSection";
import {
  TrainingSection,
  type TrainingParams,
} from "./components/TrainingSection";
import { HardwareSection } from "./components/HardwareSection";
import { ExportSection } from "./components/ExportSection";
import { ProgressPanel } from "./components/ProgressPanel";
import { ModelTester } from "./components/ModelTester";
import { useTrainingSession } from "./hooks/useTrainingSession";
import { summarizeCsv } from "./utils/csv";

export default function App() {
  const [csv, setCsv] = useState("");
  const [layers, setLayers] = useState<LayerConfig>({
    in_features: 3,
    hidden_layers: [{ units: 16, activation: "relu" }],
    out_features: 1,
    output_activation: "none",
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
    const expectedColumns = layers.in_features + layers.out_features;
    if (summary.columns !== expectedColumns) {
      setFormError(
        `CSV has ${summary.columns} columns but the network expects ` +
          `${layers.in_features} features + ${layers.out_features} target(s) (${expectedColumns}).`,
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
    <main className="px-4 py-8 sm:py-12">
      <header className="mx-auto mb-8 w-1/3 min-w-[280px] max-w-full text-center">
        <h1 className="text-3xl font-bold tracking-tight text-primary sm:text-4xl">
          TensorLoom
        </h1>
        <p className="mt-2 text-sm text-secondary">
          Train and export neural networks locally.
        </p>
      </header>

      <form onSubmit={handleSubmit}>
        <DataSection
          csv={csv}
          onChange={setCsv}
          onFeatureCountSuggested={(n) =>
            setLayers((prev) => ({ ...prev, in_features: n }))
          }
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

        {formError && (
          <p
            role="alert"
            className="neo-alert mx-auto mt-6 w-1/3 min-w-[280px] max-w-full"
          >
            {formError}
          </p>
        )}

        <p className="mx-auto mt-6 w-1/3 min-w-[280px] max-w-full">
          <button type="submit" className="neo-btn w-full" disabled={session.running}>
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

      <ModelTester suggestedPath={outputPath} />
    </main>
  );
}
