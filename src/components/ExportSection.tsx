import { useState } from "react";
import { save } from "@tauri-apps/plugin-dialog";
import type { ExportFormat } from "../types/type";

interface Props {
  format: ExportFormat;
  outputPath: string;
  onFormatChange: (format: ExportFormat) => void;
  onOutputPathChange: (path: string) => void;
}

const FORMATS: Record<ExportFormat, { label: string; ext: string }> = {
  json: { label: "JSON", ext: "json" },
  bin: { label: "Binary (bincode)", ext: "bin" },
  onnx: { label: "ONNX (placeholder only)", ext: "onnx" },
};

/** Replace (or add) the file extension, leaving the folder and base name alone. */
function withExtension(path: string, ext: string): string {
  if (!path) return path;
  const slash = Math.max(path.lastIndexOf("/"), path.lastIndexOf("\\"));
  const dot = path.lastIndexOf(".");
  const base = dot > slash ? path.slice(0, dot) : path;
  return `${base}.${ext}`;
}

export function ExportSection({
  format,
  outputPath,
  onFormatChange,
  onOutputPathChange,
}: Props) {
  const [dialogError, setDialogError] = useState("");
  const { ext, label } = FORMATS[format];

  const handleFormatChange = (next: ExportFormat) => {
    onFormatChange(next);
    if (outputPath) onOutputPathChange(withExtension(outputPath, FORMATS[next].ext));
  };

  const handleBrowse = async () => {
    setDialogError("");
    try {
      const selected = await save({
        title: "Save trained model as",
        defaultPath: outputPath || `model.${ext}`,
        filters: [{ name: label, extensions: [ext] }],
      });
      if (selected) onOutputPathChange(withExtension(selected, ext));
    } catch (err) {
      setDialogError(`Could not open the save dialog: ${String(err)}`);
    }
  };

  return (
    <section className="neo-raised mx-auto mt-6 w-1/3 min-w-[280px] max-w-full p-6 sm:p-8">
      <header className="mb-6">
        <h2 className="text-xl font-bold tracking-tight text-primary">5. Export</h2>
        <p className="mt-1 text-sm text-secondary">
          Choose format and destination for the trained model.
        </p>
      </header>

      <div className="space-y-5">
        <div className="space-y-2">
          <label
            htmlFor="export-format"
            className="block text-sm font-medium text-secondary"
          >
            Format
          </label>
          <select
            id="export-format"
            value={format}
            onChange={(e) => handleFormatChange(e.target.value as ExportFormat)}
            className="neo-field"
          >
            {Object.entries(FORMATS).map(([value, f]) => (
              <option key={value} value={value}>
                {f.label}
              </option>
            ))}
          </select>
        </div>

        <div className="space-y-2">
          <label
            htmlFor="output-path"
            className="block text-sm font-medium text-secondary"
          >
            Save to
          </label>
          <div className="neo-inset flex flex-col gap-3 px-4 py-4 sm:flex-row sm:items-center">
            <input
              id="output-path"
              type="text"
              required
              value={outputPath}
              placeholder={`/Users/you/Desktop/model.${ext}`}
              onChange={(e) => onOutputPathChange(e.target.value)}
              className="neo-field min-w-0 flex-1 border-0 shadow-none"
            />
            <button type="button" className="neo-btn shrink-0" onClick={handleBrowse}>
              Browse...
            </button>
          </div>
        </div>

        {dialogError && (
          <p role="alert" className="neo-alert">
            {dialogError}
          </p>
        )}
      </div>
    </section>
  );
}
