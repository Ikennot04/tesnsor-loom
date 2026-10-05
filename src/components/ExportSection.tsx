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

export function ExportSection({ format, outputPath, onFormatChange, onOutputPathChange }: Props) {
  const [dialogError, setDialogError] = useState("");
  const { ext, label } = FORMATS[format];

  const handleFormatChange = (next: ExportFormat) => {
    onFormatChange(next);
    // Keep the chosen folder and name, just swap the extension.
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
      // `null` means the user cancelled.
      if (selected) onOutputPathChange(withExtension(selected, ext));
    } catch (err) {
      setDialogError(`Could not open the save dialog: ${String(err)}`);
    }
  };

  return (
    <fieldset>
      <legend>5. Export</legend>
      <p>
        <label>
          Format:{" "}
          <select value={format} onChange={(e) => handleFormatChange(e.target.value as ExportFormat)}>
            {Object.entries(FORMATS).map(([value, f]) => (
              <option key={value} value={value}>
                {f.label}
              </option>
            ))}
          </select>
        </label>
      </p>
      <p>
        <label>
          Save to:{" "}
          <input
            type="text"
            size={60}
            required
            value={outputPath}
            placeholder={`/Users/you/Desktop/model.${ext}`}
            onChange={(e) => onOutputPathChange(e.target.value)}
          />
        </label>{" "}
        <button type="button" onClick={handleBrowse}>
          Browse...
        </button>
      </p>
      {dialogError && <p role="alert">{dialogError}</p>}
    </fieldset>
  );
}