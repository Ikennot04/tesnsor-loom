import { useRef, useState, type ChangeEvent } from "react";
import { generateDataset, validateEnvironment, SUDOKU_ENVIRONMENT } from "../utils/environment";

interface Props {
  json: string;
  onChange: (json: string) => void;
  onDatasetGenerated: (csv: string, inFeatures: number, outFeatures: number) => void;
}

function parseError(json: string): string | null {
  if (!json.trim()) return null;
  try { JSON.parse(json); return null; } catch (e) { return String(e); }
}

export function EnvironmentSection({ json, onChange, onDatasetGenerated }: Props) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [rows, setRows] = useState(10000);
  const [status, setStatus] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const syntaxError = parseError(json);

  const handleFile = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) onChange(await file.text());
    e.target.value = "";
  };

  const check = async () => {
    try {
      const [i, o] = await validateEnvironment(json);
      setStatus(`Valid: ${i} inputs, ${o} outputs.`);
    } catch (err) { setStatus(String(err)); }
  };

  const generate = async () => {
    setBusy(true);
    try {
      const g = await generateDataset(json, rows, Date.now() % 1_000_000);
      onDatasetGenerated(g.csv, g.in_features, g.out_features);
      setStatus(`Generated ${rows} rows.`);
    } catch (err) { setStatus(String(err)); }
    finally { setBusy(false); }
  };

  const download = () => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([json], { type: "application/json" }));
    a.download = "rules.json";
    a.click();
  };

  return (
    <section className="neo-raised h-full w-full p-6 sm:p-8">
      <header className="mb-6">
        <h2 className="text-xl font-bold tracking-tight text-primary">Environment</h2>
        <p className="mt-1 text-sm text-secondary">
          Optional. Rules for the task (sudoku, chess...). Upload or edit the JSON, then generate matching training data.
        </p>
      </header>

      <div className="neo-inset flex flex-wrap items-center gap-3 px-4 py-4">
        <input ref={fileRef} type="file" accept=".json,application/json" className="sr-only" onChange={handleFile} />
        <button type="button" className="neo-btn" onClick={() => fileRef.current?.click()}>Choose JSON</button>
        <button type="button" className="neo-btn" onClick={() => onChange(SUDOKU_ENVIRONMENT)}>Sudoku preset</button>
        <button type="button" className="neo-btn" onClick={download} disabled={!json}>Download</button>
      </div>

      <textarea
        rows={14}
        value={json}
        spellCheck={false}
        placeholder='{ "name": "sudoku", ... }'
        onChange={(e) => onChange(e.target.value)}
        className="neo-inset mt-6 box-border w-full resize-y bg-transparent px-4 py-3 font-mono text-sm text-primary placeholder:text-secondary/50"
      />

      <div className="mt-5 flex flex-wrap items-center gap-3">
        <span className="neo-chip">{syntaxError ? `JSON error: ${syntaxError}` : json ? "JSON OK" : "No environment"}</span>
        <button type="button" className="neo-btn" onClick={check} disabled={!json || !!syntaxError}>Validate</button>
        <input
          type="number" min={1} value={rows}
          onChange={(e) => setRows(Math.max(1, Number(e.target.value)))}
          className="neo-inset w-28 bg-transparent px-3 py-2 text-sm text-primary"
        />
        <button type="button" className="neo-btn" onClick={generate} disabled={busy || !json || !!syntaxError}>
          {busy ? "Generating..." : "Generate CSV"}
        </button>
      </div>
      {status && <p className="mt-3 text-sm text-secondary">{status}</p>}
    </section>
  );
}