import { useRef, type ChangeEvent } from "react";
import { summarizeCsv } from "../utils/csv";

interface Props {
  csv: string;
  onChange: (csv: string) => void;
  onFeatureCountSuggested: (count: number) => void;
}

export function DataSection({ csv, onChange, onFeatureCountSuggested }: Props) {
  const summary = summarizeCsv(csv);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const text = await file.text();
    onChange(text);
    const s = summarizeCsv(text);
    if (s.columns > 1) onFeatureCountSuggested(s.columns - 1);
  };

  const summaryText =
    summary.rows === 0
      ? "No data loaded."
      : `${summary.rows} rows, ${summary.columns} columns${
          summary.hasHeader ? " (header detected)" : ""
        }.`;

  return (
    <section className="neo-raised h-full w-full p-6 sm:p-8">
      <header className="mb-6">
        <h2 className="text-xl font-bold tracking-tight text-primary">1. Data</h2>
        <p className="mt-1 text-sm text-secondary">
          Load or paste CSV data. The last column is the target.
        </p>
      </header>

      <div className="neo-inset flex flex-wrap items-center justify-between gap-3 px-4 py-4">
        <span className="text-sm font-medium text-secondary">CSV file</span>
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv,text/csv,text/plain"
          className="sr-only"
          onChange={handleFile}
        />
        <button
          type="button"
          className="neo-btn"
          onClick={() => fileInputRef.current?.click()}
        >
          Choose CSV
        </button>
      </div>

      <div className="mt-6">
        <label
          htmlFor="csv-paste"
          className="mb-2 block text-sm font-medium text-secondary"
        >
          Or paste CSV
        </label>
        <textarea
          id="csv-paste"
          rows={8}
          value={csv}
          placeholder={"f1,f2,f3,target\n0.1,0.2,0.3,1.0"}
          onChange={(e) => onChange(e.target.value)}
          className="neo-inset box-border w-full resize-y bg-transparent px-4 py-3 font-mono text-sm text-primary placeholder:text-secondary/50"
        />
      </div>

      <div className="mt-5">
        <span className="neo-chip">{summaryText}</span>
      </div>
    </section>
  );
}
