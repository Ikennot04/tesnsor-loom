import type { ChangeEvent } from "react";
import { summarizeCsv } from "../utils/csv";

interface Props {
  csv: string;
  onChange: (csv: string) => void;
  onFeatureCountSuggested: (count: number) => void;
}

export function DataSection({ csv, onChange, onFeatureCountSuggested }: Props) {
  const summary = summarizeCsv(csv);

  const handleFile = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const text = await file.text();
    onChange(text);
    const s = summarizeCsv(text);
    if (s.columns > 1) onFeatureCountSuggested(s.columns - 1);
  };

  return (
    <fieldset>
      <legend>1. Data</legend>
      <p>
        <label>
          CSV file: <input type="file" accept=".csv,text/csv,text/plain" onChange={handleFile} />
        </label>
      </p>
      <p>
        <label>
          Or paste CSV (last column is the target):
          <br />
          <textarea
            rows={8}
            cols={70}
            value={csv}
            placeholder={"f1,f2,f3,target\n0.1,0.2,0.3,1.0"}
            onChange={(e) => onChange(e.target.value)}
          />
        </label>
      </p>
      <p>
        {summary.rows === 0
          ? "No data loaded."
          : `${summary.rows} rows, ${summary.columns} columns${
              summary.hasHeader ? " (header detected)" : ""
            }.`}
      </p>
    </fieldset>
  );
}