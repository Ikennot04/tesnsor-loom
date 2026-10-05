export interface CsvSummary {
  rows: number;
  columns: number;
  hasHeader: boolean;
}

/** Cheap client-side summary so the UI can warn before calling Rust. */
export function summarizeCsv(text: string): CsvSummary {
  const lines = text
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l.length > 0);

  if (lines.length === 0) return { rows: 0, columns: 0, hasHeader: false };

  const first = lines[0].split(",").map((c) => c.trim());
  const hasHeader = first.some((c) => c === "" || Number.isNaN(Number(c)));

  return {
    rows: lines.length - (hasHeader ? 1 : 0),
    columns: first.length,
    hasHeader,
  };
}