import type { NetworkSnapshot } from "../types/type";
import type { EpochRow } from "../hooks/useTrainingSession";
import { NetworkVisualizer } from "./NetworkVisualizer";
import { LossChart } from "./LossChart";

interface Props {
  progress: number; // 0..1
  status: string;
  epochs: EpochRow[];
  log: string[];
  snapshot: NetworkSnapshot | null;
}

const VISIBLE_ROWS = 15;

export function ProgressPanel({ progress, status, epochs, log, snapshot }: Props) {
  const pct = Math.round(Math.max(0, Math.min(1, progress)) * 100);
  const recent = epochs.slice(-VISIBLE_ROWS);

  return (
    <section>
      <h2>Progress</h2>
      <p>
        <progress value={pct} max={100} /> {pct}%
      </p>
      <p>{status}</p>

      <h3>Network (live)</h3>
      <NetworkVisualizer snapshot={snapshot} />

      <h3>Loss</h3>
      <LossChart values={epochs.map((row) => row.avgLoss)} />

      <h3>Recent epochs</h3>
      <table border={1} cellPadding={4}>
        <thead>
          <tr>
            <th>Epoch</th>
            <th>Avg loss</th>
            <th>Score</th>
          </tr>
        </thead>
        <tbody>
          {recent.map((row) => (
            <tr key={row.epoch}>
              <td>{row.epoch}</td>
              <td>{row.avgLoss.toFixed(6)}</td>
              <td>{row.score.toFixed(4)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {epochs.length > VISIBLE_ROWS && <p>Showing last {VISIBLE_ROWS} of {epochs.length} epochs.</p>}

      <h2>Log</h2>
      <pre>{log.join("\n")}</pre>
    </section>
  );
}