import { useRef, useState, type ChangeEvent } from "react";
import {
  evaluateModel,
  loadModel,
  pickModelFile,
  predictRows,
  type Evaluation,
  type ModelSummary,
} from "../api/model";

interface Props {
  /** Optional: the export path from the training form, offered as a shortcut. */
  suggestedPath?: string;
}

function describeScore(r2: number): string {
  if (r2 < 0)
    return "Worse than always guessing the average. The model needs more training (try a higher learning rate or more epochs).";
  if (r2 < 0.5)
    return "Weak. It has learned a little, but a lot of the pattern is missing.";
  if (r2 < 0.9) return "Decent. It captures most of the pattern.";
  return "Strong. Predictions track the targets closely.";
}

export function ModelTester({ suggestedPath }: Props) {
  const [summary, setSummary] = useState<ModelSummary | null>(null);
  const [error, setError] = useState("");

  const [inputText, setInputText] = useState("");
  const [prediction, setPrediction] = useState<number[] | null>(null);

  const [evalCsv, setEvalCsv] = useState("");
  const [evalFileName, setEvalFileName] = useState("");
  const [evaluation, setEvaluation] = useState<Evaluation | null>(null);
  const [busy, setBusy] = useState(false);
  const evalFileRef = useRef<HTMLInputElement>(null);

  const canUseSuggested =
    !!suggestedPath &&
    (suggestedPath.endsWith(".json") || suggestedPath.endsWith(".bin"));

  const openModel = async (path: string) => {
    setError("");
    setPrediction(null);
    setEvaluation(null);
    try {
      setSummary(await loadModel(path));
    } catch (err) {
      setSummary(null);
      setError(String(err));
    }
  };

  const handleBrowse = async () => {
    try {
      const path = await pickModelFile();
      if (path) await openModel(path);
    } catch (err) {
      setError(`Could not open the file dialog: ${String(err)}`);
    }
  };

  const handlePredict = async () => {
    if (!summary) return;
    setError("");
    setPrediction(null);

    const values = inputText
      .split(/[,\s]+/)
      .filter((s) => s.length > 0)
      .map(Number);

    if (values.length !== summary.input_size || values.some((v) => !Number.isFinite(v))) {
      setError(`Enter exactly ${summary.input_size} numbers, separated by commas.`);
      return;
    }

    setBusy(true);
    try {
      const [out] = await predictRows(summary.path, [values]);
      setPrediction(out);
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(false);
    }
  };

  const handleEvalFile = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setEvalCsv(await file.text());
    setEvalFileName(file.name);
    setEvaluation(null);
  };

  const handleEvaluate = async () => {
    if (!summary || !evalCsv) return;
    setError("");
    setBusy(true);
    try {
      setEvaluation(await evaluateModel(summary.path, evalCsv));
    } catch (err) {
      setEvaluation(null);
      setError(String(err));
    } finally {
      setBusy(false);
    }
  };

  const placeholder = summary
    ? Array.from({ length: summary.input_size }, (_, i) =>
        (0.1 * (i + 1)).toFixed(1),
      ).join(", ")
    : "";

  return (
    <section className="neo-raised h-full w-full p-6 sm:p-8">
      <header className="mb-6">
        <h2 className="text-xl font-bold tracking-tight text-primary">
          Test a trained model
        </h2>
        <p className="mt-1 text-sm text-secondary">
          Load a model to predict a row or score a CSV.
        </p>
      </header>

      <div className="neo-inset flex flex-wrap items-center justify-between gap-3 px-4 py-4">
        <span className="text-sm font-medium text-secondary">Model file</span>
        <div className="flex flex-wrap gap-2">
          <button type="button" className="neo-btn" onClick={handleBrowse}>
            Open model...
          </button>
          {canUseSuggested && (
            <button
              type="button"
              className="neo-btn-ghost"
              onClick={() => openModel(suggestedPath!)}
            >
              Use export path
            </button>
          )}
        </div>
      </div>

      {error && (
        <p role="alert" className="neo-alert mt-4">
          {error}
        </p>
      )}

      {summary && (
        <div className="mt-6 space-y-8">
          <div className="space-y-3">
            <h3 className="text-sm font-semibold text-primary">Model</h3>
            <p className="break-all text-xs text-secondary">{summary.path}</p>
            <div className="flex flex-wrap gap-2">
              <span className="neo-chip">{summary.input_size} inputs</span>
              <span className="neo-chip">{summary.output_size} output(s)</span>
              <span className="neo-chip">{summary.parameters} parameters</span>
            </div>
            <ol className="neo-inset space-y-2 px-4 py-3 text-sm text-secondary">
              {summary.layers.map((layer, i) => (
                <li key={i}>
                  {layer.inputs} to {layer.outputs} ({layer.activation})
                </li>
              ))}
            </ol>
          </div>

          <div className="space-y-3">
            <h3 className="text-sm font-semibold text-primary">Predict one row</h3>
            <label
              htmlFor="predict-features"
              className="block text-sm font-medium text-secondary"
            >
              Features ({summary.input_size} values, comma separated)
            </label>
            <div className="neo-inset flex flex-col gap-3 px-4 py-4 sm:flex-row sm:items-center">
              <input
                id="predict-features"
                type="text"
                value={inputText}
                placeholder={placeholder}
                onChange={(e) => setInputText(e.target.value)}
                className="neo-field min-w-0 flex-1 border-0 shadow-none"
              />
              <button
                type="button"
                className="neo-btn shrink-0"
                onClick={handlePredict}
                disabled={busy}
              >
                Predict
              </button>
            </div>
            {prediction && (
              <p className="neo-chip">
                Prediction: {prediction.map((v) => v.toFixed(4)).join(", ")}
              </p>
            )}
          </div>

          <div className="space-y-3">
            <h3 className="text-sm font-semibold text-primary">Score on a CSV</h3>
            <div className="neo-inset flex flex-wrap items-center justify-between gap-3 px-4 py-4">
              <span className="text-sm font-medium text-secondary">
                CSV with a target column (last column)
              </span>
              <input
                ref={evalFileRef}
                type="file"
                accept=".csv,text/csv,text/plain"
                className="sr-only"
                onChange={handleEvalFile}
              />
              <div className="flex flex-wrap gap-2">
                <button
                  type="button"
                  className="neo-btn-ghost"
                  onClick={() => evalFileRef.current?.click()}
                >
                  Choose CSV
                </button>
                <button
                  type="button"
                  className="neo-btn"
                  onClick={handleEvaluate}
                  disabled={busy || !evalCsv}
                >
                  Evaluate
                </button>
              </div>
            </div>
            {evalFileName && (
              <span className="neo-chip">Loaded {evalFileName}</span>
            )}
            {evaluation && (
              <div className="space-y-3">
                <div className="neo-inset overflow-x-auto px-2 py-2">
                  <table className="neo-table">
                    <tbody>
                      <tr>
                        <th>Rows</th>
                        <td>{evaluation.rows}</td>
                      </tr>
                      <tr>
                        <th>Mean squared error</th>
                        <td>{evaluation.mse.toFixed(4)}</td>
                      </tr>
                      <tr>
                        <th>R² score</th>
                        <td>{evaluation.r2.toFixed(3)}</td>
                      </tr>
                      <tr>
                        <th>Mean target</th>
                        <td>{evaluation.mean_target.toFixed(4)}</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
                <p className="text-sm text-secondary">{describeScore(evaluation.r2)}</p>
              </div>
            )}
          </div>
        </div>
      )}
    </section>
  );
}
