import { useState, type ChangeEvent } from "react";
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
  if (r2 < 0) return "Worse than always guessing the average. The model needs more training (try a higher learning rate or more epochs).";
  if (r2 < 0.5) return "Weak. It has learned a little, but a lot of the pattern is missing.";
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

  const canUseSuggested =
    !!suggestedPath && (suggestedPath.endsWith(".json") || suggestedPath.endsWith(".bin"));

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
    ? Array.from({ length: summary.input_size }, (_, i) => (0.1 * (i + 1)).toFixed(1)).join(", ")
    : "";

  return (
    <section>
      <h2>Test a trained model</h2>

      <p>
        <button type="button" onClick={handleBrowse}>
          Open model...
        </button>{" "}
        {canUseSuggested && (
          <button type="button" onClick={() => openModel(suggestedPath!)}>
            Use export path
          </button>
        )}
      </p>

      {error && <p role="alert">{error}</p>}

      {summary && (
        <>
          <h3>Model</h3>
          <p>{summary.path}</p>
          <p>
            {summary.input_size} inputs, {summary.output_size} output(s), {summary.parameters} parameters.
          </p>
          <ol>
            {summary.layers.map((layer, i) => (
              <li key={i}>
                {layer.inputs} to {layer.outputs} ({layer.activation})
              </li>
            ))}
          </ol>

          <h3>Predict one row</h3>
          <p>
            <label>
              Features ({summary.input_size} values, comma separated):
              <br />
              <input
                type="text"
                size={60}
                value={inputText}
                placeholder={placeholder}
                onChange={(e) => setInputText(e.target.value)}
              />
            </label>{" "}
            <button type="button" onClick={handlePredict} disabled={busy}>
              Predict
            </button>
          </p>
          {prediction && <p>Prediction: {prediction.map((v) => v.toFixed(4)).join(", ")}</p>}

          <h3>Score on a CSV</h3>
          <p>
            <label>
              CSV with a target column (last column):{" "}
              <input type="file" accept=".csv,text/csv,text/plain" onChange={handleEvalFile} />
            </label>{" "}
            <button type="button" onClick={handleEvaluate} disabled={busy || !evalCsv}>
              Evaluate
            </button>
          </p>
          {evalFileName && <p>Loaded {evalFileName}.</p>}
          {evaluation && (
            <>
              <table border={1} cellPadding={4}>
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
              <p>{describeScore(evaluation.r2)}</p>
            </>
          )}
        </>
      )}
    </section>
  );
}