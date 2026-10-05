import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";

export interface LayerSummary {
  inputs: number;
  outputs: number;
  activation: string;
}

export interface ModelSummary {
  path: string;
  metadata: string;
  input_size: number;
  output_size: number;
  parameters: number;
  layers: LayerSummary[];
}

export interface Evaluation {
  rows: number;
  mse: number;
  r2: number;
  mean_target: number;
}

/** Native open dialog. Resolves to null if the user cancels. */
export async function pickModelFile(): Promise<string | null> {
  const selected = await open({
    title: "Choose a trained model",
    multiple: false,
    directory: false,
    filters: [{ name: "TensorLoom model", extensions: ["json", "bin"] }],
  });
  return typeof selected === "string" ? selected : null;
}

export function loadModel(path: string): Promise<ModelSummary> {
  return invoke<ModelSummary>("load_model", { path });
}

export function predictRows(path: string, rows: number[][]): Promise<number[][]> {
  return invoke<number[][]>("predict_rows", { path, rows });
}

export function evaluateModel(path: string, data: string): Promise<Evaluation> {
  return invoke<Evaluation>("evaluate_model", { path, data });
}