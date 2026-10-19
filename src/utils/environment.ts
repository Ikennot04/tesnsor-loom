import { invoke } from "@tauri-apps/api/core";

export interface Generated { csv: string; in_features: number; out_features: number }

export const validateEnvironment = (json: string) =>
  invoke<[number, number]>("validate_environment", { json });

export const generateDataset = (json: string, rows: number, seed: number) =>
  invoke<Generated>("generate_dataset", { json, rows, seed });

export const SUDOKU_ENVIRONMENT = JSON.stringify({
  name: "sudoku",
  description: "Fill the 9x9 grid so every row, column and 3x3 box contains 1-9 exactly once.",
  task: { loss: "cross_entropy", classes: 9, value_scale: 9.0 },
  rules: [
    { id: "row_unique", description: "Each row has digits 1-9 once", kind: "all_different", scope: "row" },
    { id: "col_unique", description: "Each column has digits 1-9 once", kind: "all_different", scope: "col" },
    { id: "box_unique", description: "Each 3x3 box has digits 1-9 once", kind: "all_different", scope: "box" },
  ],
  generator: { kind: "sudoku", size: 9, min_clues: 28, max_clues: 40 },
}, null, 2);