import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import type { TrainEvent, TrainingRequest } from "../types/type";

// Event names emitted by the Rust side (src-tauri/src/lib.rs).
const EVENT_PROGRESS = "training-progress";
const EVENT_COMPLETE = "training-complete";
const EVENT_ERROR = "training-error";

export interface TrainingHandlers {
  onEvent: (event: TrainEvent) => void;
  onComplete: (outputPath: string) => void;
  onError: (message: string) => void;
}

/** Kick off a training session. Resolves as soon as the backend thread is spawned. */
export function startTrainingSession(request: TrainingRequest): Promise<string> {
  // Tauri maps Rust snake_case params to camelCase keys here:
  // export_format -> exportFormat, output_path -> outputPath.
  return invoke<string>("start_training_session", {
    config: request.config,
    data: request.data,
    exportFormat: request.exportFormat,
    outputPath: request.outputPath,
  });
}

/** Subscribe to all training events. Returns a function that removes every listener. */
export async function subscribeToTraining(
  handlers: TrainingHandlers,
): Promise<UnlistenFn> {
  const unlisteners: UnlistenFn[] = [];

  try {
    unlisteners.push(
      await listen<TrainEvent>(EVENT_PROGRESS, (e) => handlers.onEvent(e.payload)),
      await listen<string>(EVENT_COMPLETE, (e) => handlers.onComplete(e.payload)),
      await listen<string>(EVENT_ERROR, (e) => handlers.onError(e.payload)),
    );
  } catch (err) {
    unlisteners.forEach((fn) => fn());
    throw err;
  }

  return () => unlisteners.forEach((fn) => fn());
}