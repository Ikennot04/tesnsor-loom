import { useCallback, useEffect, useRef, useState } from "react";
import {
  startTrainingSession,
  subscribeToTraining,
} from "../api/trianing";

import type { NetworkSnapshot, TrainEvent, TrainingRequest} from "../types/type";

export interface EpochRow {
  epoch: number;
  avgLoss: number;
  score: number;
}

export interface TrainingSessionState {
  running: boolean;
  progress: number; // 0..1
  status: string;
  epochs: EpochRow[];
  log: string[];
  snapshot: NetworkSnapshot | null;
  start: (request: TrainingRequest) => Promise<void>;
}

export function useTrainingSession(): TrainingSessionState {
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState("Idle.");
  const [epochs, setEpochs] = useState<EpochRow[]>([]);
  const [log, setLog] = useState<string[]>([]);
  const [snapshot, setSnapshot] = useState<NetworkSnapshot | null>(null);
  const totalEpochs = useRef(0);

  const addLog = useCallback((message: string) => {
    const time = new Date().toLocaleTimeString();
    setLog((prev) => [...prev, `[${time}] ${message}`]);
  }, []);

  useEffect(() => {
    let cancelled = false;
    let unlisten: (() => void) | undefined;

    // One case per TrainEvent variant. A new backend event = one new case here.
    const handleEvent = (event: TrainEvent) => {
      switch (event.type) {
        case "EpochStarted":
          setStatus(`Epoch ${event.payload.epoch} of ${totalEpochs.current}...`);
          break;
        case "BatchCompleted":
          setProgress(event.payload.progress);
          break;
        case "EpochCompleted":
          setEpochs((prev) => [
            ...prev,
            {
              epoch: event.payload.epoch,
              avgLoss: event.payload.avg_loss,
              score: event.payload.val_accuracy,
            },
          ]);
          break;
        case "Network":
          setSnapshot(event.payload);
          break;
        case "Log":
          addLog(`[${event.payload.level}] ${event.payload.message}`);
          break;
        case "TrainingFinished":
          addLog(`Training finished: ${event.payload.success ? "success" : "failure"}`);
          break;
      }
    };

    subscribeToTraining({
      onEvent: handleEvent,
      onComplete: (path: string) => {
        setRunning(false);
        setProgress(1);
        setStatus(`Done. Model saved to ${path}`);
        addLog(`Model exported to ${path}`);
      },
      onError: (message: string) => {
        setRunning(false);
        setStatus(`Error: ${message}`);
        addLog(`ERROR: ${message}`);
      },
    })
      .then((fn: () => void) => {
        // StrictMode mounts twice; drop listeners from an already-cleaned-up effect.
        if (cancelled) fn();
        else unlisten = fn;
      })
      .catch((err: unknown) => addLog(`Could not subscribe to events: ${String(err)}`));

    return () => {
      cancelled = true;
      unlisten?.();
    };
  }, [addLog]);

  const start = useCallback(
    async (request: TrainingRequest) => {
      totalEpochs.current = request.config.epochs;
      setEpochs([]);
      setSnapshot(null);
      setProgress(0);
      setRunning(true);
      setStatus("Starting...");
      addLog(`Starting training: ${JSON.stringify(request.config)}`);

      try {
        const reply = await startTrainingSession(request);
        addLog(reply);
      } catch (err) {
        setRunning(false);
        setStatus(`Failed to start: ${String(err)}`);
        addLog(`ERROR: ${String(err)}`);
      }
    },
    [addLog],
  );

  return { running, progress, status, epochs, log, snapshot, start };
}