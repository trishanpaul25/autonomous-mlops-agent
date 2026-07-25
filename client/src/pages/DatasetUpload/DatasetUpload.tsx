import { useState, useEffect, useRef, type FormEvent } from "react";
import { chatApi } from "../../api/chatApi";
import { usePipelineProgress } from "../../hooks/usePipelineProgress";
import { extractErrorMessage } from "../../api/axios";
import type { ChatResponse } from "../../types/pipeline";
import "./DatasetUpload.css";

const ACCEPTED_EXTENSIONS = ".csv,.xlsx,.xls,.json,.zip";

export function DatasetUpload() {
  const [prompt, setPrompt] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ChatResponse | null>(null);

  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Stubbed for future real-time progress — see usePipelineProgress.ts for
  // why `enabled` is hardcoded false today.
  const { events: progressEvents } = usePipelineProgress({
    runId: result?.run_id ?? null,
    enabled: false,
  });

  useEffect(() => {
    if (!isRunning) {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
      return;
    }

    timerRef.current = setInterval(() => {
      setElapsedSeconds((s) => s + 1);
    }, 1000);

    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [isRunning]);
  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();

    if (!prompt.trim()) {
      setError("Describe what you want the pipeline to do.");
      return;
    }

    setError(null);
    setResult(null);

    // Reset here instead
    setElapsedSeconds(0);
    setIsRunning(true);

    try {
      const res = await chatApi.runPipeline({ prompt, file });
      setResult(res.data);
    } catch (err) {
      setError(extractErrorMessage(err));
    } finally {
      setIsRunning(false);
    }
  };

  const formatElapsed = (totalSeconds: number) => {
    const m = Math.floor(totalSeconds / 60);
    const s = totalSeconds % 60;
    return `${m}:${s.toString().padStart(2, "0")}`;
  };

  return (
    <div className="upload-page">
      <div className="upload-container">
        <header className="upload-header">
          <h1>Run a Pipeline</h1>
          <p>
            Describe your task and optionally attach a dataset. This kicks off
            the full autonomous pipeline — ingestion through deployment.
          </p>
        </header>

        <form onSubmit={handleSubmit} className="upload-form">
          {error && (
            <div className="upload-error" role="alert">
              {error}
            </div>
          )}

          <div className="form-field">
            <label htmlFor="prompt">Prompt</label>
            <textarea
              id="prompt"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="e.g. Train a model to predict customer churn from this dataset"
              rows={4}
              disabled={isRunning}
              required
            />
          </div>

          <div className="form-field">
            <label htmlFor="file">Dataset (optional)</label>
            <div className="file-drop">
              <input
                id="file"
                type="file"
                accept={ACCEPTED_EXTENSIONS}
                onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                disabled={isRunning}
              />
              <span className="file-drop-hint">
                {file ? file.name : "CSV, Excel, JSON, or ZIP"}
              </span>
            </div>
          </div>

          <button type="submit" className="upload-submit" disabled={isRunning}>
            {isRunning ? "Running pipeline..." : "Run pipeline"}
          </button>
        </form>

        {isRunning && (
          <div className="upload-progress" role="status" aria-live="polite">
            <div className="upload-progress-spinner" />
            <div>
              <p className="upload-progress-title">
                Pipeline running — {formatElapsed(elapsedSeconds)}
              </p>
              <p className="upload-progress-subtitle">
                This runs the full pipeline synchronously and can take a few
                minutes. Please don&apos;t close this tab.
              </p>
            </div>
          </div>
        )}

        {/* Renders once live progress is wireable — currently always empty. */}
        {progressEvents.length > 0 && (
          <div className="upload-progress-log">
            {progressEvents.map((evt, i) => (
              <div key={i} className={`log-line log-${evt.type}`}>
                {evt.message}
              </div>
            ))}
          </div>
        )}

        {result && (
          <div className="upload-result">
            <div
              className={`result-status result-${result.status.toLowerCase()}`}
            >
              {result.status} — {result.execution_time?.toFixed(1)}s
            </div>

            {result.assistant_message && (
              <p className="result-message">{result.assistant_message}</p>
            )}

            <div className="result-grid">
              <div className="result-card">
                <span className="result-label">Dataset</span>
                <span className="result-value">
                  {result.dataset_name ?? "—"}
                </span>
              </div>
              <div className="result-card">
                <span className="result-label">Rows × Columns</span>
                <span className="result-value">
                  {result.rows ?? "—"} × {result.columns ?? "—"}
                </span>
              </div>
              <div className="result-card">
                <span className="result-label">Problem Type</span>
                <span className="result-value">
                  {result.problem_type ?? "—"}
                </span>
              </div>
              <div className="result-card">
                <span className="result-label">Best Model</span>
                <span className="result-value">
                  {result.model_evaluation?.best_model_name ?? "—"}
                </span>
              </div>
            </div>

            {result.completed_steps.length > 0 && (
              <div className="result-steps">
                <h3>Completed Steps</h3>
                <ol>
                  {result.completed_steps.map((step, i) => (
                    <li key={`${step}-${i}`}>{step}</li>
                  ))}
                </ol>
              </div>
            )}

            {result.logs.length > 0 && (
              <details className="result-logs">
                <summary>Pipeline logs ({result.logs.length})</summary>
                <pre>{result.logs.join("\n")}</pre>
              </details>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
