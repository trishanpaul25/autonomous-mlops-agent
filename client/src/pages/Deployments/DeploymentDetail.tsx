import { useEffect, useState, type FormEvent } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { deploymentApi } from "../../api/deploymentApi";
import { extractErrorMessage } from "../../api/axios";
import type { DeploymentDetail as DeploymentDetailType } from "../../types/deployment";
import { MetricsGrid } from "../../components/MetricsGrid";
import "./DeploymentDetail.css";

const SAMPLE_RECORD_PLACEHOLDER = `[
  { "feature_1": 0, "feature_2": "value" }
]`;

function formatDate(value: string | null): string {
  if (!value) return "—";
  return new Date(value).toLocaleString();
}

export function DeploymentDetail() {
  const { deploymentId } = useParams<{ deploymentId: string }>();
  const navigate = useNavigate();

  const [deployment, setDeployment] = useState<DeploymentDetailType | null>(
    null,
  );
  const [isLoading, setIsLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [copied, setCopied] = useState(false);

  const [recordsInput, setRecordsInput] = useState("");
  const [isPredicting, setIsPredicting] = useState(false);
  const [predictError, setPredictError] = useState<string | null>(null);
  const [predictions, setPredictions] = useState<Record<
    string,
    unknown
  >[] | null>(null);

  useEffect(() => {
    if (!deploymentId) return;
    let cancelled = false;

    const load = async () => {
      setIsLoading(true);
      setLoadError(null);
      try {
        const res = await deploymentApi.get(deploymentId);
        if (!cancelled) setDeployment(res.data);
      } catch (err) {
        if (!cancelled) setLoadError(extractErrorMessage(err));
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    };

    load();
    return () => {
      cancelled = true;
    };
  }, [deploymentId]);

  const handleCopy = async () => {
    if (!deploymentId) return;
    try {
      await navigator.clipboard.writeText(deploymentId);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard API unavailable — no-op, the id is still visible/selectable.
    }
  };

  const handlePredict = async (e: FormEvent) => {
    e.preventDefault();
    if (!deploymentId) return;

    setPredictError(null);
    setPredictions(null);

    let records: Record<string, unknown>[];
    try {
      const parsed = JSON.parse(recordsInput);
      records = Array.isArray(parsed) ? parsed : [parsed];
    } catch {
      setPredictError(
        "That's not valid JSON. Enter a single record ({...}) or an array of records ([{...}, {...}]).",
      );
      return;
    }

    if (records.length === 0) {
      setPredictError("Enter at least one record to score.");
      return;
    }

    setIsPredicting(true);
    try {
      const res = await deploymentApi.predict(deploymentId, records);
      setPredictions(res.data.predictions);
    } catch (err) {
      setPredictError(extractErrorMessage(err));
    } finally {
      setIsPredicting(false);
    }
  };

  if (isLoading) {
    return (
      <div className="deployment-detail-page">
        <div className="deployment-detail-loading" role="status" aria-live="polite">
          <div className="deployment-detail-spinner" />
          <span>Loading deployment…</span>
        </div>
      </div>
    );
  }

  if (loadError || !deployment) {
    return (
      <div className="deployment-detail-page">
        <div className="deployment-detail-container">
          <button
            type="button"
            className="deployment-detail-back"
            onClick={() => navigate("/deployments")}
          >
            ← Back to Deployments
          </button>
          <div className="deployment-detail-error" role="alert">
            {loadError ?? "Deployment not found."}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="deployment-detail-page">
      <div className="deployment-detail-container">
        <button
          type="button"
          className="deployment-detail-back"
          onClick={() => navigate("/deployments")}
        >
          ← Back to Deployments
        </button>

        <header className="deployment-detail-header">
          <div>
            <h1>{deployment.model_name ?? "Unnamed model"}</h1>
            <p className="deployment-detail-subtitle">
              {deployment.dataset_name
                ? `Trained on ${deployment.dataset_name}`
                : "Dataset unavailable"}
              {deployment.target_column && (
                <> · Predicting <strong>{deployment.target_column}</strong></>
              )}
            </p>
          </div>
          <span
            className={
              deployment.is_active
                ? "deployment-status deployment-status-active"
                : "deployment-status deployment-status-cold"
            }
          >
            {deployment.is_active ? "Active" : "Sleeping"}
          </span>
        </header>

        <div className="deployment-detail-id-row">
          <span className="deployment-detail-id-label">Deployment ID</span>
          <button
            type="button"
            className="deployment-detail-id-chip"
            onClick={handleCopy}
            title="Copy deployment ID"
          >
            <span className="deployment-id-text">{deploymentId}</span>
            <span className="deployment-id-copy">
              {copied ? "Copied!" : "Copy"}
            </span>
          </button>
        </div>

        <MetricsGrid
          problemType={deployment.problem_type}
          metrics={deployment.metrics}
          variant="detail"
        />

        <div className="deployment-detail-grid">
          <div className="deployment-detail-card">
            <span className="deployment-detail-label">Status</span>
            <span className="deployment-detail-value">
              {deployment.status ?? "—"}
            </span>
          </div>
          <div className="deployment-detail-card">
            <span className="deployment-detail-label">Deployed</span>
            <span className="deployment-detail-value deployment-detail-value-sm">
              {formatDate(deployment.deployed_at)}
            </span>
          </div>
        </div>

        {deployment.monitoring && (
          <div className="deployment-detail-monitoring">
            <h3>Monitoring</h3>
            <div className="deployment-detail-grid">
              <div className="deployment-detail-card">
                <span className="deployment-detail-label">Predictions logged</span>
                <span className="deployment-detail-value">
                  {deployment.monitoring.prediction_count ?? "—"}
                </span>
              </div>
              <div className="deployment-detail-card">
                <span className="deployment-detail-label">Avg latency</span>
                <span className="deployment-detail-value">
                  {deployment.monitoring.average_latency !== null
                    ? `${deployment.monitoring.average_latency.toFixed(1)}ms`
                    : "—"}
                </span>
              </div>
              <div className="deployment-detail-card">
                <span className="deployment-detail-label">Drift score</span>
                <span className="deployment-detail-value">
                  {deployment.monitoring.drift_score !== null
                    ? deployment.monitoring.drift_score.toFixed(3)
                    : "—"}
                </span>
              </div>
              <div className="deployment-detail-card">
                <span className="deployment-detail-label">Alert status</span>
                <span className="deployment-detail-value deployment-detail-value-sm">
                  {deployment.monitoring.alert_status ?? "—"}
                </span>
              </div>
            </div>
          </div>
        )}

        <div className="deployment-predict-panel">
          <h3>Predict</h3>
          <p className="deployment-predict-hint">
            Paste one or more raw input rows as JSON — the deployed model
            handles feature engineering itself.
          </p>

          <form onSubmit={handlePredict} className="deployment-predict-form">
            <textarea
              value={recordsInput}
              onChange={(e) => setRecordsInput(e.target.value)}
              placeholder={SAMPLE_RECORD_PLACEHOLDER}
              rows={6}
              disabled={isPredicting}
              required
            />

            {predictError && (
              <div className="deployment-detail-error" role="alert">
                {predictError}
              </div>
            )}

            <button
              type="submit"
              className="deployment-predict-submit"
              disabled={isPredicting}
            >
              {isPredicting ? "Predicting…" : "Run prediction"}
            </button>
          </form>

          {predictions && (
            <div className="deployment-predict-results">
              <h4>Predictions</h4>
              <pre>{JSON.stringify(predictions, null, 2)}</pre>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
