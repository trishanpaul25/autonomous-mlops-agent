import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { deploymentApi } from "../../api/deploymentApi";
import { extractErrorMessage } from "../../api/axios";
import type { DeploymentSummary } from "../../types/deployment";
import "./Deployments.css";

function formatMetric(value: number | null): string {
  return value === null ? "—" : `${(value * 100).toFixed(1)}%`;
}

function formatDate(value: string | null): string {
  if (!value) return "—";
  return new Date(value).toLocaleString();
}

function CopyableId({ id }: { id: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(id);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard API unavailable (e.g. insecure context) — no-op, the
      // id is still visible and selectable in the DOM.
    }
  };

  return (
    <button
      type="button"
      className="deployment-id-chip"
      onClick={handleCopy}
      title="Copy deployment ID"
    >
      <span className="deployment-id-text">{id}</span>
      <span className="deployment-id-copy">{copied ? "Copied!" : "Copy"}</span>
    </button>
  );
}

export function Deployments() {
  const navigate = useNavigate();
  const [deployments, setDeployments] = useState<DeploymentSummary[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const res = await deploymentApi.list();
        if (!cancelled) setDeployments(res.data);
      } catch (err) {
        if (!cancelled) setError(extractErrorMessage(err));
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    };

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="deployments-page">
      <div className="deployments-container">
        <header className="deployments-header">
          <h1>Deployments</h1>
          <p>
            Every model your pipeline has deployed. Copy a Deployment ID to
            call it directly, or use Predict below to try it in the browser.
          </p>
        </header>

        {error && (
          <div className="deployments-error" role="alert">
            {error}
          </div>
        )}

        {isLoading && (
          <div className="deployments-loading" role="status" aria-live="polite">
            <div className="deployments-spinner" />
            <span>Loading deployments…</span>
          </div>
        )}

        {!isLoading && !error && deployments.length === 0 && (
          <div className="deployments-empty">
            <p>No deployments yet.</p>
            <p className="deployments-empty-hint">
              Run a pipeline that trains and deploys a model, then it'll show
              up here.
            </p>
          </div>
        )}

        {!isLoading && deployments.length > 0 && (
          <div className="deployments-grid">
            {deployments.map((deployment) => (
              <div key={deployment.id} className="deployment-card">
                <div className="deployment-card-top">
                  <h2 className="deployment-model-name">
                    {deployment.model_name ?? "Unnamed model"}
                  </h2>
                  <span
                    className={
                      deployment.is_active
                        ? "deployment-status deployment-status-active"
                        : "deployment-status deployment-status-cold"
                    }
                  >
                    {deployment.is_active ? "Active" : "Sleeping"}
                  </span>
                </div>

                <CopyableId id={deployment.deployment_id} />

                <div className="deployment-metrics">
                  <div className="deployment-metric">
                    <span className="deployment-metric-value">
                      {formatMetric(deployment.accuracy)}
                    </span>
                    <span className="deployment-metric-label">Accuracy</span>
                  </div>
                  <div className="deployment-metric">
                    <span className="deployment-metric-value">
                      {formatMetric(deployment.f1_score)}
                    </span>
                    <span className="deployment-metric-label">F1</span>
                  </div>
                  <div className="deployment-metric">
                    <span className="deployment-metric-value">
                      {formatMetric(deployment.precision)}
                    </span>
                    <span className="deployment-metric-label">Precision</span>
                  </div>
                  <div className="deployment-metric">
                    <span className="deployment-metric-value">
                      {formatMetric(deployment.recall)}
                    </span>
                    <span className="deployment-metric-label">Recall</span>
                  </div>
                </div>

                <div className="deployment-card-footer">
                  <div className="deployment-meta">
                    <span className="deployment-meta-label">Dataset</span>
                    <span className="deployment-meta-value">
                      {deployment.dataset_name ?? "—"}
                    </span>
                  </div>
                  <div className="deployment-meta">
                    <span className="deployment-meta-label">Deployed</span>
                    <span className="deployment-meta-value">
                      {formatDate(deployment.deployed_at)}
                    </span>
                  </div>
                </div>

                <button
                  type="button"
                  className="deployment-predict-button"
                  onClick={() =>
                    navigate(`/deployments/${deployment.deployment_id}`)
                  }
                >
                  Predict using this model
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
