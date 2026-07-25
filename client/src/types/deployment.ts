// Mirrors server/schemas/deployment.py and server/schemas/monitoring.py.

export interface DeploymentSummary {
  id: string;
  // The value that works in POST /predict/{deployment_id} — this is the
  // pipeline run_id, NOT `id` above. This is what's shown/copied in the UI.
  deployment_id: string;

  model_name: string | null;
  dataset_name: string | null;
  status: string | null;

  // Whether the model is currently loaded in the backend's in-memory
  // ModelServerRegistry and will answer /predict immediately. False means
  // the deployment still exists (Postgres + MLflow have it) but hasn't
  // been reloaded since a restart.
  is_active: boolean;

  problem_type: string | null;
  target_column: string | null;
  // Generic metrics bag — {"accuracy": .., "f1": ..} for classification,
  // {"r2": .., "mae": ..} for regression. Replaces the old fixed
  // accuracy/precision/recall/f1_score columns; see MetricsGrid.
  metrics: Record<string, number | null>;

  endpoint: string | null;
  deployed_at: string | null;
}

export interface MonitoringSnapshot {
  id: string;
  deployment_id: string | null;
  prediction_count: number | null;
  average_latency: number | null;
  drift_score: number | null;
  accuracy: number | null;
  alert_status: string | null;
  last_checked: string | null;
}

export interface DeploymentDetail extends DeploymentSummary {
  monitoring: MonitoringSnapshot | null;
}

export interface PredictResponse {
  deployment_id: string;
  predictions: Record<string, unknown>[];
}
