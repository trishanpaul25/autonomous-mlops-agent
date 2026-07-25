// Shared across any component that renders model metrics
// (DatasetUpload results, Deployments list/detail, comparison tables).

export type ProblemType = "classification" | "regression";

export type MetricFormat = "percent" | "decimal2" | "decimal3" | "currency" | "number";

export interface MetricFieldConfig {
  key: string; // key into the metrics record, e.g. "accuracy" | "r2"
  label: string; // display label, e.g. "Accuracy" | "R² Score"
  format: MetricFormat;
}

export interface ProblemTypeConfig {
  primary: string; // key of the "headline" metric for this problem type
  fields: MetricFieldConfig[];
}

// Generic metrics bag — matches what the backend already sends:
// ChatResponse.metrics, ModelEvaluationResult.best_model_metrics,
// and (once added) DeploymentSummary.metrics.
export type MetricsRecord = Record<string, number | null | undefined>;
