import type { MetricsRecord, ProblemType, ProblemTypeConfig } from "../types/metrics";

// Single source of truth for "what metrics exist for this problem type,
// what do we call them, and how do we format them". Add a new problem
// type (e.g. "clustering") by adding one block here — no new branches
// in any component.
export const METRIC_CONFIG: Record<ProblemType, ProblemTypeConfig> = {
  classification: {
    primary: "accuracy",
    fields: [
      { key: "accuracy", label: "Accuracy", format: "percent" },
      { key: "f1", label: "F1 Score", format: "percent" },
      { key: "precision", label: "Precision", format: "percent" },
      { key: "recall", label: "Recall", format: "percent" },
    ],
  },
  regression: {
    primary: "r2",
    fields: [
      { key: "r2", label: "R\u00b2 Score", format: "decimal3" },
      { key: "mae", label: "MAE", format: "currency" },
      { key: "rmse", label: "RMSE", format: "currency" },
      { key: "mape", label: "MAPE", format: "percent" },
    ],
  },
};

const REGRESSION_ONLY_KEYS = ["r2", "mae", "mse", "rmse", "mape"];

/**
 * `problem_type` should always be present — the pipeline knows this from
 * the moment model selection runs. But every consumer of this config
 * reads from a DB row or an API response, either of which can be stale,
 * missing, or (during rollout of this feature) simply not backfilled
 * yet. Rather than let a missing/unexpected value crash the metrics
 * grid, fall back to inferring the type from which metric keys are
 * actually present.
 */
export function resolveProblemType(
  problemType: string | null | undefined,
  metrics: MetricsRecord | null | undefined,
): ProblemType {
  if (problemType === "classification" || problemType === "regression") {
    return problemType;
  }

  const hasRegressionKey = Object.keys(metrics ?? {}).some((key) =>
    REGRESSION_ONLY_KEYS.includes(key),
  );

  return hasRegressionKey ? "regression" : "classification";
}

export function getMetricConfig(
  problemType: string | null | undefined,
  metrics: MetricsRecord | null | undefined,
): ProblemTypeConfig {
  return METRIC_CONFIG[resolveProblemType(problemType, metrics)];
}
