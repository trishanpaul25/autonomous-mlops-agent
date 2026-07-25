import { getMetricConfig } from "../config/metricsConfig";
import { formatMetricValue } from "../utils/formatMetricValue";
import type { MetricsRecord } from "../types/metrics";
import "./MetricsGrid.css";

interface MetricsGridProps {
  problemType: string | null | undefined;
  metrics: MetricsRecord | null | undefined;
  /** "detail" = big labeled cards (DeploymentDetail), "compact" = small
   *  tiles (Deployments list card, DatasetUpload result grid). */
  variant?: "detail" | "compact";
}

export function MetricsGrid({
  problemType,
  metrics,
  variant = "detail",
}: MetricsGridProps) {
  const config = getMetricConfig(problemType, metrics);
  const values = metrics ?? {};

  return (
    <div className={`metrics-grid metrics-grid-${variant}`}>
      {config.fields.map((field) => (
        <div
          key={field.key}
          className={`metrics-tile${
            field.key === config.primary ? " metrics-tile-primary" : ""
          }`}
        >
          <span className="metrics-tile-value">
            {formatMetricValue(values[field.key], field.format)}
          </span>
          <span className="metrics-tile-label">{field.label}</span>
        </div>
      ))}
    </div>
  );
}
