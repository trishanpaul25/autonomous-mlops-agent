import type { MetricFormat } from "../types/metrics";

// Extend this if a metric ever needs a currency other than USD — pull
// it from the dataset/target metadata instead of hardcoding "$".
export function formatMetricValue(
  value: number | null | undefined,
  format: MetricFormat,
): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "—";
  }

  switch (format) {
    case "percent":
      return `${(value * 100).toFixed(1)}%`;
    case "decimal2":
      return value.toFixed(2);
    case "decimal3":
      return value.toFixed(3);
    case "currency":
      return value.toLocaleString(undefined, {
        style: "currency",
        currency: "USD",
        maximumFractionDigits: 0,
      });
    case "number":
    default:
      return value.toLocaleString();
  }
}
