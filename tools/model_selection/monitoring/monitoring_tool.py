"""
Monitoring Tool.

Pure computation, no LLM, no DB access — takes raw prediction data and
a reference snapshot, returns aggregated metrics. Mirrors the
"deterministic tool" contract used elsewhere (e.g. DatasetSnapshotBuilder):
in, transform, out, nothing hidden.

Drift detection here is a transparent heuristic, not a formal
statistical test. DatasetSnapshot only stores summary statistics
(mean/std/min/max for numerical columns, value frequencies for
categorical ones) rather than raw training samples, so a real
two-sample test (KS, chi-square) isn't actually possible against what
we persisted — there's nothing to compare the live sample to at the
distribution level, only at the summary-statistic level. What's
implemented instead:

- Numerical columns: standardized mean shift, i.e. how many reference
  standard deviations the live batch's mean has moved:
  `abs(live_mean - ref_mean) / (ref_std + epsilon)`, clipped and
  normalized to [0, 1].
- Categorical columns: total variation distance between live and
  reference frequency distributions, which needs only proportions
  (already bounded in [0, 1], no clipping needed).

Per-feature scores are averaged into one `drift_score`. This will
miss shape changes that don't move the mean (e.g. bimodal splitting,
variance changes without a mean shift) — worth knowing if this number
is ever trusted for something high-stakes; it's a cheap first signal,
not a replacement for a proper drift library.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

_EPSILON = 1e-9

# Numerical mean-shift is unbounded in principle; clip at this many
# reference std-devs before normalizing to [0, 1] so one wild outlier
# column doesn't dominate the averaged drift_score.
_MAX_STD_SHIFT = 5.0

DEFAULT_DRIFT_WARNING_THRESHOLD = 0.3
DEFAULT_DRIFT_CRITICAL_THRESHOLD = 0.6

# Below this many live predictions, drift and accuracy numbers are
# statistically too noisy to act on — still computed, but alert_status
# won't fire off them alone.
DEFAULT_MIN_PREDICTIONS_FOR_ALERTING = 30


class MonitoringTool:

    def __init__(
        self,
        drift_warning_threshold: float = DEFAULT_DRIFT_WARNING_THRESHOLD,
        drift_critical_threshold: float = DEFAULT_DRIFT_CRITICAL_THRESHOLD,
        min_predictions_for_alerting: int = DEFAULT_MIN_PREDICTIONS_FOR_ALERTING,
    ):
        self.drift_warning_threshold = drift_warning_threshold
        self.drift_critical_threshold = drift_critical_threshold
        self.min_predictions_for_alerting = min_predictions_for_alerting

    def compute(
        self,
        prediction_logs: list[Any],
        feature_statistics: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """
        `prediction_logs` — PredictionLog rows (needs .latency_ms,
        .input_payload, .ground_truth, .prediction attributes).
        `feature_statistics` — DatasetSnapshot.feature_statistics, or
        None if no snapshot exists for this deployment yet.

        Returns a dict matching the Monitoring table's columns plus
        `warnings` and `summary` for the caller to log/display.
        """
        warnings: list[str] = []
        prediction_count = len(prediction_logs)

        average_latency = self._average_latency(prediction_logs)
        accuracy = self._accuracy(prediction_logs)

        drift_score = None
        if feature_statistics is None:
            warnings.append(
                "No DatasetSnapshot found for this deployment — drift_score "
                "cannot be computed. This can happen if the snapshot capture "
                "step failed at deployment time; check earlier warnings for "
                "this deployment_id."
            )
        elif prediction_count == 0:
            warnings.append("No prediction logs in this window — nothing to compare for drift.")
        else:
            drift_score, per_feature_warnings = self._compute_drift(
                prediction_logs, feature_statistics
            )
            warnings.extend(per_feature_warnings)

        alert_status = self._determine_alert_status(
            prediction_count=prediction_count,
            drift_score=drift_score,
            warnings=warnings,
        )

        summary = self._build_summary(
            prediction_count=prediction_count,
            average_latency=average_latency,
            drift_score=drift_score,
            accuracy=accuracy,
            alert_status=alert_status,
        )

        return {
            "prediction_count": prediction_count,
            "average_latency": average_latency,
            "drift_score": drift_score,
            "accuracy": accuracy,
            "alert_status": alert_status,
            "warnings": warnings,
            "summary": summary,
        }

    # ------------------------------------------------------------------

    @staticmethod
    def _average_latency(prediction_logs: list[Any]) -> float | None:
        latencies = [p.latency_ms for p in prediction_logs if p.latency_ms is not None]
        if not latencies:
            return None
        return sum(latencies) / len(latencies)

    @staticmethod
    def _accuracy(prediction_logs: list[Any]) -> float | None:
        """
        Only counts rows where ground_truth has actually been filled
        in. There's currently no write path for ground_truth anywhere
        in the codebase, so in practice this returns None until that
        exists — that's expected, not a bug here.
        """
        labeled = [
            p for p in prediction_logs
            if p.ground_truth is not None and p.prediction is not None
        ]
        if not labeled:
            return None

        correct = 0
        total = 0
        for p in labeled:
            # prediction/ground_truth are JSON lists of per-row records,
            # matching PredictRequest/PredictResponse shape.
            pred_rows = p.prediction if isinstance(p.prediction, list) else [p.prediction]
            truth_rows = p.ground_truth if isinstance(p.ground_truth, list) else [p.ground_truth]
            for pred_row, truth_row in zip(pred_rows, truth_rows):
                total += 1
                if pred_row == truth_row:
                    correct += 1

        if total == 0:
            return None
        return correct / total

    def _compute_drift(
        self,
        prediction_logs: list[Any],
        feature_statistics: dict[str, Any],
    ) -> tuple[float | None, list[str]]:
        warnings: list[str] = []

        rows: list[dict[str, Any]] = []
        for log in prediction_logs:
            payload = log.input_payload
            if isinstance(payload, list):
                rows.extend(payload)
            elif isinstance(payload, dict):
                rows.append(payload)

        if not rows:
            warnings.append("Prediction logs had no usable input_payload — drift not computed.")
            return None, warnings

        live_df = pd.DataFrame(rows)

        per_feature_scores: list[float] = []
        for column, ref_stats in feature_statistics.items():
            if column not in live_df.columns:
                # Column present in training but never sent in live
                # requests — not itself a drift signal, just missing data.
                continue

            live_series = live_df[column].dropna()
            if live_series.empty:
                continue

            if ref_stats.get("type") == "numerical":
                score = self._numerical_drift(live_series, ref_stats)
            elif ref_stats.get("type") == "categorical":
                score = self._categorical_drift(live_series, ref_stats)
            else:
                continue

            if score is not None:
                per_feature_scores.append(score)

        if not per_feature_scores:
            warnings.append(
                "No overlapping columns between live requests and the "
                "training snapshot — drift not computed."
            )
            return None, warnings

        return sum(per_feature_scores) / len(per_feature_scores), warnings

    @staticmethod
    def _numerical_drift(live_series: pd.Series, ref_stats: dict[str, Any]) -> float | None:
        ref_mean = ref_stats.get("mean")
        ref_std = ref_stats.get("std")
        if ref_mean is None:
            return None

        live_numeric = pd.to_numeric(live_series, errors="coerce").dropna()
        if live_numeric.empty:
            return None

        live_mean = float(live_numeric.mean())
        std_shift = abs(live_mean - ref_mean) / ((ref_std or 0.0) + _EPSILON)
        return min(std_shift, _MAX_STD_SHIFT) / _MAX_STD_SHIFT

    @staticmethod
    def _categorical_drift(live_series: pd.Series, ref_stats: dict[str, Any]) -> float | None:
        ref_freqs: dict[str, float] = ref_stats.get("frequencies") or {}
        if not ref_freqs:
            return None

        live_freqs = live_series.astype(str).value_counts(normalize=True).to_dict()

        categories = set(ref_freqs) | set(live_freqs)
        total_variation = 0.5 * sum(
            abs(live_freqs.get(cat, 0.0) - ref_freqs.get(cat, 0.0)) for cat in categories
        )
        return total_variation

    def _determine_alert_status(
        self,
        prediction_count: int,
        drift_score: float | None,
        warnings: list[str],
    ) -> str:
        if prediction_count < self.min_predictions_for_alerting:
            return "insufficient_data"

        if drift_score is None:
            return "unknown"

        if drift_score >= self.drift_critical_threshold:
            return "critical"

        if drift_score >= self.drift_warning_threshold:
            return "warning"

        return "ok"

    @staticmethod
    def _build_summary(
        prediction_count: int,
        average_latency: float | None,
        drift_score: float | None,
        accuracy: float | None,
        alert_status: str,
    ) -> str:
        parts = [f"{prediction_count} prediction(s) observed"]
        if average_latency is not None:
            parts.append(f"avg latency {average_latency:.1f}ms")
        if drift_score is not None:
            parts.append(f"drift score {drift_score:.3f}")
        if accuracy is not None:
            parts.append(f"accuracy {accuracy:.3f}")
        parts.append(f"status: {alert_status}")
        return " | ".join(parts)
