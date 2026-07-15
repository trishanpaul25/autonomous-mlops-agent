"""
Metrics Calculator for the Model Evaluation Agent.

Computes all evaluation metrics for a single model given its predictions.

Design Principles
-----------------
* Single Responsibility: This module only computes metric values — it
  does not run predictions, rank models, or write state.
* Error Isolation: each metric is computed inside its own try/except.
  One metric failure (e.g. MAPE division by zero, roc_auc with no proba)
  never prevents the remaining metrics from being computed.
* Graceful Degradation: when predict_proba() is unavailable (e.g. SVM
  without probability=True, linear models), probability-based metrics
  (roc_auc, log_loss) are simply omitted from the result dict.
* Open/Closed: the full metric catalogue is owned by MetricsRegistry.
  MetricsCalculator contains zero metric-specific logic.

Usage
-----
    calculator = MetricsCalculator()
    metrics = calculator.compute(
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,          # may be None
        task_type="binary_classification",
    )
    # Returns {"accuracy": 0.82, "f1": 0.81, "roc_auc": 0.88, ...}
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tools.model_evaluation.metrics_registry import MetricConfig, MetricsRegistry
from utils.logger import logger


class MetricsCalculator:
    """
    Computes evaluation metrics for a single model from its predictions.

    Constructor wires up MetricsRegistry as the sole source of metric
    callables and configuration. No metric-specific logic lives here.
    """

    def __init__(self) -> None:
        self.registry = MetricsRegistry()

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray | None,
        task_type: str,
    ) -> dict[str, float]:
        """
        Compute all registered metrics for the given task type.

        Each metric is computed independently inside a try/except block.
        Failures are logged as warnings and the metric is omitted from
        the result (not set to 0.0, which would be misleading).

        Parameters
        ----------
        y_true : np.ndarray
            Ground-truth target values from the test set.
        y_pred : np.ndarray
            Model predictions (class labels for classification,
            continuous values for regression).
        y_proba : np.ndarray | None
            Class probability estimates from predict_proba().
            None when the model does not support probabilities.
        task_type : str
            ML task type determining which metric set to use.

        Returns
        -------
        dict[str, float]
            Computed metric values keyed by metric name.
            Metrics that fail or require unavailable probabilities
            are omitted from the dict.
        """
        metrics_config: list[MetricConfig] = self.registry.get_metrics(task_type)

        if not metrics_config:
            logger.warning(
                "MetricsCalculator: no metrics registered for task type '%s'. "
                "Returning empty dict.",
                task_type,
            )
            return {}

        results: dict[str, float] = {}

        for config in metrics_config:
            # Skip probability-based metrics when proba is unavailable
            if config.requires_proba and y_proba is None:
                logger.debug(
                    "MetricsCalculator: skipping '%s' — no probability "
                    "predictions available.",
                    config.name,
                )
                continue

            try:
                if config.requires_proba:
                    value = self._compute_proba_metric(
                        config, y_true, y_pred, y_proba
                    )
                else:
                    value = float(config.fn(y_true, y_pred, **config.kwargs))

                results[config.name] = round(value, 6)

            except Exception as exc:
                logger.warning(
                    "MetricsCalculator: failed to compute '%s' — %s",
                    config.name,
                    exc,
                )

        return results

    def _compute_proba_metric(
        self,
        config: MetricConfig,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
    ) -> float:
        """
        Compute a probability-based metric, handling binary vs.
        multiclass shapes automatically.

        For binary classification, roc_auc_score expects the positive
        class probabilities (column 1 of y_proba).
        For multiclass, the full y_proba matrix is passed.

        Parameters
        ----------
        config : MetricConfig
            Metric configuration to compute.
        y_true : np.ndarray
            Ground-truth labels.
        y_pred : np.ndarray
            Predicted class labels (used for non-proba fallback).
        y_proba : np.ndarray
            Probability matrix from predict_proba().

        Returns
        -------
        float
            Computed metric value.
        """
        unique_classes = np.unique(y_true)
        is_binary = len(unique_classes) == 2

        # For binary tasks: roc_auc uses positive class column only
        # unless the metric already specifies multi_class in kwargs
        if is_binary and "multi_class" not in config.kwargs:
            proba_input = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
        else:
            proba_input = y_proba

        return float(config.fn(y_true, proba_input, **config.kwargs))

    def get_primary_metric_value(
        self,
        metrics: dict[str, float],
        task_type: str,
        has_proba: bool,
    ) -> tuple[str, float]:
        """
        Extract the primary metric name and value from a computed metrics dict.

        Falls back gracefully if the primary metric was not computed
        (e.g. roc_auc missing because no proba available).

        Parameters
        ----------
        metrics : dict[str, float]
            Already-computed metric values.
        task_type : str
            ML task type for primary metric lookup.
        has_proba : bool
            Whether probabilities were available.

        Returns
        -------
        tuple[str, float]
            (primary_metric_name, primary_metric_value).
            Falls back to the highest-valued available metric if the
            designated primary metric is missing.
        """
        primary_name = self.registry.get_primary_metric(task_type, has_proba)

        if primary_name in metrics:
            return primary_name, metrics[primary_name]

        # Fallback: use whatever metric is available with the highest value
        if metrics:
            fallback_name = max(metrics, key=lambda k: metrics[k])
            logger.warning(
                "MetricsCalculator: primary metric '%s' not available. "
                "Falling back to '%s' for ranking.",
                primary_name,
                fallback_name,
            )
            return fallback_name, metrics[fallback_name]

        return primary_name, 0.0
