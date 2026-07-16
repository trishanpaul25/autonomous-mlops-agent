"""
Metrics Registry for the Model Evaluation Agent.

Declarative catalogue of evaluation metrics organized by ML task type.

Design Principles
-----------------
* Open/Closed: New metrics are added by updating _METRIC_CONFIGS only.
  No changes to MetricsCalculator or any other module required.
* Single Responsibility: This module only knows ABOUT metrics — their
  names, callables, and whether they require probability predictions.
  It never calls any metric function itself.
* Dependency Inversion: MetricsCalculator depends on the MetricsRegistry
  abstraction, not on concrete sklearn metric functions.

Primary Metric Selection
------------------------
The primary metric is used to rank models. It is selected based on
task type and whether probability predictions are available:
  - binary_classification + proba available  → roc_auc
  - binary_classification + no proba         → f1
  - multiclass_classification                → f1_weighted
  - regression                               → r2

Usage
-----
    registry = MetricsRegistry()
    configs = registry.get_metrics("binary_classification")
    primary = registry.get_primary_metric("binary_classification", has_proba=True)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

@dataclass
class MetricConfig:
    """
    Declarative description of a single evaluation metric.

    Attributes
    ----------
    name : str
        Canonical name used as the key in metrics dicts (e.g. "roc_auc").
    callable : Callable
        The scikit-learn metric function.
    requires_proba : bool
        If True, this metric needs probability predictions (y_proba).
        The calculator skips it gracefully when proba is unavailable.
    kwargs : dict
        Additional keyword arguments passed to the metric function.
    """

    name: str
    fn: Callable[..., Any]
    requires_proba: bool = False
    kwargs: dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.kwargs is None:
            self.kwargs = {}

def _build_registry() -> dict[str, list[MetricConfig]]:
    """
    Build the full metric registry mapping task_type → list[MetricConfig].

    Using a factory function so imports happen once at first registry
    instantiation rather than at module load time.
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        log_loss,
        r2_score,
        mean_absolute_error,
        mean_squared_error,
        mean_absolute_percentage_error,
    )
    import math

    def rmse(y_true: Any, y_pred: Any) -> float:
        return math.sqrt(mean_squared_error(y_true, y_pred))

    _CLASSIFICATION_BASE = [
        MetricConfig(
            name="accuracy",
            fn=accuracy_score,
        ),
        MetricConfig(
            name="precision",
            fn=precision_score,
            kwargs={"average": "weighted", "zero_division": 0},
        ),
        MetricConfig(
            name="recall",
            fn=recall_score,
            kwargs={"average": "weighted", "zero_division": 0},
        ),
        MetricConfig(
            name="f1",
            fn=f1_score,
            kwargs={"average": "weighted", "zero_division": 0},
        ),
        MetricConfig(
            name="log_loss",
            fn=log_loss,
            requires_proba=True,
        ),
    ]

    _BINARY_ONLY = [
        MetricConfig(
            name="roc_auc",
            fn=roc_auc_score,
            requires_proba=True,
        ),
    ]

    _MULTICLASS_ROC_AUC = [
        MetricConfig(
            name="roc_auc",
            fn=roc_auc_score,
            requires_proba=True,
            kwargs={"multi_class": "ovr", "average": "weighted"},
        ),
    ]

    _REGRESSION = [
        MetricConfig(
            name="r2",
            fn=r2_score,
        ),
        MetricConfig(
            name="mae",
            fn=mean_absolute_error,
        ),
        MetricConfig(
            name="mse",
            fn=mean_squared_error,
        ),
        MetricConfig(
            name="rmse",
            fn=rmse,
        ),
        MetricConfig(
            name="mape",
            fn=mean_absolute_percentage_error,
        ),
    ]

    return {
        "binary_classification": _CLASSIFICATION_BASE + _BINARY_ONLY,
        "multiclass_classification": _CLASSIFICATION_BASE + _MULTICLASS_ROC_AUC,
        "regression": _REGRESSION,
    }

_PRIMARY_METRIC: dict[tuple[str, bool], str] = {
    ("binary_classification",     True):  "roc_auc",
    ("binary_classification",     False): "f1",
    ("multiclass_classification", True):  "roc_auc",
    ("multiclass_classification", False): "f1",
    ("regression",                True):  "r2",
    ("regression",                False): "r2",
}

# Fallback when task_type is unknown
_DEFAULT_PRIMARY_METRIC = "f1"


class MetricsRegistry:
    """
    Registry of evaluation metrics organized by ML task type.

    Methods
    -------
    get_metrics(task_type) -> list[MetricConfig]
        Returns the list of MetricConfig objects for the given task type.
        Returns an empty list if the task type is not supported.
    get_primary_metric(task_type, has_proba) -> str
        Returns the name of the primary ranking metric.
    is_supported(task_type) -> bool
        Returns True if the task type has registered metrics.
    list_supported_tasks() -> list[str]
        Returns all registered task type strings.
    """

    def __init__(self) -> None:
        self._registry: dict[str, list[MetricConfig]] = _build_registry()

    def get_metrics(self, task_type: str) -> list[MetricConfig]:
        """
        Return all MetricConfig objects for the given task type.

        Parameters
        ----------
        task_type : str
            ML task type (e.g. 'binary_classification', 'regression').

        Returns
        -------
        list[MetricConfig]
            Registered metrics, or empty list if task type unknown.
        """
        normalized = (task_type or "").lower().strip()
        return list(self._registry.get(normalized, []))

    def get_primary_metric(
        self,
        task_type: str,
        has_proba: bool = True,
    ) -> str:
        """
        Return the name of the primary metric used for ranking models.

        Parameters
        ----------
        task_type : str
            ML task type.
        has_proba : bool
            Whether the model supports predict_proba(). When False, metrics
            requiring probabilities (roc_auc, log_loss) are unavailable,
            so the primary metric falls back to a label-based alternative.

        Returns
        -------
        str
            Primary metric name (e.g. 'roc_auc', 'f1', 'r2').
        """
        normalized = (task_type or "").lower().strip()
        return _PRIMARY_METRIC.get(
            (normalized, has_proba),
            _DEFAULT_PRIMARY_METRIC,
        )

    def is_supported(self, task_type: str) -> bool:
        """
        Return True if evaluation metrics are registered for the task type.

        Parameters
        ----------
        task_type : str
            ML task type to check.

        Returns
        -------
        bool
            True if any metrics are registered for this task type.
        """
        normalized = (task_type or "").lower().strip()
        return normalized in self._registry

    def list_supported_tasks(self) -> list[str]:
        """
        Return all task type strings currently registered.

        Returns
        -------
        list[str]
            All registered task type strings.
        """
        return list(self._registry.keys())
