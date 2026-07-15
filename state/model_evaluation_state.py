"""
State used by the Model Evaluation Agent.

Stores every artifact produced during the evaluation step:
  - Per-model metric records (serializable dicts)
  - Ranked model list and best model identification
  - Side-by-side comparison table
  - Visualization-ready data structures (JSON-serializable)
  - Optional LLM-generated narrative

Design notes
------------
Visualization data is stored as plain Python dicts of lists so that
the state remains fully JSON-serializable. No numpy arrays are stored here.
Downstream consumers (API layer, dashboards) reconstruct arrays as needed.

The best model's fitted estimator is NOT stored here — it lives in
HyperparameterOptimizationState.optimized_model_objects keyed by
best_model_identifier. The Registry Agent uses best_model_identifier
to retrieve and serialize the fitted object.

This state is written once by ModelEvaluationAgent and consumed
read-only by downstream agents (Registry, Deployment, Monitoring).
"""

from typing import Any

from pydantic import Field

from .base_state import BaseState


class ModelEvaluationState(BaseState):
    """
    Stores the results of the model evaluation step.

    Written once by the ModelEvaluationAgent and consumed read-only
    by downstream agents (Registry, Deployment, Monitoring).
    """

    # Whether evaluation completed (at least one model evaluated)
    is_completed: bool = False

    # Overall evaluation outcome
    # "completed" — all models evaluated successfully
    # "partial"   — at least one succeeded, at least one failed/skipped
    # "failed"    — every model failed; no usable results
    evaluation_status: str | None = None

    # ML task type used for metric selection (mirrors model_selection.task_type)
    task_type: str | None = None

    # Primary metric used for ranking all models
    # e.g. "roc_auc", "f1", "r2"
    primary_metric: str | None = None

    # Serialisable per-model evaluation records (ModelEvaluationRecord dicts)
    # Ordered by rank ascending (rank 1 = best model)
    evaluated_models: list[dict[str, Any]] = Field(default_factory=list)

    # Models that failed evaluation or had no estimator available
    failed_models: list[dict[str, Any]] = Field(default_factory=list)

    # Name and identifier of the best-performing model
    best_model_name: str | None = None
    best_model_identifier: str | None = None

    # All computed metrics for the best model
    best_model_metrics: dict[str, float] = Field(default_factory=dict)

    # Side-by-side metric comparison for all evaluated models.
    # Each entry: {"model_name": str, "rank": int, **metric_values}
    comparison_table: list[dict[str, Any]] = Field(default_factory=list)

    # Visualization-ready data structures (JSON-serializable dicts of lists)
    # Keyed by model_name, then by chart type:
    # {
    #   "Random Forest Classifier": {
    #     "confusion_matrix": {"labels": [...], "matrix": [[...], ...]},
    #     "roc_curve": {"fpr": [...], "tpr": [...], "auc": float},
    #     "pr_curve": {"precision": [...], "recall": [...], "avg_precision": float},
    #     "feature_importance": {"features": [...], "importances": [...]},
    #   },
    #   ...
    # }
    visualization_data: dict[str, Any] = Field(default_factory=dict)

    # LLM-generated narrative (None when no API key is available)
    narrative: str | None = None

    # Per-field structured narrative (from EvaluationNarrativeOutput)
    narrative_structured: dict[str, Any] = Field(default_factory=dict)

    # Wall-clock time for the entire evaluation step (seconds)
    total_execution_time_seconds: float = 0.0

    # Collected error messages from failed evaluations
    errors: list[str] = Field(default_factory=list)

    # Non-blocking warnings
    warnings: list[str] = Field(default_factory=list)

    # Short human-readable summary for logging and API display
    summary: str | None = None
