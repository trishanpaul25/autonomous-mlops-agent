"""
State used by the Hyperparameter Optimization Agent.

Stores every artifact produced during the HPO step:
  - Optimized estimator objects (arbitrary Python objects)
  - Per-model optimization records (timing, best params, status)
  - Aggregate statistics and warnings

Design note on storage
----------------------
Optimized estimator objects are stored in `optimized_model_objects` as a
plain dict keyed by `model_identifier`. Because BaseState sets
`arbitrary_types_allowed=True`, Pydantic accepts the sklearn/xgboost
estimator objects without serialization.

This state is written once by the HyperparameterOptimizationAgent and
is intended to be consumed read-only by the downstream Evaluation Agent.

The Evaluation Agent should prefer `optimized_model_objects` over
`state.model_training.trained_model_objects`. If a model was not
optimized (failed or skipped), the Evaluation Agent falls back to
the original trained object from ModelTrainingState.
"""

from typing import Any

from pydantic import Field

from .base_state import BaseState


class HyperparameterOptimizationState(BaseState):
    """
    Stores the results of the hyperparameter optimization step.

    Written once by the HyperparameterOptimizationAgent and consumed
    read-only by downstream agents (Evaluation, Registry).
    """

    # Whether HPO completed (at least one model optimized or gracefully skipped)
    is_completed: bool = False

    # Overall optimization outcome
    # "completed"  — all candidate models optimized successfully
    # "partial"    — at least one succeeded, at least one failed/skipped
    # "failed"     — every model failed; no optimized estimators produced
    # "skipped"    — entire HPO step skipped (e.g. unsupported task type)
    optimization_status: str | None = None

    # Scoring metric used uniformly across all model searches
    # (e.g. "roc_auc", "f1_weighted", "r2")
    scoring_metric: str | None = None

    # Optimized fitted estimator objects
    # Key: model_identifier (e.g. "hpo_random_forest_classifier_a3f1b2c0")
    # Value: fitted estimator object refitted with best hyperparameters
    optimized_model_objects: dict[str, Any] = Field(default_factory=dict)

    # Serialisable optimization records for successfully optimized models
    # Each entry is an OptimizedModelRecord dict
    optimized_models: list[dict[str, Any]] = Field(default_factory=list)

    # Models that failed or were skipped during HPO
    # Same shape as optimized_models
    failed_models: list[dict[str, Any]] = Field(default_factory=list)

    # Name of the model with the best cross-validation score
    best_overall_model_name: str | None = None

    # The highest best_score achieved across all optimized models
    best_overall_score: float = 0.0

    # Wall-clock time for the entire HPO step (seconds)
    total_execution_time_seconds: float = 0.0

    # Collected error messages from all failed optimizations
    errors: list[str] = Field(default_factory=list)

    # Non-blocking warnings (e.g. search space not defined, strategy fallback)
    warnings: list[str] = Field(default_factory=list)

    # Short human-readable summary for logging and API display
    summary: str | None = None
