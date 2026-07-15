"""
Schema for the Hyperparameter Optimization Agent.

Defines the structured output contract that the
HyperparameterOptimizationAgent produces after completing the
optimization step. No LLM is involved — these schemas are populated
purely from deterministic scikit-learn CV search computation.

Design notes
------------
* OptimizedModelRecord captures the outcome for a single model:
  optimization_status, best_parameters, best_score, timing, a unique
  identifier, and human-readable notes.

* HPOptimizationOutput is the top-level result that the
  HyperparameterOptimizationTool writes into
  HyperparameterOptimizationState. The optimization_status Literal
  gives downstream agents an unambiguous signal about what happened.

* Both successfully optimized models and failed models use the same
  OptimizedModelRecord schema so downstream agents can iterate a
  uniform list.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class OptimizedModelRecord(BaseModel):
    """
    Captures the hyperparameter optimization outcome for a single ML model.

    Both successfully optimized models and failed models share this schema
    so the Evaluation Agent can iterate a uniform list regardless of outcome.
    """

    model_name: str = Field(
        ...,
        description=(
            "Human-readable name of the model "
            "(e.g. 'Random Forest Classifier')."
        ),
    )

    class_path: str = Field(
        ...,
        description=(
            "Fully-qualified importable class path used to instantiate "
            "the model (e.g. 'sklearn.ensemble.RandomForestClassifier')."
        ),
    )

    optimization_status: Literal["optimized", "failed", "skipped"] = Field(
        ...,
        description=(
            "Outcome of the optimization attempt. "
            "'optimized' — CV search completed and best params found. "
            "'failed'    — an exception occurred during the search. "
            "'skipped'   — model type or task type not supported for HPO."
        ),
    )

    best_parameters: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Best hyperparameter values found by the CV search. "
            "Empty dict for failed or skipped models."
        ),
    )

    best_score: float = Field(
        default=0.0,
        description=(
            "Best cross-validation score achieved with the best parameters. "
            "0.0 for failed or skipped models."
        ),
    )

    optimization_time_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Wall-clock duration of the optimization search in seconds. "
            "0.0 for failed or skipped models."
        ),
    )

    model_identifier: str = Field(
        default="",
        description=(
            "Unique key used to retrieve the optimized fitted estimator from "
            "HyperparameterOptimizationState.optimized_model_objects. "
            "Format: 'hpo_{sanitized_model_name}_{8-char hex}'. "
            "Empty string for failed or skipped models."
        ),
    )

    strategy_used: str = Field(
        default="",
        description=(
            "Name of the search strategy used "
            "(e.g. 'RandomizedSearchCV', 'GridSearchCV', 'Optuna'). "
            "Empty for failed or skipped models."
        ),
    )

    scoring_metric: str = Field(
        default="",
        description=(
            "Scoring metric used during the CV search "
            "(e.g. 'roc_auc', 'f1_weighted', 'r2'). "
            "Empty for skipped models."
        ),
    )

    notes: str = Field(
        default="",
        description=(
            "Human-readable notes: success summary, error message, "
            "or reason for skipping."
        ),
    )


class HPOptimizationOutput(BaseModel):
    """
    Top-level structured result produced by the Hyperparameter Optimization Agent.

    Written by HyperparameterOptimizationTool into
    HyperparameterOptimizationState after all candidate models have
    been processed.
    """

    optimization_status: Literal["completed", "partial", "failed", "skipped"] = Field(
        ...,
        description=(
            "Overall optimization outcome. "
            "'completed' — all candidates optimized successfully. "
            "'partial'   — at least one succeeded, at least one failed/skipped. "
            "'failed'    — all candidates failed; no optimized models produced. "
            "'skipped'   — task type not supported (e.g. clustering)."
        ),
    )

    optimized_models: list[OptimizedModelRecord] = Field(
        default_factory=list,
        description="Records for all successfully optimized models.",
    )

    failed_models: list[OptimizedModelRecord] = Field(
        default_factory=list,
        description=(
            "Records for all models that failed or were skipped during HPO."
        ),
    )

    best_overall_model_name: str = Field(
        default="",
        description=(
            "Name of the model that achieved the highest best_score "
            "across all successfully optimized models."
        ),
    )

    best_overall_score: float = Field(
        default=0.0,
        description="The highest best_score across all optimized models.",
    )

    scoring_metric: str = Field(
        default="",
        description=(
            "Scoring metric used uniformly across all model optimizations."
        ),
    )

    optimization_summary: str = Field(
        ...,
        description=(
            "Human-readable summary: how many models were attempted, "
            "optimized, and failed."
        ),
    )

    total_execution_time_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Total wall-clock time for the entire optimization step in seconds."
        ),
    )

    errors: list[str] = Field(
        default_factory=list,
        description="Collected error messages from all failed optimizations.",
    )
