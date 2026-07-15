"""
Schema for the Model Evaluation Agent.

Defines the structured output contracts produced by the evaluation step.
No LLM is used for metric computation — these schemas are populated from
deterministic scikit-learn calculations.

Design notes
------------
* ModelEvaluationRecord  — per-model evaluation outcome, metrics dict,
  rank, and visualization-ready data.

* ModelEvaluationOutput  — top-level result written to ModelEvaluationState
  by ModelEvaluationTool. Contains the ranked list, best model, comparison
  table, and aggregate metadata.

* EvaluationNarrativeOutput — optional structured LLM output that provides
  a human-readable interpretation of the metrics. The LLM receives the
  comparison table and produces plain-English text. It NEVER computes
  any metrics itself.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ModelEvaluationRecord(BaseModel):
    """
    Captures the evaluation outcome for a single ML model.

    Both successfully evaluated models and failed models use this schema
    so the Registry Agent can iterate a uniform list.
    """

    model_name: str = Field(
        ...,
        description=(
            "Human-readable name of the model "
            "(e.g. 'Random Forest Classifier')."
        ),
    )

    class_path: str = Field(
        default="",
        description=(
            "Fully-qualified importable class path of the model "
            "(e.g. 'sklearn.ensemble.RandomForestClassifier')."
        ),
    )

    model_identifier: str = Field(
        default="",
        description=(
            "Unique key used to retrieve the fitted estimator from "
            "HyperparameterOptimizationState.optimized_model_objects "
            "or ModelTrainingState.trained_model_objects."
        ),
    )

    evaluation_status: Literal["evaluated", "failed", "skipped"] = Field(
        ...,
        description=(
            "'evaluated' — prediction and metric computation succeeded. "
            "'failed'    — an exception occurred during evaluation. "
            "'skipped'   — no fitted estimator found in any store."
        ),
    )

    rank: int = Field(
        default=0,
        ge=0,
        description=(
            "Rank of this model among all evaluated models, ordered by the "
            "primary metric (1 = best). 0 for failed or skipped models."
        ),
    )

    metrics: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "All computed evaluation metrics for this model. "
            "Empty dict for failed or skipped models. "
            "Keys: 'accuracy', 'f1', 'roc_auc', 'r2', 'mae', etc."
        ),
    )

    primary_metric_name: str = Field(
        default="",
        description=(
            "Name of the metric used to rank this model "
            "(e.g. 'roc_auc' for binary classification, 'r2' for regression)."
        ),
    )

    primary_metric_value: float = Field(
        default=0.0,
        description=(
            "Value of the primary metric. Used for ranking. "
            "0.0 for failed or skipped models."
        ),
    )

    prediction_time_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Wall-clock time to generate predictions on the test set. "
            "0.0 for failed or skipped models."
        ),
    )

    notes: str = Field(
        default="",
        description=(
            "Human-readable notes: metric highlights, error message, "
            "or reason for skipping."
        ),
    )


class ModelEvaluationOutput(BaseModel):
    """
    Top-level structured result produced by the Model Evaluation Agent.

    Written by ModelEvaluationTool into ModelEvaluationState after all
    candidate models have been evaluated.
    """

    evaluation_status: Literal["completed", "partial", "failed"] = Field(
        ...,
        description=(
            "'completed' — all candidates evaluated successfully. "
            "'partial'   — at least one succeeded, at least one failed/skipped. "
            "'failed'    — all candidates failed; no usable evaluations produced."
        ),
    )

    evaluated_models: list[ModelEvaluationRecord] = Field(
        default_factory=list,
        description=(
            "Records for all successfully evaluated models, "
            "ordered by rank (rank 1 = best)."
        ),
    )

    failed_models: list[ModelEvaluationRecord] = Field(
        default_factory=list,
        description="Records for all models that failed or were skipped.",
    )

    best_model_name: str = Field(
        default="",
        description=(
            "Name of the model ranked 1st by the primary metric."
        ),
    )

    best_model_identifier: str = Field(
        default="",
        description=(
            "Identifier used to retrieve the best model's fitted estimator "
            "from HyperparameterOptimizationState or ModelTrainingState."
        ),
    )

    best_model_metrics: dict[str, float] = Field(
        default_factory=dict,
        description="All computed metrics for the best model.",
    )

    primary_metric: str = Field(
        default="",
        description=(
            "Metric used to rank all models "
            "(e.g. 'roc_auc', 'f1', 'r2')."
        ),
    )

    comparison_table: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Side-by-side metric comparison across all evaluated models. "
            "Each entry is a dict with 'model_name' and all metric values. "
            "Ordered by rank."
        ),
    )

    evaluation_summary: str = Field(
        ...,
        description=(
            "Human-readable summary: models attempted, succeeded, failed, "
            "and the best model with its primary metric value."
        ),
    )

    total_execution_time_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Total wall-clock time for the entire evaluation step in seconds."
        ),
    )

    errors: list[str] = Field(
        default_factory=list,
        description="Collected error messages from all failed evaluations.",
    )


class EvaluationNarrativeOutput(BaseModel):
    """
    Optional structured LLM output: a human-readable interpretation of
    the model evaluation results.

    The LLM receives the comparison table and produces structured text.
    It NEVER computes metrics — it only interprets existing numbers.
    """

    best_model_explanation: str = Field(
        ...,
        description=(
            "Why the best model outperformed the others, in plain English."
        ),
    )

    model_comparisons: list[str] = Field(
        default_factory=list,
        description=(
            "One sentence per model pair comparing their relative performance."
        ),
    )

    strengths_weaknesses: dict[str, dict[str, list[str]]] = Field(
        default_factory=dict,
        description=(
            "Per-model strengths and weaknesses based on the metric profile. "
            "Keys: model_name. Values: {'strengths': [...], 'weaknesses': [...]}."
        ),
    )

    business_summary: str = Field(
        ...,
        description=(
            "A non-technical, business-friendly summary of which model to "
            "deploy and why, suitable for a stakeholder report."
        ),
    )
