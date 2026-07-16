"""
Schema for the Model Training Agent.

Defines the structured output contract that the ModelTrainingAgent
produces after completing the training step. No LLM is involved —
these schemas are populated purely from deterministic computation.

Design notes
------------
* TrainedModelRecord captures the outcome for a single model:
  success/failure, timing, a unique identifier, and human-readable notes.

* ModelTrainingOutput is the top-level result that the ModelTrainingTool
  writes into ModelTrainingState. The training_status Literal gives
  downstream agents an unambiguous signal about what happened.
"""

from typing import Literal

from pydantic import BaseModel, Field


class TrainedModelRecord(BaseModel):
    """
    Captures the training outcome for a single ML model.

    Both successfully trained models and failed models use this same
    schema so downstream agents can iterate a uniform list.
    """

    model_name: str = Field(
        ...,
        description="Human-readable name of the model (e.g. 'Random Forest Classifier').",
    )

    class_path: str = Field(
        ...,
        description=(
            "Fully-qualified importable class path used to instantiate "
            "the model (e.g. 'sklearn.ensemble.RandomForestClassifier')."
        ),
    )

    status: Literal["success", "failed", "skipped"] = Field(
        ...,
        description=(
            "Outcome of the training attempt. "
            "'success' — model was fitted without error. "
            "'failed'  — an exception occurred during instantiation or fitting. "
            "'skipped' — model was intentionally not trained (e.g. unsupported task)."
        ),
    )

    training_time_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Wall-clock training duration in seconds. 0.0 for failed/skipped models.",
    )

    model_identifier: str = Field(
        ...,
        description=(
            "Unique key used to retrieve the fitted model object from "
            "ModelTrainingState.trained_model_objects. "
            "Format: '{sanitized_model_name}_{8-char hex}'. "
            "Empty string for failed/skipped models."
        ),
    )

    notes: str = Field(
        default="",
        description=(
            "Human-readable notes: success summary, error message, "
            "or reason for skipping."
        ),
    )


class ModelTrainingOutput(BaseModel):
    """
    Top-level structured result produced by the Model Training Agent.

    Written by ModelTrainingTool into ModelTrainingState after all
    candidate models have been processed.
    """

    training_status: Literal["completed", "partial", "failed"] = Field(
        ...,
        description=(
            "Overall training outcome. "
            "'completed' — all candidates trained successfully. "
            "'partial'   — at least one succeeded, at least one failed. "
            "'failed'    — all candidates failed; no usable models produced."
        ),
    )

    trained_models: list[TrainedModelRecord] = Field(
        default_factory=list,
        description="Records for all successfully trained models.",
    )

    failed_models: list[TrainedModelRecord] = Field(
        default_factory=list,
        description="Records for all models that failed to train.",
    )

    training_summary: str = Field(
        ...,
        description=(
            "Human-readable summary of the training step: how many models "
            "were attempted, succeeded, and failed."
        ),
    )

    total_execution_time_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Total wall-clock time for the entire training step in seconds.",
    )

    errors: list[str] = Field(
        default_factory=list,
        description="Collected error messages from all failed models.",
    )
