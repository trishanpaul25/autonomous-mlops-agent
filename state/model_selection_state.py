"""
State used by the Model Selection Agent.

Stores every artifact produced by the agent: the detected task type,
the primary model, all ranked candidate models, the reasoning, and a
snapshot of the dataset profile that was used to make the decision.
"""

from typing import Any

from pydantic import Field

from .base_state import BaseState


class ModelSelectionState(BaseState):
    """
    Stores the results of the model selection step.

    This state is written once by the ModelSelectionAgent and is
    intended to be consumed read-only by downstream agents
    (training, evaluation, registry).
    """
    is_completed: bool = False
    # Examples: "binary_classification", "multiclass_classification",
    #           "regression", "clustering", "time_series"
    task_type: str | None = None
    primary_model_name: str | None = None

    # Importable Python class path for the primary model

    primary_model_class_path: str | None = None

    # Library that provides the primary model (e.g. "sklearn", "xgboost")
    primary_model_library: str | None = None

    # Serialized list of CandidateModel dicts, ordered by rank (ascending)
    candidate_models: list[dict[str, Any]] = Field(
        default_factory=list
    )

    # Ordered list of model names from rank-1 (best) to rank-N
    ranking: list[str] = Field(
        default_factory=list
    )

    # Human-readable list of criteria used for ranking decisions
    ranking_criteria: list[str] = Field(
        default_factory=list
    )

    # Narrative explanation of why these models were chosen
    reasoning: str | None = None

    # Any assumptions made due to ambiguous or missing dataset information
    assumptions: list[str] = Field(
        default_factory=list
    )

    # Confidence score produced by the LLM or heuristic (0.0 – 1.0)
    confidence: float = 0.0

    # Snapshot of the DatasetProfile used for the selection decision.
    dataset_profile: dict[str, Any] = Field(
        default_factory=dict
    )

    # Non-blocking warnings generated during model selection
    warnings: list[str] = Field(
        default_factory=list
    )

    # Short human-readable summary for display / logging
    summary: str | None = None
