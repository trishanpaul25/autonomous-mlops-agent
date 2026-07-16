"""
Schema for the Model Selection Agent.

The LLM returns a ModelSelectionOutput object after analyzing the
dataset profile. The agent NEVER trains models, tunes hyperparameters,
or evaluates performance — its only responsibility is recommending
the most appropriate algorithms for the dataset.

Design notes
------------
* CandidateModel captures all metadata for a single algorithm
  recommendation, including a suitability score, strengths, and
  known limitations for this specific dataset context.

* ModelSelectionOutput is the top-level structured return that the
  LLM produces. The task_type field uses a granular Literal so
  downstream agents can branch without additional inference.
"""

from typing import Literal

from pydantic import BaseModel, Field

TaskTypeLiteral = Literal[
    "binary_classification",
    "multiclass_classification",
    "regression",
    "clustering",
    "time_series",
]


class CandidateModel(BaseModel):
    """
    Represents a single algorithm recommended by the Model Selection Agent.
    """

    name: str = Field(
        ...,
        description=(
            "Human-readable name of the algorithm. "
            "Example: 'Random Forest Classifier'."
        ),
    )

    library: str = Field(
        ...,
        description=(
            "Python library that provides this algorithm. "
            "Example: 'sklearn', 'xgboost', 'lightgbm'."
        ),
    )

    class_path: str = Field(
        ...,
        description=(
            "Fully qualified importable class path. "
            "Example: 'sklearn.ensemble.RandomForestClassifier'."
        ),
    )

    rank: int = Field(
        ...,
        ge=1,
        description=(
            "Rank among all candidates. 1 is the best recommendation."
        ),
    )

    suitability_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Estimated suitability for this specific dataset context, "
            "on a scale from 0.0 (poor fit) to 1.0 (ideal fit)."
        ),
    )

    rationale: str = Field(
        ...,
        description=(
            "Concise explanation of why this model is appropriate "
            "for the detected task type and dataset characteristics."
        ),
    )

    strengths: list[str] = Field(
        default_factory=list,
        description=(
            "Key advantages of this model given the current dataset context."
        ),
    )

    limitations: list[str] = Field(
        default_factory=list,
        description=(
            "Known weaknesses or risks of this model given the current "
            "dataset context."
        ),
    )
class ModelSelectionOutput(BaseModel):
    """
    Structured output returned by the Model Selection Agent.

    The LLM is constrained to return exactly this shape so results
    are always machine-readable and can be stored directly into
    ModelSelectionState without post-processing.
    """

    task_type: TaskTypeLiteral = Field(
        ...,
        description=(
            "The specific ML task type inferred from the dataset. "
            "Choices: binary_classification, multiclass_classification, "
            "regression, clustering, time_series."
        ),
    )

    primary_model: CandidateModel = Field(
        ...,
        description=(
            "The single best model recommendation for this dataset. "
            "Must also appear in candidate_models with rank=1."
        ),
    )

    candidate_models: list[CandidateModel] = Field(
        ...,
        min_length=1,
        description=(
            "All recommended candidate models ordered by rank ascending. "
            "Include between 3 and 7 candidates."
        ),
    )

    ranking_criteria: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "Ordered list of criteria used to rank the candidate models. "
            "Example: ['dataset size', 'interpretability', 'class imbalance handling']."
        ),
    )

    reasoning: str = Field(
        ...,
        description=(
            "Detailed narrative explaining the task type detection, "
            "why these specific models were selected, and how they were ranked."
        ),
    )

    assumptions: list[str] = Field(
        default_factory=list,
        description=(
            "Explicit assumptions made due to ambiguous or incomplete "
            "dataset information. Empty list if none."
        ),
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score of the overall recommendation (0.0 – 1.0).",
    )

    needs_clarification: bool = Field(
        default=False,
        description=(
            "Set to True if critical information is missing and the agent "
            "cannot make a reliable recommendation without user input."
        ),
    )

    clarification_question: str | None = Field(
        default=None,
        description=(
            "Specific question to ask the user if needs_clarification is True. "
            "Must be None when needs_clarification is False."
        ),
    )
