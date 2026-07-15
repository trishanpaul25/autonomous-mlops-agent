"""
Schema for the Validation Agent.

The LLM returns this structured output after analyzing
the dataset metadata.
"""

from typing import Literal

from pydantic import BaseModel, Field


class ValidationOutput(BaseModel):
    """
    Structured output returned by the Validation Agent.
    """

    check_missing_values: bool = Field(
        default=True,
        description="Whether missing value analysis should be performed."
    )

    check_duplicates: bool = Field(
        default=True,
        description="Whether duplicate row detection should be performed."
    )

    check_data_types: bool = Field(
        default=True,
        description="Whether column data types should be analyzed."
    )

    detect_target_column: bool = Field(
        default=True,
        description="Whether the target column should be detected."
    )

    target_column: str | None = Field(
        default=None,
        description="The detected target column name inferred from the user prompt and dataset columns."
    )

    infer_problem_type: bool = Field(
        default=True,
        description="Whether the ML problem type should be inferred."
    )

    problem_type: Literal[
        "classification",
        "regression",
        "clustering",
        "unknown"
    ] = Field(
        default="unknown",
        description="Predicted machine learning problem type."
    )

    reasoning: str = Field(
        ...,
        description="Explanation of why these validation checks were selected."
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score of the LLM."
    )

    needs_clarification: bool = Field(
        default=False,
        description="Whether more information is required from the user."
    )

    clarification_question: str | None = Field(
        default=None,
        description="Question to ask the user if clarification is needed."
    )