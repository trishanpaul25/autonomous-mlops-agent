"""
schemas/dataset_resolver_schema.py

Schema used by the Dataset Resolver Agent.

The Dataset Resolver Agent analyzes the user's prompt and
returns a structured response describing where the dataset
comes from and whether additional clarification is required.
"""

from typing import Literal

from pydantic import BaseModel, Field


class DatasetResolverOutput(BaseModel):
    """
    Structured output returned by the Dataset Resolver Agent.
    """
    source_type: Literal[
        "csv",
        "excel",
        "json",
        "url",
        "zip",
        "database",
        "kaggle",
        "builtin",
    ] = Field(
        ...,
        description="Type of dataset source."
    )
    source: str = Field(
        ...,
        description="Dataset path, URL, database table name, Kaggle dataset identifier, or built-in dataset name."
    )
    dataset_name: str | None = Field(
        default=None,
        description="Human-readable dataset name if available."
    )
    reasoning: str = Field(
        ...,
        description="Short explanation of how the dataset source was identified."
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score assigned by the LLM (0 to 1)."
    )
    needs_clarification: bool = Field(
        default=False,
        description="Whether the user needs to provide additional information."
    )

    clarification_question: str | None = Field(
        default=None,
        description="Question that should be shown to the user if clarification is required."
    )