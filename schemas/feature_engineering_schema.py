"""
Schema for the Feature Engineering Agent.

The LLM returns this structured output after analyzing the dataset
metadata and validation results. It NEVER performs the transformations
itself — it only decides which ones should be executed and how.
"""

from typing import Literal

from pydantic import BaseModel, Field


class FeatureEngineeringOutput(BaseModel):
    """
    Structured output returned by the Feature Engineering Agent.
    """

    drop_columns: list[str] = Field(
        default_factory=list,
        description=(
            "Columns that should be dropped before modeling "
            "(e.g. IDs, free-text, constant, or leakage columns). "
            "Never include the target column."
        )
    )

    handle_missing_values: bool = Field(
        default=True,
        description="Whether missing values should be imputed."
    )

    numeric_missing_strategy: Literal[
        "mean",
        "median",
        "constant",
        "drop",
    ] = Field(
        default="mean",
        description="Imputation strategy for numeric columns with missing values."
    )

    categorical_missing_strategy: Literal[
        "mode",
        "constant",
        "drop",
    ] = Field(
        default="mode",
        description="Imputation strategy for categorical columns with missing values."
    )

    handle_outliers: bool = Field(
        default=False,
        description="Whether outliers in numeric columns should be treated."
    )

    outlier_method: Literal[
        "iqr",
        "zscore",
        "none",
    ] = Field(
        default="none",
        description="Method used to detect and cap outliers in numeric columns."
    )

    encode_categorical: bool = Field(
        default=True,
        description="Whether categorical columns should be encoded."
    )

    encoding_method: Literal[
        "onehot",
        "label",
    ] = Field(
        default="onehot",
        description="Encoding method for categorical columns."
    )

    scale_numerical: bool = Field(
        default=True,
        description="Whether numeric columns should be scaled."
    )

    scaling_method: Literal[
        "standard",
        "minmax",
        "robust",
        "none",
    ] = Field(
        default="standard",
        description="Scaling method for numeric columns."
    )

    reasoning: str = Field(
        ...,
        description="Explanation of why these feature engineering steps were selected."
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
