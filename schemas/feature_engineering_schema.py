"""
Schema for the Feature Engineering Agent.

The LLM returns this structured output after analyzing the dataset
metadata and validation results. It NEVER performs the transformations
itself — it only decides which ones should be executed and how.
"""

from typing import Literal

from pydantic import BaseModel, Field


class DerivedFeatureSpec(BaseModel):
    """
    A single new column to engineer before dropping/imputing/encoding runs.

    This is intentionally dataset-agnostic: each operation is a generic
    transformation over one or more existing columns, not a hardcoded
    feature name. The same spec type can express "extract a title from
    a name column", "sum sibling+parent counts into a family size",
    "take the first letter of a mostly-missing category column", etc.
    """

    new_column: str = Field(
        ..., description="Name of the new feature column to create."
    )

    operation: Literal[
        "regex_extract",
        "first_char",
        "sum_columns",
        "ratio_columns",
        "log1p",
        "missing_flag",
        "equals_flag",
    ] = Field(..., description="Which derivation to apply.")

    source_columns: list[str] = Field(
        default_factory=list,
        description=(
            "Column(s) the derivation reads from. One column for "
            "regex_extract/first_char/log1p/missing_flag/equals_flag; "
            "exactly two for ratio_columns; two or more for sum_columns."
        ),
    )

    pattern: str | None = Field(
        default=None,
        description=(
            "Regex with exactly one capture group, used only for "
            "operation='regex_extract' (e.g. r',\\s*([A-Za-z]+)\\.' to pull "
            "a title out of a 'Last, Title. First' style name column)."
        ),
    )

    constant: float | None = Field(
        default=None,
        description=(
            "For sum_columns: a constant to add (e.g. 1, to count the "
            "person themselves in a family-size feature). For "
            "equals_flag: the value being compared against."
        ),
    )

    fillna: str | None = Field(
        default=None,
        description="Value to fill when the derivation yields null/no match (e.g. 'Rare', 'Unknown').",
    )


class FeatureEngineeringOutput(BaseModel):
    """
    Structured output returned by the Feature Engineering Agent.
    """

    derived_features: list[DerivedFeatureSpec] = Field(
        default_factory=list,
        description=(
            "New columns to engineer from existing ones BEFORE dropping, "
            "imputing, encoding, or scaling runs. Look for: free-text "
            "columns with a consistent extractable pattern (e.g. a title "
            "inside a name); pairs of numeric columns that represent "
            "counts which combine meaningfully (e.g. siblings + parents "
            "into family size); mostly-missing high-cardinality columns "
            "where a coarse extracted category still carries signal "
            "(e.g. the first letter of a cabin code) rather than dropping "
            "them outright. Do not invent columns that don't exist."
        ),
    )

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