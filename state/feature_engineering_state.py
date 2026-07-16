"""
State used by the Feature Engineering Agent.
"""

from typing import Any

from pydantic import Field

from .base_state import BaseState


class FeatureEngineeringState(BaseState):
    """
    Stores the results of feature engineering.
    """

    # Whether feature engineering completed successfully
    is_completed: bool = False

    # New columns created (e.g. extracted title, family size), in order applied
    derived_features: list[str] = Field(
        default_factory=list
    )

    # Columns dropped (ids, constant columns, user/LLM specified, etc.)
    dropped_columns: list[str] = Field(
        default_factory=list
    )

    # Columns whose missing values were imputed, mapped to the strategy used
    imputed_columns: dict[str, str] = Field(
        default_factory=dict
    )

    # Categorical columns that were encoded, mapped to the method used
    encoded_columns: dict[str, str] = Field(
        default_factory=dict
    )

    # Numerical columns that were scaled
    scaled_columns: list[str] = Field(
        default_factory=list
    )

    # Numerical columns where outliers were capped/removed
    outlier_treated_columns: list[str] = Field(
        default_factory=list
    )

    # Method used for outlier treatment, if any
    outlier_method: str | None = None

    # Number of rows removed as a result of feature engineering (e.g. dropna)
    rows_removed: int = 0

    # Final list of feature columns after transformation (excludes target)
    final_feature_columns: list[str] = Field(
        default_factory=list
    )

    # Shape of the dataset after feature engineering
    final_shape: dict[str, int] = Field(
        default_factory=dict
    )

    # High level list of transformations applied, in order
    transformations_applied: list[str] = Field(
        default_factory=list
    )

    # Any warnings generated during feature engineering
    warnings: list[str] = Field(
        default_factory=list
    )

    # Free-form extra info bag
    extra: dict[str, Any] = Field(
        default_factory=dict
    )

    # Summary of feature engineering
    summary: str | None = None