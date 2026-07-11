"""
State used by the Validation Agent.
"""

from typing import Any

from pydantic import Field

from .base_state import BaseState


class ValidationState(BaseState):
    """
    Stores the results of dataset validation.
    """

    # Whether the dataset passed validation
    is_valid: bool = False

    # Missing values per column
    missing_values: dict[str, int] = Field(
        default_factory=dict
    )

    # Number of duplicate rows
    duplicate_rows: int = 0

    # Data type of each column
    data_types: dict[str, str] = Field(
        default_factory=dict
    )

    # Target column detected (if any)
    target_column: str | None = None

    # Classification / Regression / Clustering
    problem_type: str | None = None

    # Any warnings generated during validation
    warnings: list[str] = Field(
        default_factory=list
    )

    # Summary of validation
    summary: str | None = None