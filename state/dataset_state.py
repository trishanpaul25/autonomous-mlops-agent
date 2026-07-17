"""
State used by Dataset Resolver and Data Ingestion Agent.
"""

from typing import Any
from uuid import UUID #add on to generate dataset id
import pandas as pd
from pydantic import Field

from .base_state import BaseState


class DatasetState(BaseState):
    """
    Stores all information related to the dataset.
    """
    source_type: str | None = None
    dataset_name: str | None = None
    dataset_id: UUID | None = None
    
    # Original uploaded/downloaded dataset
    dataset_path: str | None = None

    # Latest processed dataset
    current_dataset_path: str | None = None

    # DVC version / dataset version
    dataset_version: str | None = None

    dataframe: pd.DataFrame | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)

    target_column: str | None = None

    feature_columns: list[str] = Field(default_factory=list)

    problem_type: str | None = None

    num_rows: int | None = None

    num_columns: int | None = None

    loaded: bool = False