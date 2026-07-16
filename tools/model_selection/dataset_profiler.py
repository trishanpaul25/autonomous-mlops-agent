"""
Dataset Profiler for the Model Selection Agent.

Extracts quantitative characteristics from the current PipelineState
and returns a DatasetProfile dataclass. The profiler is a pure
read-only utility — it never modifies the state or the dataframe.

The profile is the single input consumed by both the LLM prompt builder
and the heuristic fallback ranker, ensuring both paths reason over the
same facts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

from tools.model_selection.model_registry import (
    TaskType,
    DatasetSizeCategory,
    categorise_dataset_size,
)

if TYPE_CHECKING:
    from state.pipeline_state import PipelineState
@dataclass
class DatasetProfile:
    """
    A snapshot of all dataset characteristics relevant to model selection.

    All attributes are derived from PipelineState and are immutable once
    the profile is created. Downstream code should treat this as read-only.
    """

    # Coarse task type hint from the Validation Agent
    # (may be refined to binary/multiclass by the profiler)
    task_type: TaskType

    # Target column name (None for clustering)
    target_column: str | None

    # Pandas dtype string of the target column (e.g. "int64", "object")
    target_dtype: str | None

    # Dataset dimensions after feature engineering
    num_rows: int
    num_feature_cols: int

    # Feature name lists after engineering
    numerical_features: list[str]
    categorical_features: list[str]

    # Class distribution mapping: class_label -> count
    # Empty dict for regression and clustering
    class_distribution: dict[str, int]

    # Ratio of the majority to minority class (None for non-classification)
    class_imbalance_ratio: float | None

    # Whether any missing values remain after feature engineering
    has_missing_after_engineering: bool

    # Human-readable size category
    dataset_size_category: DatasetSizeCategory

    # Data type of each column (from ValidationState)
    data_types: dict[str, str]

    # Transformations applied during feature engineering
    transformations_applied: list[str]

    # Columns encoded during feature engineering
    encoded_columns: dict[str, str]

    # Columns scaled during feature engineering
    scaled_columns: list[str]

    # Columns dropped during feature engineering
    dropped_columns: list[str]

    # Feature engineering summary text
    fe_summary: str | None

    # Extra metadata bag from DatasetState
    metadata: dict[str, Any] = field(default_factory=dict)

    # Non-blocking warnings produced during profiling
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialise the profile to a plain dict for storage in
        ModelSelectionState.dataset_profile.
        """
        return {
            "task_type": self.task_type.value,
            "target_column": self.target_column,
            "target_dtype": self.target_dtype,
            "num_rows": self.num_rows,
            "num_feature_cols": self.num_feature_cols,
            "numerical_features": self.numerical_features,
            "categorical_features": self.categorical_features,
            "class_distribution": self.class_distribution,
            "class_imbalance_ratio": self.class_imbalance_ratio,
            "has_missing_after_engineering": self.has_missing_after_engineering,
            "dataset_size_category": self.dataset_size_category.value,
            "data_types": self.data_types,
            "transformations_applied": self.transformations_applied,
            "encoded_columns": self.encoded_columns,
            "scaled_columns": self.scaled_columns,
            "dropped_columns": self.dropped_columns,
            "fe_summary": self.fe_summary,
            "warnings": self.warnings,
        }
class DatasetProfiler:
    """
    Reads PipelineState and produces a DatasetProfile.

    No state mutation occurs. All analysis is derived from:
      - state.dataset.dataframe        (the live DataFrame)
      - state.validation.*             (validation results)
      - state.feature_engineering.*   (feature engineering results)
    """

    # Number of unique target values at or below which we treat the
    # problem as classification even if the dtype is numeric
    _CLASSIFICATION_CARDINALITY_THRESHOLD: int = 20

    def build(self, state: "PipelineState") -> DatasetProfile:
        """
        Build and return a DatasetProfile from the current PipelineState.

        Parameters
        ----------
        state : PipelineState
            The shared pipeline state after Feature Engineering has run.

        Returns
        -------
        DatasetProfile
            Fully populated profile ready for the LLM prompt and
            heuristic ranker.
        """
        warnings: list[str] = []

        df = state.dataset.dataframe
        target_column = state.validation.target_column
        if df is not None:
            num_rows = len(df)
        else:
            num_rows = state.feature_engineering.final_shape.get("rows", 0)
            warnings.append(
                "DataFrame not available; using final_shape.rows for row count."
            )

        final_feature_cols: list[str] = state.feature_engineering.final_feature_columns
        num_feature_cols = len(final_feature_cols) if final_feature_cols else (
            state.feature_engineering.final_shape.get("cols", 0)
        )
        numerical_features: list[str] = []
        categorical_features: list[str] = []
        data_types: dict[str, str] = state.validation.data_types or {}

        if df is not None and final_feature_cols:
            for col in final_feature_cols:
                if col not in df.columns:
                    continue
                if pd.api.types.is_numeric_dtype(df[col]):
                    numerical_features.append(col)
                else:
                    categorical_features.append(col)
        else:
            # Fall back to ValidationState data_types
            for col, dtype in data_types.items():
                if col == target_column:
                    continue
                if dtype in ("int64", "float64", "int32", "float32", "int16"):
                    numerical_features.append(col)
                else:
                    categorical_features.append(col)
        coarse_type = state.validation.problem_type or "unknown"
        task_type = TaskType.from_validation_type(coarse_type)

        target_dtype: str | None = None
        class_distribution: dict[str, int] = {}
        class_imbalance_ratio: float | None = None

        if target_column and df is not None and target_column in df.columns:
            target_series = df[target_column]
            target_dtype = str(target_series.dtype)
            n_unique = target_series.nunique()

            if coarse_type == "classification":
                # Refine to binary vs. multiclass
                if n_unique == 2:
                    task_type = TaskType.BINARY_CLASSIFICATION
                else:
                    task_type = TaskType.MULTICLASS_CLASSIFICATION

                # Compute class distribution
                vc = target_series.value_counts()
                class_distribution = {
                    str(k): int(v) for k, v in vc.items()
                }

                # Imbalance ratio
                if len(vc) >= 2:
                    class_imbalance_ratio = round(
                        float(vc.iloc[0]) / float(vc.iloc[-1]), 2
                    )
                    if class_imbalance_ratio > 5.0:
                        warnings.append(
                            f"High class imbalance detected "
                            f"(majority/minority ratio = {class_imbalance_ratio:.1f}). "
                            "Consider imbalance-aware models or resampling."
                        )

            elif coarse_type == "regression":
                task_type = TaskType.REGRESSION
                # Sanity check: if very low cardinality, warn
                if n_unique <= self._CLASSIFICATION_CARDINALITY_THRESHOLD:
                    warnings.append(
                        f"Target column '{target_column}' has only {n_unique} "
                        "unique values but problem type is 'regression'. "
                        "Verify this is not actually a classification problem."
                    )

        elif target_column is None:
            task_type = TaskType.CLUSTERING
        has_missing_after_engineering = False
        if df is not None:
            missing_count = int(df.isnull().sum().sum())
            has_missing_after_engineering = missing_count > 0
            if has_missing_after_engineering:
                warnings.append(
                    f"Dataset still contains {missing_count} missing values "
                    "after feature engineering. Some models may be affected."
                )
        else:
            # Assume engineering handled missing values
            has_missing_after_engineering = False

        dataset_size_category = categorise_dataset_size(num_rows)

        if dataset_size_category == DatasetSizeCategory.TINY:
            warnings.append(
                f"Very small dataset ({num_rows} rows). Model selection will "
                "favour interpretable models to reduce overfitting risk."
            )

        fe = state.feature_engineering

        return DatasetProfile(
            task_type=task_type,
            target_column=target_column,
            target_dtype=target_dtype,
            num_rows=num_rows,
            num_feature_cols=num_feature_cols,
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            class_distribution=class_distribution,
            class_imbalance_ratio=class_imbalance_ratio,
            has_missing_after_engineering=has_missing_after_engineering,
            dataset_size_category=dataset_size_category,
            data_types=data_types,
            transformations_applied=fe.transformations_applied,
            encoded_columns=fe.encoded_columns,
            scaled_columns=fe.scaled_columns,
            dropped_columns=fe.dropped_columns,
            fe_summary=fe.summary,
            metadata=dict(state.dataset.metadata),
            warnings=warnings,
        )
