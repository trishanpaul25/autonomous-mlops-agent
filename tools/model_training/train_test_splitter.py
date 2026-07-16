"""
Train/Test Splitter for the Model Training Agent.

Reads the processed dataframe from PipelineState and produces
train/test splits that are stored back into ModelTrainingState.

Design notes
------------
* Idempotent: if X_train already exists in state the splitter is a no-op,
  allowing the agent to be safely re-run without re-splitting.
* Stratification is applied automatically for classification tasks
  (binary_classification, multiclass_classification) when the target
  has sufficient samples per class.
* All arrays are stored as Python lists (JSON-serialisable).
  Downstream agents reconstruct numpy arrays via np.array(state.model_training.X_train).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.logger import logger

if TYPE_CHECKING:
    from state.pipeline_state import PipelineState


# Task types that should use stratified splitting
_CLASSIFICATION_TASK_TYPES = {
    "binary_classification",
    "multiclass_classification",
    "classification",
}

# Minimum samples per class required to enable stratification.
# Below this threshold we fall back to plain random splitting.
_MIN_SAMPLES_PER_CLASS_FOR_STRATIFY = 2


class TrainTestSplitter:
    """
    Prepares train/test splits from the pipeline dataframe.

    The splitter is a pure data transformation utility — it does not
    modify any model objects or pipeline configuration.
    """

    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        """
        Parameters
        ----------
        test_size : float
            Fraction of data to reserve for testing. Default 0.2 (20%).
        random_state : int
            Random seed for reproducibility. Default 42.
        """
        self.test_size = test_size
        self.random_state = random_state

    def split(self, state: "PipelineState") -> "PipelineState":
        """
        Perform the train/test split and write results into
        ModelTrainingState.

        If X_train is already populated (e.g. the agent is being re-run),
        this method is a no-op and returns state unchanged.

        Parameters
        ----------
        state : PipelineState
            Shared pipeline state after Model Selection.

        Returns
        -------
        PipelineState
            Updated state with X_train, X_test, y_train, y_test populated.

        Raises
        ------
        ValueError
            If the dataframe, feature columns, or target column are missing.
        """
        mt = state.model_training

        # Idempotency guard
        if mt.X_train is not None:
            logger.info(
                "TrainTestSplitter: split already exists (%d train / %d test). Skipping.",
                mt.train_samples,
                mt.test_samples,
            )
            return state
        df: pd.DataFrame | None = state.dataset.dataframe

        if df is None or len(df) == 0:
            raise ValueError(
                "TrainTestSplitter: dataframe is missing or empty. "
                "Ensure the Data Ingestion and Feature Engineering agents "
                "have completed successfully."
            )

        feature_columns: list[str] = state.feature_engineering.final_feature_columns
        target_column: str | None = state.validation.target_column

        if not feature_columns:
            raise ValueError(
                "TrainTestSplitter: no feature columns found. "
                "Feature Engineering must complete before Model Training."
            )

        if target_column is None:
            # Clustering tasks have no target — split features only
            task_type = state.model_selection.task_type or ""
            if "clustering" not in task_type.lower():
                raise ValueError(
                    "TrainTestSplitter: target column is None but task type "
                    f"is '{task_type}'. A target column is required for "
                    "supervised learning tasks."
                )
        missing_cols = [c for c in feature_columns if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"TrainTestSplitter: feature columns {missing_cols} are not "
                "present in the dataframe. Verify feature engineering output."
            )
        X: np.ndarray = df[feature_columns].values

        if target_column is not None and target_column in df.columns:
            y: np.ndarray = df[target_column].values
        else:
            y = np.zeros(len(df))
        task_type = (state.model_selection.task_type or "").lower()
        stratify = None
        use_stratify = task_type in _CLASSIFICATION_TASK_TYPES

        if use_stratify:
            # Check minimum class frequency — sklearn requires >= 2 per class
            unique, counts = np.unique(y, return_counts=True)
            min_count = int(counts.min()) if len(counts) > 0 else 0

            if min_count < _MIN_SAMPLES_PER_CLASS_FOR_STRATIFY:
                warning = (
                    f"TrainTestSplitter: cannot stratify — minority class has "
                    f"only {min_count} sample(s). Falling back to plain split."
                )
                logger.warning(warning)
                state.model_training.warnings.append(warning)
                use_stratify = False
            else:
                stratify = y
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify,
        )
        mt.X_train = X_train.tolist()
        mt.X_test = X_test.tolist()
        mt.y_train = y_train.tolist()
        mt.y_test = y_test.tolist()
        mt.feature_columns = feature_columns
        mt.target_column = target_column
        mt.test_size = self.test_size
        mt.random_state = self.random_state
        mt.stratified = use_stratify
        mt.train_samples = len(X_train)
        mt.test_samples = len(X_test)

        state.model_training = mt

        logger.info(
            "TrainTestSplitter: split complete. "
            "Train: %d | Test: %d | Stratified: %s | Features: %d",
            mt.train_samples,
            mt.test_samples,
            mt.stratified,
            len(feature_columns),
        )

        return state
