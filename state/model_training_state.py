"""
State used by the Model Training Agent.

Stores every artifact produced during the training step:
  - Train/test split metadata and data arrays
  - Fitted model objects (arbitrary Python objects)
  - Per-model training records (timing, status, errors)
  - Aggregate statistics and warnings

Design note on storage
----------------------
Fitted model objects are stored in `trained_model_objects` as a plain
dict keyed by `model_identifier`. Because BaseState sets
`arbitrary_types_allowed=True`, Pydantic accepts the sklearn/xgboost
estimator objects without serialization.

The split arrays (X_train, X_test, y_train, y_test) are stored as
Python lists so they remain JSON-serialisable when the state is
dumped to the API layer. Downstream agents reconstruct numpy arrays
via `np.array(state.model_training.X_train)`.
"""

from typing import Any

from pydantic import Field

from .base_state import BaseState


class ModelTrainingState(BaseState):
    """
    Stores the results of the model training step.

    Written once by the ModelTrainingAgent and consumed read-only by
    downstream agents (Evaluation, HPO, Registry).
    """

    # Whether training completed without a total failure
    is_completed: bool = False

    # Overall training outcome
    # "completed"  — all candidate models trained successfully
    # "partial"    — at least one succeeded, at least one failed
    # "failed"     — every model failed; pipeline should halt
    training_status: str | None = None
    # Fraction of data held out for testing
    test_size: float = 0.2

    # Random seed used for reproducibility
    random_state: int = 42

    # Whether stratified splitting was applied (classification only)
    stratified: bool = False

    # Number of samples in each split
    train_samples: int = 0
    test_samples: int = 0

    # Feature and target column names used for the split
    feature_columns: list[str] = Field(default_factory=list)
    target_column: str | None = None

    # Split data arrays (stored as lists for JSON serialisability)
    # Downstream agents call np.array(state.model_training.X_train)

    X_train: list | None = None
    X_test: list | None = None
    y_train: list | None = None
    y_test: list | None = None

    # Fitted model objects
    # Key: model_identifier (e.g. "random_forest_classifier_a3f1b2c0")
    # Value: fitted estimator object (sklearn, xgboost, lightgbm, etc.)

    trained_model_objects: dict[str, Any] = Field(default_factory=dict)

    # Serialisable training records
    # Successfully trained models; each entry is a TrainedModelRecord dict
    trained_models: list[dict[str, Any]] = Field(default_factory=list)

    # Models that failed to train; same shape as trained_models
    failed_models: list[dict[str, Any]] = Field(default_factory=list)

    # Wall-clock time for the entire training step (seconds)
    total_execution_time_seconds: float = 0.0

    # Collected error messages from all failed models
    errors: list[str] = Field(default_factory=list)

    # Non-blocking warnings (e.g. library not installed, split skipped)
    warnings: list[str] = Field(default_factory=list)

    # Short human-readable summary for logging and API display
    summary: str | None = None
