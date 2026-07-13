"""
Model Trainer for the Model Training Agent.

Executes the training loop across all candidate models.

Design Principles
-----------------
* Error isolation: each model is wrapped in its own try/except block.
  One model failure never stops the remaining models from training.
* Timing: wall-clock training duration is recorded per model using
  time.perf_counter() for sub-second precision.
* Unique identifiers: each successfully trained model receives a
  unique `model_identifier` so downstream agents can retrieve the
  exact fitted object from ModelTrainingState.trained_model_objects.
* Clustering support: K-Means and similar unsupervised models call
  fit(X) without a target array.
"""

from __future__ import annotations

import time
import uuid
import re
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import numpy as np

from schemas.model_training_schema import TrainedModelRecord
from tools.model_training.model_instantiator import ModelInstantiator
from utils.logger import logger

if TYPE_CHECKING:
    from state.pipeline_state import PipelineState


# Task types that use unsupervised fit(X) instead of fit(X, y)
_UNSUPERVISED_TASK_TYPES = {"clustering"}


@dataclass
class TrainingResult:
    """
    Internal result produced for each training attempt.

    This is an intermediate structure used within ModelTrainer.
    The ModelTrainingTool converts these to TrainedModelRecord objects.
    """
    model_name: str
    class_path: str
    status: str               # "success" | "failed" | "skipped"
    training_time_seconds: float = 0.0
    model_identifier: str = ""
    fitted_model: Any = None
    notes: str = ""
    error: str = ""


class ModelTrainer:
    """
    Trains all candidate models from ModelSelectionState.

    Each model is trained independently. Failures are caught and
    recorded without interrupting the remaining training loop.
    """

    def __init__(self) -> None:
        self.instantiator = ModelInstantiator()

    def train_all(self, state: "PipelineState") -> list[TrainingResult]:
        """
        Iterate over all candidate models and train each one.

        Parameters
        ----------
        state : PipelineState
            Shared pipeline state with split arrays ready in
            state.model_training.X_train / y_train.

        Returns
        -------
        list[TrainingResult]
            One result per candidate model, in rank order.
        """
        candidates: list[dict[str, Any]] = (
            state.model_selection.candidate_models
        )

        task_type: str = (state.model_selection.task_type or "").lower()
        is_unsupervised = task_type in _UNSUPERVISED_TASK_TYPES

        mt = state.model_training
        X_train = np.array(mt.X_train)
        y_train = np.array(mt.y_train) if not is_unsupervised else None

        results: list[TrainingResult] = []

        for candidate in candidates:
            model_name: str = candidate.get("name", "Unknown")
            class_path: str = candidate.get("class_path", "")
            rank: int = candidate.get("rank", 99)

            logger.info(
                "ModelTrainer: [rank %d] starting '%s' (%s)...",
                rank,
                model_name,
                class_path,
            )

            state.logs.append(f"Training started: {model_name}")

            result = self._train_single(
                model_name=model_name,
                class_path=class_path,
                X_train=X_train,
                y_train=y_train,
                is_unsupervised=is_unsupervised,
            )

            results.append(result)

            if result.status == "success":
                logger.info(
                    "ModelTrainer: '%s' trained in %.3fs. Identifier: %s",
                    model_name,
                    result.training_time_seconds,
                    result.model_identifier,
                )
                state.logs.append(
                    f"Training completed: {model_name} "
                    f"({result.training_time_seconds:.3f}s)"
                )
            else:
                logger.warning(
                    "ModelTrainer: '%s' failed — %s",
                    model_name,
                    result.error,
                )
                state.logs.append(
                    f"Training failed: {model_name} — {result.error}"
                )

        return results

    def _train_single(
        self,
        model_name: str,
        class_path: str,
        X_train: np.ndarray,
        y_train: np.ndarray | None,
        is_unsupervised: bool,
    ) -> TrainingResult:
        """
        Attempt to instantiate and fit a single model.

        All exceptions are caught and encoded into the TrainingResult
        so the caller's loop is never interrupted.

        Parameters
        ----------
        model_name : str
            Human-readable model name for logging.
        class_path : str
            Fully-qualified class path passed to ModelInstantiator.
        X_train : np.ndarray
            Training features array.
        y_train : np.ndarray | None
            Training target array; None for unsupervised models.
        is_unsupervised : bool
            When True, fit(X) is called without a target array.

        Returns
        -------
        TrainingResult
            Outcome of the training attempt.
        """
        if not class_path:
            return TrainingResult(
                model_name=model_name,
                class_path=class_path,
                status="skipped",
                notes="No class_path available — model was skipped.",
                error="class_path is empty",
            )

        start_time = time.perf_counter()

        try:
            model = self.instantiator.instantiate(class_path)
        except (ImportError, AttributeError, TypeError) as e:
            elapsed = time.perf_counter() - start_time
            error_msg = str(e)
            return TrainingResult(
                model_name=model_name,
                class_path=class_path,
                status="failed",
                training_time_seconds=round(elapsed, 4),
                notes=f"Instantiation failed: {error_msg}",
                error=error_msg,
            )

        try:
            if is_unsupervised:
                model.fit(X_train)
            else:
                model.fit(X_train, y_train)

        except Exception as e:
            elapsed = time.perf_counter() - start_time
            error_msg = str(e)
            return TrainingResult(
                model_name=model_name,
                class_path=class_path,
                status="failed",
                training_time_seconds=round(elapsed, 4),
                notes=f"Training (fit) failed: {error_msg}",
                error=error_msg,
            )

        elapsed = time.perf_counter() - start_time
        model_identifier = self._make_identifier(model_name)

        return TrainingResult(
            model_name=model_name,
            class_path=class_path,
            status="success",
            training_time_seconds=round(elapsed, 4),
            model_identifier=model_identifier,
            fitted_model=model,
            notes=(
                f"Trained successfully in {elapsed:.3f}s. "
                f"Identifier: {model_identifier}"
            ),
        )

    @staticmethod
    def _make_identifier(model_name: str) -> str:
        """
        Produce a unique, filesystem-safe identifier for a fitted model.

        Format: {sanitized_model_name}_{8-char hex suffix}

        Example: "random_forest_classifier_a3f1b2c0"
        """
        sanitized = re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")
        suffix = uuid.uuid4().hex[:8]
        return f"{sanitized}_{suffix}"

    @staticmethod
    def to_trained_model_record(result: TrainingResult) -> TrainedModelRecord:
        """
        Convert an internal TrainingResult to a Pydantic TrainedModelRecord.
        """
        return TrainedModelRecord(
            model_name=result.model_name,
            class_path=result.class_path,
            status=result.status,  # type: ignore[arg-type]
            training_time_seconds=result.training_time_seconds,
            model_identifier=result.model_identifier,
            notes=result.notes,
        )
