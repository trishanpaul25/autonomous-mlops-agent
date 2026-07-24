"""
Model Training Agent.

Responsible for training every candidate model recommended by the
Model Selection Agent and storing the fitted objects and training
metadata in ModelTrainingState.

The agent MUST NOT:
  - Perform feature engineering
  - Perform model selection or hyperparameter tuning
  - Evaluate model performance
  - Deploy models

Its sole responsibility is fitting models on the training split.

Architecture
------------
The agent orchestrates four internal utilities:

1. TrainTestSplitter  — produces X_train/X_test/y_train/y_test
2. ModelTrainer       — fits each candidate with per-model error isolation
3. ModelTrainingTool  — writes results into ModelTrainingState

No LLM is used. Training is a deterministic computation step.
"""

from __future__ import annotations

import time

from agents.base_agent import BaseAgent

from schemas.model_training_schema import (
    ModelTrainingOutput,
    TrainedModelRecord,
)

from state.pipeline_state import PipelineState

from tools.model_training.model_trainer import ModelTrainer, TrainingResult
from tools.model_training.model_training_tool import ModelTrainingTool
from tools.model_training.train_test_splitter import TrainTestSplitter

from utils.logger import logger

from server.core.constants import PipelineStatus

from server.services.progress_service import ProgressService
from server.services.progress_types import ProgressEventType

class ModelTrainingAgent(BaseAgent):
    """
    Trains every candidate model recommended by the Model Selection Agent.

    Constructor wires up:
      - TrainTestSplitter  for data preparation
      - ModelTrainer       for the fit loop with error isolation
      - ModelTrainingTool  for writing results back to PipelineState
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
            Fraction of data reserved for testing. Default 0.2.
        random_state : int
            Random seed for the train/test split. Default 42.
        """
        self.splitter = TrainTestSplitter(
            test_size=test_size,
            random_state=random_state,
        )
        self.trainer = ModelTrainer()
        self.tool = ModelTrainingTool()

    def run(self, state: PipelineState) -> PipelineState:
        """
        Execute the Model Training Agent.

        Execution flow
        --------------
        1. Guard: verify model selection completed.
        2. Guard: verify dataframe and feature columns exist.
        3. Perform train/test split (idempotent).
        4. Train all candidate models with per-model error isolation.
        5. Build ModelTrainingOutput from results.
        6. Write output to PipelineState via ModelTrainingTool.
        7. Fail the pipeline if no model succeeded.

        Parameters
        ----------
        state : PipelineState
            Shared pipeline state after Model Selection.

        Returns
        -------
        PipelineState
            Updated state with model_training fully populated.
        """
        pipeline_start = time.perf_counter()

        try:
            state.current_agent = "ModelTrainingAgent"

            logger.info("ModelTrainingAgent: starting.")

            ProgressService.emit(
                state.run_id,
                "🧠 Preparing model training...",
                ProgressEventType.THINKING,
            )

            state.logs.append("Model training started.")
            if not state.model_selection.is_completed:
                raise ValueError(
                    "ModelTrainingAgent requires Model Selection to have "
                    "completed successfully before it can run."
                )
            candidates = state.model_selection.candidate_models
            if not candidates:
                raise ValueError(
                    "ModelTrainingAgent: no candidate models found in "
                    "ModelSelectionState. Model Selection may not have "
                    "produced any recommendations."
                )
            if state.dataset.dataframe is None or len(state.dataset.dataframe) == 0:
                raise ValueError(
                    "ModelTrainingAgent: the dataset dataframe is missing or "
                    "empty. Ensure Data Ingestion and Feature Engineering "
                    "completed successfully."
                )

            if not state.feature_engineering.final_feature_columns:
                raise ValueError(
                    "ModelTrainingAgent: final_feature_columns is empty. "
                    "Feature Engineering must complete before Model Training."
                )
            logger.info(
                "ModelTrainingAgent: preparing train/test split "
                "(test_size=%.2f, random_state=%d)...",
                self.splitter.test_size,
                self.splitter.random_state,
            )

            ProgressService.emit(
                state.run_id,
                "✂️ Creating train/test split...",
                ProgressEventType.STEP,
            )

            state = self.splitter.split(state)

            state.logs.append(
                f"Train/test split complete. "
                f"Train: {state.model_training.train_samples} samples | "
                f"Test: {state.model_training.test_samples} samples | "
                f"Stratified: {state.model_training.stratified}"
            )
            logger.info(
                "ModelTrainingAgent: training %d candidate model(s)...",
                len(candidates),
            )

            ProgressService.emit(
                state.run_id,
                f"🚀 Training {len(candidates)} candidate model(s)...",
                ProgressEventType.STEP,
            )

            training_results: list[TrainingResult] = self.trainer.train_all(state)

            ProgressService.emit(
                state.run_id,
                "📊 Collecting training results...",
                ProgressEventType.THINKING,
            )
            
            total_elapsed = time.perf_counter() - pipeline_start

            succeeded = [r for r in training_results if r.status == "success"]
            failed = [r for r in training_results if r.status != "success"]
            all_errors = [r.error for r in failed if r.error]

            # Determine overall training status
            if len(succeeded) == len(training_results):
                training_status = "completed"
            elif len(succeeded) > 0:
                training_status = "partial"
            else:
                training_status = "failed"

            # Build Pydantic records
            trained_records = [
                ModelTrainer.to_trained_model_record(r) for r in succeeded
            ]
            failed_records = [
                ModelTrainer.to_trained_model_record(r) for r in failed
            ]

            summary = (
                f"Trained {len(succeeded)}/{len(training_results)} models "
                f"in {total_elapsed:.2f}s. "
                f"Primary model: {state.model_selection.primary_model_name}."
            )

            ProgressService.emit(
                state.run_id,
                "📝 Building training summary...",
                ProgressEventType.INFO,
            )

            output = ModelTrainingOutput(
                training_status=training_status,  # type: ignore[arg-type]
                trained_models=trained_records,
                failed_models=failed_records,
                training_summary=summary,
                total_execution_time_seconds=round(total_elapsed, 4),
                errors=all_errors,
            )
            state = self.tool.execute(
                state,
                output,
                training_results=training_results,
            )

            state.completed_steps.append("Model Training")

            ProgressService.emit(
                            state.run_id,
                            "💾 Saving trained model metadata...",
                            ProgressEventType.INFO,
                        )
            
            logger.info(
                "ModelTrainingAgent: finished. "
                "Status: %s | Succeeded: %d | Failed: %d | "
                "Total time: %.2fs",
                training_status,
                len(succeeded),
                len(failed),
                total_elapsed,
            )

            state.logs.append(
                f"Model training finished. "
                f"Status: {training_status} | "
                f"Succeeded: {len(succeeded)} | "
                f"Failed: {len(failed)} | "
                f"Time: {total_elapsed:.2f}s"
            )
            if training_status == "failed":
                state.status = PipelineStatus.FAILED
                state.error = (
                    "All candidate models failed to train. "
                    f"Errors: {'; '.join(all_errors)}"
                )
                logger.error(
                    "ModelTrainingAgent: all models failed. Pipeline halting."
                )

            ProgressService.emit(
                state.run_id,
                "✅ Model training completed.",
                ProgressEventType.SUCCESS,
            )

            return state

        except Exception as e:

            state.status = PipelineStatus.FAILED
            state.error = str(e)

            logger.error(
                "ModelTrainingAgent: unrecoverable error — %s",
                e,
                exc_info=True,
            )

            state.logs.append(f"Model training failed: {e}")

            ProgressService.emit(
                state.run_id,
                f"❌ Model training failed: {e}",
                ProgressEventType.ERROR,
            )

            return state
