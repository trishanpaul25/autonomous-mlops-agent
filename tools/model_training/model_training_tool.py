"""
Model Training Tool.

Applies a ModelTrainingOutput (built by ModelTrainingAgent from the
list of TrainingResults) to the PipelineState by populating
ModelTrainingState's serialisable fields.

This tool is a pure writer. It does not make any decisions,
train any models, or perform any IO.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from schemas.model_training_schema import ModelTrainingOutput
from tools.base_tool import BaseTool
from tools.model_training.model_trainer import TrainingResult
from utils.logger import logger

if TYPE_CHECKING:
    from state.pipeline_state import PipelineState


class ModelTrainingTool(BaseTool):
    """
    Writes the structured model training result into PipelineState.

    execute() transfers data from:
      - ModelTrainingOutput   → ModelTrainingState (serialisable fields)
      - list[TrainingResult]  → ModelTrainingState.trained_model_objects (fitted objects)
    """

    def execute(
        self,
        state: "PipelineState",
        output: ModelTrainingOutput,
        training_results: list[TrainingResult],
    ) -> "PipelineState":
        """
        Populate ModelTrainingState from a ModelTrainingOutput and the
        raw TrainingResult list (which carries fitted model objects).

        Parameters
        ----------
        state : PipelineState
            Shared pipeline state to update.
        output : ModelTrainingOutput
            Structured aggregate result from ModelTrainingAgent.
        training_results : list[TrainingResult]
            Raw results containing fitted model objects. These are NOT
            part of ModelTrainingOutput because Pydantic cannot serialise
            arbitrary estimator objects.

        Returns
        -------
        PipelineState
            Updated state with model_training fully populated.
        """
        mt = state.model_training
        mt.training_status = output.training_status
        mt.total_execution_time_seconds = output.total_execution_time_seconds
        mt.errors = output.errors
        mt.trained_models = [
            record.model_dump()
            for record in output.trained_models
        ]

        mt.failed_models = [
            record.model_dump()
            for record in output.failed_models
        ]
        for result in training_results:
            if result.status == "success" and result.fitted_model is not None:
                mt.trained_model_objects[result.model_identifier] = (
                    result.fitted_model
                )
        n_success = len(output.trained_models)
        n_failed = len(output.failed_models)

        mt.summary = (
            f"Training {output.training_status}. "
            f"Succeeded: {n_success} | Failed: {n_failed} | "
            f"Total time: {output.total_execution_time_seconds:.2f}s"
        )

        mt.is_completed = output.training_status in ("completed", "partial")

        state.model_training = mt

        logger.info(
            "ModelTrainingTool: state updated. "
            "Status: %s | Objects stored: %d | Summary: %s",
            mt.training_status,
            len(mt.trained_model_objects),
            mt.summary,
        )

        return state
