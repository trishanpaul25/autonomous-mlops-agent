"""
Model Evaluation Tool.

Transfers a ModelEvaluationOutput (built by ModelEvaluationAgent) and
the raw EvaluationResult list into ModelEvaluationState.

This tool is a pure writer. It makes no decisions, runs no evaluations,
and performs no IO. It mirrors ModelTrainingTool and
HyperparameterOptimizationTool exactly.

Responsibility boundary
-----------------------
This tool is responsible for:
  - Transferring ModelEvaluationOutput → ModelEvaluationState scalars
  - Writing serializable model records to evaluated_models / failed_models
  - Writing visualization_data from EvaluationResult objects
  - Building the comparison_table from evaluated model records
  - Setting is_completed and summary

It is NOT responsible for:
  - Running predictions
  - Computing metrics
  - Ranking models
  - Generating narrative
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from schemas.model_evaluation_schema import ModelEvaluationOutput
from tools.base_tool import BaseTool
from tools.model_evaluation.model_evaluator import EvaluationResult
from utils.logger import logger

if TYPE_CHECKING:
    from state.pipeline_state import PipelineState


class ModelEvaluationTool(BaseTool):
    """
    Writes the structured evaluation result into PipelineState.

    execute() transfers data from:
      - ModelEvaluationOutput  → ModelEvaluationState (serializable fields)
      - list[EvaluationResult] → ModelEvaluationState.visualization_data
    """

    def execute(
        self,
        state: "PipelineState",
        output: ModelEvaluationOutput,
        evaluation_results: list[EvaluationResult],
    ) -> "PipelineState":
        """
        Populate ModelEvaluationState from a ModelEvaluationOutput and
        the raw EvaluationResult list (which carries visualization data).

        Parameters
        ----------
        state : PipelineState
            Shared pipeline state to update.
        output : ModelEvaluationOutput
            Structured aggregate result from ModelEvaluationAgent.
        evaluation_results : list[EvaluationResult]
            Raw results containing visualization_data per model.

        Returns
        -------
        PipelineState
            Updated state with model_evaluation fully populated.
        """
        ev = state.model_evaluation
        ev.evaluation_status = output.evaluation_status
        ev.task_type = state.model_selection.task_type
        ev.primary_metric = output.primary_metric
        ev.best_model_name = output.best_model_name
        ev.best_model_identifier = output.best_model_identifier
        ev.best_model_metrics = output.best_model_metrics
        ev.total_execution_time_seconds = output.total_execution_time_seconds
        ev.errors = output.errors
        ev.comparison_table = output.comparison_table
        ev.evaluated_models = [
            record.model_dump()
            for record in output.evaluated_models
        ]
        ev.failed_models = [
            record.model_dump()
            for record in output.failed_models
        ]
        viz: dict[str, Any] = {}
        for result in evaluation_results:
            if result.status == "evaluated" and result.visualization_data:
                viz[result.model_name] = result.visualization_data
        ev.visualization_data = viz
        n_evaluated = len(output.evaluated_models)
        n_failed = len(output.failed_models)
        ev.summary = (
            f"Evaluation {output.evaluation_status}. "
            f"Evaluated: {n_evaluated} | Failed/Skipped: {n_failed} | "
            f"Total time: {output.total_execution_time_seconds:.2f}s"
        )
        if output.best_model_name:
            primary_value = output.best_model_metrics.get(
                output.primary_metric, 0.0
            )
            ev.summary += (
                f" | Best model: {output.best_model_name} "
                f"({output.primary_metric}={primary_value:.4f})"
            )
        ev.is_completed = output.evaluation_status in ("completed", "partial")

        state.model_evaluation = ev

        logger.info(
            "ModelEvaluationTool: state updated. "
            "Status: %s | Evaluated: %d | Charts stored: %d | Summary: %s",
            ev.evaluation_status,
            n_evaluated,
            len(ev.visualization_data),
            ev.summary,
        )

        return state
