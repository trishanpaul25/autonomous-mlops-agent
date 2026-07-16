"""
Hyperparameter Optimization Tool.

Applies a HPOptimizationOutput (built by HyperparameterOptimizationAgent
from the list of HPOptimizationResults) to the PipelineState by
populating HyperparameterOptimizationState's serialisable fields.

This tool is a pure writer. It does not make any decisions, run any
CV search, or perform any IO. It mirrors ModelTrainingTool exactly.

Responsibility boundary
-----------------------
This tool is responsible for:
  - Transferring HPOptimizationOutput data → HyperparameterOptimizationState
  - Storing fitted optimized estimators in optimized_model_objects
  - Computing the best_overall_model_name from scores
  - Setting is_completed and summary

It is NOT responsible for:
  - Running any optimization
  - Making decisions about which model to use
  - Evaluating models
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from schemas.hyperparameter_optimization_schema import HPOptimizationOutput
from tools.base_tool import BaseTool
from tools.hyperparameter_optimization.hp_optimizer import HPOptimizationResult
from utils.logger import logger

if TYPE_CHECKING:
    from state.pipeline_state import PipelineState


class HyperparameterOptimizationTool(BaseTool):
    """
    Writes the structured HPO result into PipelineState.

    execute() transfers data from:
      - HPOptimizationOutput       → HyperparameterOptimizationState (serialisable fields)
      - list[HPOptimizationResult] → optimized_model_objects (fitted estimators)
    """

    def execute(
        self,
        state: "PipelineState",
        output: HPOptimizationOutput,
        optimization_results: list[HPOptimizationResult],
    ) -> "PipelineState":
        """
        Populate HyperparameterOptimizationState from a HPOptimizationOutput
        and the raw HPOptimizationResult list (which carries fitted models).

        Parameters
        ----------
        state : PipelineState
            Shared pipeline state to update.
        output : HPOptimizationOutput
            Structured aggregate result from HyperparameterOptimizationAgent.
        optimization_results : list[HPOptimizationResult]
            Raw results containing fitted optimized estimators. These are NOT
            part of HPOptimizationOutput because Pydantic cannot serialise
            arbitrary estimator objects.

        Returns
        -------
        PipelineState
            Updated state with hyperparameter_optimization fully populated.
        """
        hpo = state.hyperparameter_optimization

        # -- Scalars -------------------------------------------------------
        hpo.optimization_status = output.optimization_status
        hpo.total_execution_time_seconds = output.total_execution_time_seconds
        hpo.errors = output.errors
        hpo.scoring_metric = output.scoring_metric
        hpo.best_overall_model_name = output.best_overall_model_name
        hpo.best_overall_score = output.best_overall_score

        # -- Serialisable record lists ------------------------------------
        hpo.optimized_models = [
            record.model_dump()
            for record in output.optimized_models
        ]
        hpo.failed_models = [
            record.model_dump()
            for record in output.failed_models
        ]

        # -- Fitted estimator objects (not JSON-serialisable) -------------
        for result in optimization_results:
            if result.status == "optimized" and result.fitted_model is not None:
                hpo.optimized_model_objects[result.model_identifier] = (
                    result.fitted_model
                )

        # -- Derived summary ---------------------------------------------
        n_optimized = len(output.optimized_models)
        n_failed = len(output.failed_models)

        hpo.summary = (
            f"HPO {output.optimization_status}. "
            f"Optimized: {n_optimized} | Failed/Skipped: {n_failed} | "
            f"Total time: {output.total_execution_time_seconds:.2f}s"
        )
        if output.best_overall_model_name:
            hpo.summary += (
                f" | Best model: {output.best_overall_model_name} "
                f"({output.scoring_metric}={output.best_overall_score:.4f})"
            )

        # -- Completion flag ---------------------------------------------
        hpo.is_completed = output.optimization_status in (
            "completed", "partial", "skipped"
        )

        state.hyperparameter_optimization = hpo

        logger.info(
            "HyperparameterOptimizationTool: state updated. "
            "Status: %s | Estimators stored: %d | Summary: %s",
            hpo.optimization_status,
            len(hpo.optimized_model_objects),
            hpo.summary,
        )

        return state
