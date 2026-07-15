"""
Model Evaluation Agent.

Responsible exclusively for evaluating every optimized (or trained
fallback) model on the held-out test set and producing a ranked
comparison of their performance.

The agent MUST NOT:
  - Perform feature engineering
  - Select models
  - Train models
  - Tune hyperparameters
  - Register or deploy models
  - Modify the train/test split

Its sole responsibility is evaluating fitted models using the test data
that was held out during Model Training.

Architecture
------------
The agent orchestrates four internal components:

1. MetricsRegistry       — declarative catalogue of metrics by task type
2. MetricsCalculator     — computes all metrics; error-isolated per metric
3. VisualizationDataBuilder — chart-ready data structures
4. ModelEvaluator        — the per-model evaluation loop
5. ModelEvaluationTool   — pure state writer
6. (optional) LLM chain  — narrative interpretation only

Deterministic path (always executed):
  ModelEvaluator → MetricsCalculator → ModelEvaluationTool

Optional LLM path (requires GOOGLE_API_KEY / GEMINI_API_KEY):
  model_evaluation_prompt | structured_llm → EvaluationNarrativeOutput
  → stored in state.model_evaluation.narrative

Error Handling
--------------
  Model-level failures: caught by ModelEvaluator, recorded as "failed"
  Pipeline fails only: when ALL models fail evaluation
  LLM errors: caught and logged; narrative is set to None (non-blocking)
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from agents.base_agent import BaseAgent

from prompts.model_evaluation_prompt import model_evaluation_prompt

from schemas.model_evaluation_schema import (
    EvaluationNarrativeOutput,
    ModelEvaluationOutput,
    ModelEvaluationRecord,
)

from services.llm_service import LLMService

from state.pipeline_state import PipelineState

from tools.model_evaluation.model_evaluator import EvaluationResult, ModelEvaluator
from tools.model_evaluation.model_evaluation_tool import ModelEvaluationTool
from tools.model_evaluation.metrics_registry import MetricsRegistry

from utils.logger import logger

from server.core.constants import PipelineStatus

class ModelEvaluationAgent(BaseAgent):
    """
    Evaluates every optimized (or trained fallback) model on the
    held-out test set and produces ranked performance results.

    Constructor wires up:
      - ModelEvaluator        for the per-model evaluation loop
      - MetricsRegistry       for primary metric name lookup
      - ModelEvaluationTool   for writing results back to state
      - (optional) LLM chain  for narrative interpretation
    """

    def __init__(self) -> None:
        self.evaluator = ModelEvaluator()
        self.registry = MetricsRegistry()
        self.tool = ModelEvaluationTool()

        # Optional LLM chain for narrative generation
        self.chain = None
        if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
            try:
                llm = LLMService.get_structured_llm(EvaluationNarrativeOutput)
                self.chain = model_evaluation_prompt | llm
            except Exception as exc:
                logger.warning(
                    "ModelEvaluationAgent: LLM initialisation failed (%s). "
                    "Narrative generation will be skipped.",
                    exc,
                )

    def run(self, state: PipelineState) -> PipelineState:
        """
        Execute the Model Evaluation Agent.

        Execution flow
        --------------
        1. Set current_agent and log start.
        2. Guard: HPO is_completed (or model training if HPO skipped).
        3. Guard: X_test / y_test not None/empty.
        4. Guard: task type is supported for evaluation.
        5. Delegate evaluation loop to ModelEvaluator.
        6. Rank evaluated models by primary metric value.
        7. Identify best model (rank 1).
        8. Build comparison table.
        9. Build ModelEvaluationOutput.
        10. Write output to PipelineState via ModelEvaluationTool.
        11. Optionally invoke LLM for narrative.
        12. Fail pipeline only if ALL models fail evaluation.

        Parameters
        ----------
        state : PipelineState
            Shared pipeline state after HPO.

        Returns
        -------
        PipelineState
            Updated state with model_evaluation fully populated.
        """
        pipeline_start = time.perf_counter()

        try:
            state.current_agent = "ModelEvaluationAgent"

            logger.info("[Eval] Model Evaluation Agent: starting.")
            state.logs.append("Model evaluation started.")
            if not state.model_training.is_completed:
                raise ValueError(
                    "ModelEvaluationAgent requires Model Training to have "
                    "completed successfully before it can run."
                )
            if (
                state.model_training.X_test is None
                or state.model_training.y_test is None
                or len(state.model_training.X_test) == 0
            ):
                raise ValueError(
                    "ModelEvaluationAgent: X_test or y_test is missing or empty "
                    "from ModelTrainingState. Ensure the train/test split "
                    "completed successfully."
                )
            task_type: str = (state.model_selection.task_type or "").lower()

            if not self.registry.is_supported(task_type):
                logger.warning(
                    "[Eval] Task type '%s' has no registered evaluation metrics. "
                    "Skipping evaluation.",
                    task_type,
                )
                state.logs.append(
                    f"[Eval] Skipped — task type '{task_type}' has no "
                    "registered evaluation metrics."
                )
                state.model_evaluation.evaluation_status = "failed"
                state.model_evaluation.summary = (
                    f"Evaluation skipped: unsupported task type '{task_type}'."
                )
                state.model_evaluation.errors = [
                    f"No evaluation metrics registered for task type '{task_type}'."
                ]
                state.completed_steps.append("Model Evaluation")
                return state

            logger.info(
                "[Eval] Task: %s | Test samples: %d | Models in pool: %d",
                task_type,
                len(state.model_training.X_test),
                len(state.hyperparameter_optimization.optimized_models)
                + len(state.hyperparameter_optimization.failed_models),
            )
            logger.info("[Eval] Evaluation loop starting.")
            state.logs.append(
                f"[Eval] Evaluating models on {len(state.model_training.X_test)} "
                "test samples."
            )

            evaluation_results: list[EvaluationResult] = (
                self.evaluator.evaluate_all(state)
            )

            total_elapsed = time.perf_counter() - pipeline_start
            succeeded = [r for r in evaluation_results if r.status == "evaluated"]
            failed_or_skipped = [
                r for r in evaluation_results if r.status != "evaluated"
            ]
            all_errors = [r.error for r in failed_or_skipped if r.error]

            # Determine the primary metric name (use first succeeded model's name)
            primary_metric_name = ""
            if succeeded:
                primary_metric_name = succeeded[0].primary_metric_name
            else:
                # No succeeded models — use registry default
                has_proba = False
                primary_metric_name = self.registry.get_primary_metric(
                    task_type, has_proba
                )
            # For regression r2: higher is better
            # For all other metrics: higher is better
            ranked = sorted(
                succeeded,
                key=lambda r: r.primary_metric_value,
                reverse=True,
            )
            best_model_name = ""
            best_model_identifier = ""
            best_model_metrics: dict[str, float] = {}

            if ranked:
                best = ranked[0]
                best_model_name = best.model_name
                best_model_identifier = best.model_identifier
                best_model_metrics = best.metrics
            evaluated_records = [
                self._to_record(result, rank=i + 1)
                for i, result in enumerate(ranked)
            ]
            failed_records = [
                self._to_record(result, rank=0)
                for result in failed_or_skipped
            ]
            comparison_table = self._build_comparison_table(ranked)
            if len(succeeded) == len(evaluation_results):
                evaluation_status = "completed"
            elif len(succeeded) > 0:
                evaluation_status = "partial"
            else:
                evaluation_status = "failed"
            summary = (
                f"Evaluated {len(succeeded)}/{len(evaluation_results)} models "
                f"in {total_elapsed:.2f}s using '{primary_metric_name}' as primary metric."
            )
            if best_model_name:
                primary_val = best_model_metrics.get(primary_metric_name, 0.0)
                summary += (
                    f" Best model: {best_model_name} "
                    f"({primary_metric_name}={primary_val:.4f})."
                )

            output = ModelEvaluationOutput(
                evaluation_status=evaluation_status,  # type: ignore[arg-type]
                evaluated_models=evaluated_records,
                failed_models=failed_records,
                best_model_name=best_model_name,
                best_model_identifier=best_model_identifier,
                best_model_metrics=best_model_metrics,
                primary_metric=primary_metric_name,
                comparison_table=comparison_table,
                evaluation_summary=summary,
                total_execution_time_seconds=round(total_elapsed, 4),
                errors=all_errors,
            )
            state = self.tool.execute(
                state,
                output,
                evaluation_results=evaluation_results,
            )

            state.completed_steps.append("Model Evaluation")

            logger.info(
                "[Eval] Evaluation finished. "
                "Status: %s | Evaluated: %d | Failed/Skipped: %d | "
                "Best: %s | Total time: %.2fs",
                evaluation_status,
                len(succeeded),
                len(failed_or_skipped),
                best_model_name,
                total_elapsed,
            )

            state.logs.append(
                f"Model evaluation finished. "
                f"Status: {evaluation_status} | "
                f"Evaluated: {len(succeeded)} | "
                f"Failed/Skipped: {len(failed_or_skipped)} | "
                f"Best model: {best_model_name} | "
                f"Time: {total_elapsed:.2f}s"
            )
            if self.chain is not None and succeeded:
                state = self._generate_narrative(state, output)
            if evaluation_status == "failed":
                state.status = PipelineStatus.FAILED
                state.error = (
                    "All candidate models failed evaluation. "
                    f"Errors: {'; '.join(all_errors)}"
                )
                logger.error("[Eval] All models failed. Pipeline halting.")

            return state

        except Exception as exc:
            state.status = PipelineStatus.FAILED
            state.error = str(exc)

            logger.error(
                "[Eval] ModelEvaluationAgent: unrecoverable error — %s",
                exc,
                exc_info=True,
            )
            state.logs.append(f"Model evaluation failed: {exc}")

            return state

    def _generate_narrative(
        self,
        state: PipelineState,
        output: ModelEvaluationOutput,
    ) -> PipelineState:
        """
        Invoke the optional LLM chain to produce a human-readable
        narrative interpretation of the evaluation results.

        This method never raises. All LLM errors are caught and logged.
        The narrative fields are set to None/empty on failure.

        Parameters
        ----------
        state : PipelineState
            Current pipeline state (model_evaluation already populated).
        output : ModelEvaluationOutput
            The structured evaluation output for prompt injection.

        Returns
        -------
        PipelineState
            State with narrative fields populated (or unchanged on error).
        """
        try:
            logger.info("[Eval] Invoking LLM for narrative generation...")

            # Format comparison table and best model metrics as readable text
            comparison_text = "\n".join(
                self._format_row(row) for row in output.comparison_table
            )
            best_metrics_text = "\n".join(
                f"  {k}: {v:.4f}" for k, v in output.best_model_metrics.items()
            )
            failed_text = (
                ", ".join(r.model_name for r in output.failed_models)
                if output.failed_models
                else "None"
            )

            narrative: EvaluationNarrativeOutput = self.chain.invoke(
                {
                    "task_type": state.model_selection.task_type,
                    "primary_metric": output.primary_metric,
                    "n_test_samples": len(state.model_training.X_test),
                    "comparison_table": comparison_text,
                    "best_model_name": output.best_model_name,
                    "best_model_metrics": best_metrics_text,
                    "failed_models": failed_text,
                }
            )

            # Store structured narrative
            state.model_evaluation.narrative = narrative.business_summary
            state.model_evaluation.narrative_structured = narrative.model_dump()

            logger.info("[Eval] LLM narrative generated successfully.")
            state.logs.append("[Eval] LLM narrative generated.")

        except Exception as exc:
            logger.warning(
                "[Eval] LLM narrative generation failed (%s). "
                "Evaluation results are unaffected.",
                exc,
            )

        return state

    @staticmethod
    def _to_record(
        result: EvaluationResult,
        rank: int,
    ) -> ModelEvaluationRecord:
        """
        Convert an internal EvaluationResult to a Pydantic
        ModelEvaluationRecord.

        Parameters
        ----------
        result : EvaluationResult
            Raw internal result from ModelEvaluator.
        rank : int
            Rank position (1 = best, 0 = failed/skipped).

        Returns
        -------
        ModelEvaluationRecord
            Serializable Pydantic record for state storage.
        """
        return ModelEvaluationRecord(
            model_name=result.model_name,
            class_path=result.class_path,
            model_identifier=result.model_identifier,
            evaluation_status=result.status,  # type: ignore[arg-type]
            rank=rank,
            metrics=result.metrics,
            primary_metric_name=result.primary_metric_name,
            primary_metric_value=result.primary_metric_value,
            prediction_time_seconds=result.prediction_time_seconds,
            notes=result.notes,
        )

    @staticmethod
    def _build_comparison_table(
        ranked: list[EvaluationResult],
    ) -> list[dict[str, Any]]:
        """
        Build a side-by-side metric comparison table across all
        evaluated models, ordered by rank.

        Each row contains model_name, rank, and all metric values.

        Parameters
        ----------
        ranked : list[EvaluationResult]
            Evaluated results ordered by primary metric (best first).

        Returns
        -------
        list[dict[str, Any]]
            One dict per model, with all metric keys unified across
            all models. Missing metrics are set to None.
        """
        if not ranked:
            return []

        # Collect all unique metric names across all models
        all_metric_names: set[str] = set()
        for result in ranked:
            all_metric_names.update(result.metrics.keys())
        sorted_metric_names = sorted(all_metric_names)

        table = []
        for i, result in enumerate(ranked):
            row: dict[str, Any] = {
                "rank": i + 1,
                "model_name": result.model_name,
                "prediction_time_seconds": result.prediction_time_seconds,
            }
            for metric_name in sorted_metric_names:
                value = result.metrics.get(metric_name)
                row[metric_name] = round(value, 4) if value is not None else None
            table.append(row)

        return table

    @staticmethod
    def _format_row(row: dict[str, Any]) -> str:
        """
        Format a comparison table row as a readable string for the LLM prompt.

        Parameters
        ----------
        row : dict[str, Any]
            A single comparison table row dict.

        Returns
        -------
        str
            Human-readable one-liner for the prompt.
        """
        model = row.get("model_name", "Unknown")
        rank = row.get("rank", "-")
        metrics_text = " | ".join(
            f"{k}={v:.4f}"
            for k, v in row.items()
            if k not in ("rank", "model_name", "prediction_time_seconds")
            and isinstance(v, float)
        )
        return f"  Rank {rank}: {model} — {metrics_text}"
