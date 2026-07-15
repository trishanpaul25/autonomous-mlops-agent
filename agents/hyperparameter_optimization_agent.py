"""
Hyperparameter Optimization Agent.

Responsible exclusively for hyperparameter optimization of every
candidate model that was successfully trained by the Model Training Agent.

The agent MUST NOT:
  - Perform feature engineering
  - Perform model selection
  - Evaluate model performance (metrics on test set)
  - Deploy or register models
  - Modify the train/test split

Its sole responsibility is finding the best hyperparameters for each
trained model using cross-validated search on the training data.

Architecture
------------
The agent orchestrates three internal tools:

1. ScoringStrategySelector  — maps task_type → sklearn scoring metric
2. HPOptimizer              — runs per-model CV search with error isolation
3. HyperparameterOptimizationTool — writes results into PipelineState

No LLM is used. Hyperparameter optimization is a deterministic
computation step.

Error Handling Matrix
---------------------
┌─────────────────────────────────────────────┬────────────────────────────────────┐
│ Scenario                                    │ Behaviour                          │
├─────────────────────────────────────────────┼────────────────────────────────────┤
│ Model Training not completed                │ ValueError → pipeline fails        │
│ No trained models found                     │ ValueError → pipeline fails        │
│ X_train / y_train missing                   │ ValueError → pipeline fails        │
│ Task type not supported (clustering)        │ status="skipped" → pipeline OK     │
│ Individual model search space missing       │ Record "skipped", continue loop    │
│ Individual model import fails               │ Record "failed", continue loop     │
│ Individual CV search fails                  │ Record "failed", continue loop     │
│ All models fail                             │ status="failed" → pipeline fails   │
│ ≥ 1 model optimized                         │ status="partial" or "completed"    │
└─────────────────────────────────────────────┴────────────────────────────────────┘
"""

from __future__ import annotations

import time

from agents.base_agent import BaseAgent

from schemas.hyperparameter_optimization_schema import (
    HPOptimizationOutput,
    OptimizedModelRecord,
)

from state.pipeline_state import PipelineState

from tools.hyperparameter_optimization.hp_optimizer import (
    HPOptimizer,
    HPOptimizationResult,
)
from tools.hyperparameter_optimization.hyperparameter_optimization_tool import (
    HyperparameterOptimizationTool,
)
from tools.hyperparameter_optimization.scoring_strategy_selector import (
    ScoringStrategySelector,
)

from utils.logger import logger


class HyperparameterOptimizationAgent(BaseAgent):
    """
    Optimizes hyperparameters for every successfully trained model.

    Constructor wires up:
      - ScoringStrategySelector  for task-type → scoring metric mapping
      - HPOptimizer              for the per-model CV search loop
      - HyperparameterOptimizationTool  for writing results back to state
    """

    def __init__(
        self,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        """
        Parameters
        ----------
        random_state : int
            Random seed for reproducibility in CV searches. Default 42.
        n_jobs : int
            Parallel jobs for CV search. -1 uses all available CPUs.
        """
        self.scorer_selector = ScoringStrategySelector()
        self.optimizer = HPOptimizer(
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.tool = HyperparameterOptimizationTool()

    def run(self, state: PipelineState) -> PipelineState:
        """
        Execute the Hyperparameter Optimization Agent.

        Execution flow
        --------------
        1. Set current_agent and log start.
        2. Guard: verify Model Training completed with trained models.
        3. Guard: verify X_train / y_train are available.
        4. Check task type support; skip entire step if not applicable.
        5. Log optimization start.
        6. Delegate optimization loop to HPOptimizer.
        7. Separate results into optimized vs. failed/skipped.
        8. Determine best overall model by score.
        9. Build HPOptimizationOutput.
        10. Write output to PipelineState via HyperparameterOptimizationTool.
        11. Fail pipeline only if every model failed (not just some).

        Parameters
        ----------
        state : PipelineState
            Shared pipeline state after Model Training.

        Returns
        -------
        PipelineState
            Updated state with hyperparameter_optimization fully populated.
        """
        pipeline_start = time.perf_counter()

        try:
            state.current_agent = "HyperparameterOptimizationAgent"

            logger.info("[HPO] Hyperparameter Optimization Agent: starting.")
            state.logs.append("Hyperparameter optimization started.")

            if not state.model_training.is_completed:
                raise ValueError(
                    "HyperparameterOptimizationAgent requires Model Training "
                    "to have completed successfully before it can run."
                )

            trained_models = state.model_training.trained_models
            if not trained_models:
                raise ValueError(
                    "HyperparameterOptimizationAgent: no trained models found "
                    "in ModelTrainingState. Model Training may not have "
                    "produced any successful models."
                )
            if state.model_training.X_train is None or state.model_training.y_train is None:
                raise ValueError(
                    "HyperparameterOptimizationAgent: X_train or y_train is "
                    "missing from ModelTrainingState. Ensure the train/test "
                    "split completed successfully."
                )

            task_type: str = (state.model_selection.task_type or "").lower()
            if not self.scorer_selector.is_supported(task_type):
                scoring_metric = None
                logger.warning(
                    "[HPO] Task type '%s' does not support standard CV scoring. "
                    "Skipping hyperparameter optimization.",
                    task_type,
                )
                state.logs.append(
                    f"[HPO] Skipped — task type '{task_type}' does not support "
                    "hyperparameter optimization with standard cross-validation."
                )

                skipped_records = [
                    OptimizedModelRecord(
                        model_name=m.get("model_name", "Unknown"),
                        class_path=m.get("class_path", ""),
                        optimization_status="skipped",
                        notes=(
                            f"Task type '{task_type}' not supported for HPO."
                        ),
                    )
                    for m in trained_models
                ]

                total_elapsed = time.perf_counter() - pipeline_start

                output = HPOptimizationOutput(
                    optimization_status="skipped",
                    optimized_models=[],
                    failed_models=skipped_records,
                    optimization_summary=(
                        f"HPO skipped. Task type '{task_type}' does not "
                        "support standard cross-validation scoring."
                    ),
                    total_execution_time_seconds=round(total_elapsed, 4),
                )

                state = self.tool.execute(state, output, [])
                state.completed_steps.append("Hyperparameter Optimization")

                logger.info(
                    "[HPO] Optimization step skipped for task type '%s'.",
                    task_type,
                )
                return state

            scoring_metric = self.scorer_selector.get_scoring_metric(task_type)

            logger.info(
                "[HPO] Task: %s | Scoring: %s | Models to optimize: %d",
                task_type,
                scoring_metric,
                len(trained_models),
            )
            logger.info("[HPO] Optimization loop starting.")
            state.logs.append(
                f"[HPO] Starting optimization for {len(trained_models)} model(s) "
                f"using '{scoring_metric}' scoring."
            )

            optimization_results: list[HPOptimizationResult] = (
                self.optimizer.optimize_all(state)
            )

            total_elapsed = time.perf_counter() - pipeline_start
            succeeded = [
                r for r in optimization_results if r.status == "optimized"
            ]
            failed_or_skipped = [
                r for r in optimization_results if r.status != "optimized"
            ]
            all_errors = [r.error for r in failed_or_skipped if r.error]

            optimized_records = [
                self._to_record(r) for r in succeeded
            ]
            failed_records = [
                self._to_record(r) for r in failed_or_skipped
            ]

            # Determine overall optimization status
            if len(succeeded) == len(optimization_results):
                optimization_status = "completed"
            elif len(succeeded) > 0:
                optimization_status = "partial"
            else:
                optimization_status = "failed"
            best_overall_model_name = ""
            best_overall_score = 0.0

            if succeeded:
                best_result = max(succeeded, key=lambda r: r.best_score)
                best_overall_model_name = best_result.model_name
                best_overall_score = best_result.best_score
            summary = (
                f"Optimized {len(succeeded)}/{len(optimization_results)} models "
                f"in {total_elapsed:.2f}s using '{scoring_metric}' scoring."
            )
            if best_overall_model_name:
                summary += (
                    f" Best model: {best_overall_model_name} "
                    f"(score={best_overall_score:.4f})."
                )

            output = HPOptimizationOutput(
                optimization_status=optimization_status,  # type: ignore[arg-type]
                optimized_models=optimized_records,
                failed_models=failed_records,
                best_overall_model_name=best_overall_model_name,
                best_overall_score=best_overall_score,
                scoring_metric=scoring_metric or "",
                optimization_summary=summary,
                total_execution_time_seconds=round(total_elapsed, 4),
                errors=all_errors,
            )

            state = self.tool.execute(
                state,
                output,
                optimization_results=optimization_results,
            )

            state.completed_steps.append("Hyperparameter Optimization")

            logger.info(
                "[HPO] Optimization finished. "
                "Status: %s | Optimized: %d | Failed/Skipped: %d | "
                "Total time: %.2fs",
                optimization_status,
                len(succeeded),
                len(failed_or_skipped),
                total_elapsed,
            )

            state.logs.append(
                f"Hyperparameter optimization finished. "
                f"Status: {optimization_status} | "
                f"Optimized: {len(succeeded)} | "
                f"Failed/Skipped: {len(failed_or_skipped)} | "
                f"Time: {total_elapsed:.2f}s"
            )
            if optimization_status == "failed":
                state.status = "failed"
                state.error = (
                    "All candidate models failed hyperparameter optimization. "
                    f"Errors: {'; '.join(all_errors)}"
                )
                logger.error(
                    "[HPO] All models failed. Pipeline halting."
                )

            return state

        except Exception as exc:
            state.status = "failed"
            state.error = str(exc)

            logger.error(
                "[HPO] HyperparameterOptimizationAgent: unrecoverable error — %s",
                exc,
                exc_info=True,
            )

            state.logs.append(
                f"Hyperparameter optimization failed: {exc}"
            )

            return state

    @staticmethod
    def _to_record(result: HPOptimizationResult) -> OptimizedModelRecord:
        """
        Convert an internal HPOptimizationResult to a Pydantic
        OptimizedModelRecord.

        Parameters
        ----------
        result : HPOptimizationResult
            Raw internal result from HPOptimizer.

        Returns
        -------
        OptimizedModelRecord
            Serialisable Pydantic record for state storage.
        """
        return OptimizedModelRecord(
            model_name=result.model_name,
            class_path=result.class_path,
            optimization_status=result.status,  # type: ignore[arg-type]
            best_parameters=result.best_parameters,
            best_score=result.best_score,
            optimization_time_seconds=result.optimization_time_seconds,
            model_identifier=result.model_identifier,
            strategy_used=result.strategy_used,
            scoring_metric=result.scoring_metric,
            notes=result.notes,
        )
