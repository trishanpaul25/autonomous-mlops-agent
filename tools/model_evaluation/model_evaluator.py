"""
Model Evaluator for the Model Evaluation Agent.

Executes the evaluation loop across all available optimized (or trained)
models.

Design Principles
-----------------
* Error isolation: each model is wrapped in its own try/except block.
  One model failure never prevents the remaining models from being
  evaluated.
* Fallback chain: the evaluator first looks for an optimized estimator
  in HyperparameterOptimizationState.optimized_model_objects. If absent
  (HPO was skipped or failed for that model), it falls back to
  ModelTrainingState.trained_model_objects by matching model_name.
* Timing: wall-clock prediction time is recorded per model.
* Visualization: builds chart data for each model via
  VisualizationDataBuilder.

Fallback Chain for Estimator Retrieval
---------------------------------------
1. state.hyperparameter_optimization.optimized_model_objects[model_identifier]
   → Preferred: HPO-tuned refitted estimator.
2. state.model_training.trained_model_objects (scan by model_name match)
   → Fallback: original trained estimator without HPO tuning.
3. Neither found → EvaluationResult(status="skipped")

"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import numpy as np

from tools.model_evaluation.metrics_calculator import MetricsCalculator
from tools.model_evaluation.visualization_data_builder import VisualizationDataBuilder
from utils.logger import logger

if TYPE_CHECKING:
    from state.pipeline_state import PipelineState

@dataclass
class EvaluationResult:
    """
    Internal result produced for each model evaluation attempt.

    This intermediate structure is used within ModelEvaluator.
    ModelEvaluationTool converts these to ModelEvaluationRecord objects.
    """

    model_name: str
    class_path: str
    model_identifier: str
    status: str                          # "evaluated" | "failed" | "skipped"
    metrics: dict[str, float] = field(default_factory=dict)
    primary_metric_name: str = ""
    primary_metric_value: float = 0.0
    prediction_time_seconds: float = 0.0
    visualization_data: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    error: str = ""


class ModelEvaluator:
    """
    Evaluates all available optimized (or trained fallback) models on
    the held-out test set.

    Constructor wires up:
      - MetricsCalculator       for deterministic metric computation
      - VisualizationDataBuilder for chart-ready data structures
    """

    def __init__(self) -> None:
        self.calculator = MetricsCalculator()
        self.viz_builder = VisualizationDataBuilder()

    def evaluate_all(
        self,
        state: "PipelineState",
    ) -> list[EvaluationResult]:
        """
        Evaluate every model listed in optimized_models (and fall back
        to trained models for any that were HPO-skipped).

        Parameters
        ----------
        state : PipelineState
            Shared pipeline state after HPO. Must have:
            - state.model_training.X_test / y_test
            - state.hyperparameter_optimization.optimized_models
            - state.hyperparameter_optimization.optimized_model_objects
            - state.model_training.trained_model_objects (fallback)
            - state.model_selection.task_type
            - state.feature_engineering.final_feature_columns

        Returns
        -------
        list[EvaluationResult]
            One result per model in the candidate pool.
        """
        task_type: str = (state.model_selection.task_type or "").lower()
        X_test = np.array(state.model_training.X_test)
        y_test = np.array(state.model_training.y_test)
        feature_names: list[str] = state.feature_engineering.final_feature_columns

        # Resolve the candidate pool: prefer optimized models,
        # supplement with training fallbacks
        candidates = self._resolve_candidates(state)

        results: list[EvaluationResult] = []

        for entry in candidates:
            model_name: str = entry["model_name"]
            class_path: str = entry["class_path"]
            model_identifier: str = entry["model_identifier"]
            estimator = entry["estimator"]

            logger.info(
                "[Eval] %s: evaluation started.",
                model_name,
            )
            state.logs.append(f"[Eval] {model_name}: evaluation started.")

            if estimator is None:
                result = EvaluationResult(
                    model_name=model_name,
                    class_path=class_path,
                    model_identifier=model_identifier,
                    status="skipped",
                    notes=(
                        "No fitted estimator found in optimized_model_objects "
                        "or trained_model_objects."
                    ),
                )
            else:
                result = self._evaluate_single(
                    model_name=model_name,
                    class_path=class_path,
                    model_identifier=model_identifier,
                    estimator=estimator,
                    X_test=X_test,
                    y_test=y_test,
                    task_type=task_type,
                    feature_names=feature_names,
                )

            results.append(result)

            if result.status == "evaluated":
                logger.info(
                    "[Eval] %s: evaluation completed in %.3fs. "
                    "%s=%.4f",
                    model_name,
                    result.prediction_time_seconds,
                    result.primary_metric_name,
                    result.primary_metric_value,
                )
                state.logs.append(
                    f"[Eval] {model_name}: evaluation completed "
                    f"({result.prediction_time_seconds:.3f}s). "
                    f"{result.primary_metric_name}="
                    f"{result.primary_metric_value:.4f}."
                )
            elif result.status == "skipped":
                logger.warning(
                    "[Eval] %s: skipped — %s", model_name, result.notes
                )
                state.logs.append(
                    f"[Eval] {model_name}: skipped — {result.notes}"
                )
            else:
                logger.warning(
                    "[Eval] %s: evaluation failed — %s",
                    model_name,
                    result.error,
                )
                state.logs.append(
                    f"[Eval] {model_name}: evaluation failed — {result.error}"
                )

        return results

    def _evaluate_single(
        self,
        model_name: str,
        class_path: str,
        model_identifier: str,
        estimator: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        task_type: str,
        feature_names: list[str],
    ) -> EvaluationResult:
        """
        Attempt to evaluate a single fitted estimator on the test set.

        All exceptions are caught and encoded in the EvaluationResult
        so the caller's loop is never interrupted.

        Parameters
        ----------
        model_name : str
            Human-readable model name for logging.
        class_path : str
            Fully-qualified class path for the record.
        model_identifier : str
            Unique identifier for state storage.
        estimator : Any
            Fitted sklearn-compatible estimator.
        X_test : np.ndarray
            Feature matrix for the held-out test set.
        y_test : np.ndarray
            Target vector for the held-out test set.
        task_type : str
            ML task type for metric selection.
        feature_names : list[str]
            Feature column names for feature importance charts.

        Returns
        -------
        EvaluationResult
            Outcome of the evaluation attempt.
        """
        start_time = time.perf_counter()
        try:
            y_pred: np.ndarray = estimator.predict(X_test)
        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            error_msg = str(exc)
            return EvaluationResult(
                model_name=model_name,
                class_path=class_path,
                model_identifier=model_identifier,
                status="failed",
                prediction_time_seconds=round(elapsed, 4),
                notes=f"predict() failed: {error_msg}",
                error=error_msg,
            )
        y_proba: np.ndarray | None = None
        if "classification" in task_type:
            try:
                y_proba = estimator.predict_proba(X_test)
            except (AttributeError, NotImplementedError):
                logger.debug(
                    "[Eval] %s: predict_proba() not available — "
                    "probability-based metrics will be skipped.",
                    model_name,
                )
            except Exception as exc:
                logger.warning(
                    "[Eval] %s: predict_proba() failed (%s) — "
                    "probability-based metrics will be skipped.",
                    model_name,
                    exc,
                )

        elapsed = time.perf_counter() - start_time
        metrics = self.calculator.compute(
            y_true=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
            task_type=task_type,
        )

        has_proba = y_proba is not None
        primary_name, primary_value = self.calculator.get_primary_metric_value(
            metrics=metrics,
            task_type=task_type,
            has_proba=has_proba,
        )
        viz_data: dict[str, Any] = {}
        try:
            viz_data = self.viz_builder.build_all(
                model=estimator,
                y_true=y_test,
                y_pred=y_pred,
                y_proba=y_proba,
                task_type=task_type,
                feature_names=feature_names if feature_names else None,
            )
        except Exception as exc:
            logger.warning(
                "[Eval] %s: visualization data build failed — %s",
                model_name,
                exc,
            )
        notes = (
            f"Evaluated in {elapsed:.3f}s. "
            f"Primary metric ({primary_name}): {primary_value:.4f}. "
            f"Total metrics computed: {len(metrics)}."
        )
        if not has_proba and "classification" in task_type:
            notes += " Note: probability-based metrics unavailable."

        return EvaluationResult(
            model_name=model_name,
            class_path=class_path,
            model_identifier=model_identifier,
            status="evaluated",
            metrics=metrics,
            primary_metric_name=primary_name,
            primary_metric_value=round(primary_value, 6),
            prediction_time_seconds=round(elapsed, 4),
            visualization_data=viz_data,
            notes=notes,
        )

    @staticmethod
    def _resolve_candidates(
        state: "PipelineState",
    ) -> list[dict[str, Any]]:
        """
        Build the unified candidate list by merging optimized models,
        HPO-failed models (with trained fallback), and any trained models
        not covered by HPO.

        Fallback chain per model:
        1. optimized_model_objects[model_identifier]  ← preferred
        2. trained_model_objects (scan by model_name) ← fallback
        3. None                                        ← skipped

        Parameters
        ----------
        state : PipelineState
            Shared pipeline state.

        Returns
        -------
        list[dict]
            Each entry: {model_name, class_path, model_identifier, estimator}
        """
        hpo = state.hyperparameter_optimization
        mt = state.model_training

        # Index trained_model_objects by model_name for fallback lookup
        trained_by_name: dict[str, Any] = {}
        trained_id_by_name: dict[str, str] = {}
        for record in mt.trained_models:
            name = record.get("model_name", "")
            identifier = record.get("model_identifier", "")
            if name and identifier and identifier in mt.trained_model_objects:
                trained_by_name[name] = mt.trained_model_objects[identifier]
                trained_id_by_name[name] = identifier

        # Start with all HPO records (optimized + failed/skipped)
        all_hpo_records = list(hpo.optimized_models) + list(hpo.failed_models)

        candidates: list[dict[str, Any]] = []
        covered_names: set[str] = set()

        for record in all_hpo_records:
            model_name = record.get("model_name", "Unknown")
            class_path = record.get("class_path", "")
            model_identifier = record.get("model_identifier", "")

            covered_names.add(model_name)

            # Try HPO store first
            estimator = hpo.optimized_model_objects.get(model_identifier)

            # Fall back to trained store by model_name
            if estimator is None:
                estimator = trained_by_name.get(model_name)
                if estimator is not None:
                    model_identifier = trained_id_by_name.get(model_name, "")
                    logger.info(
                        "[Eval] %s: using trained estimator (HPO was skipped).",
                        model_name,
                    )

            candidates.append({
                "model_name": model_name,
                "class_path": class_path,
                "model_identifier": model_identifier,
                "estimator": estimator,
            })

        # Include any trained models NOT covered by HPO records
        for record in mt.trained_models:
            model_name = record.get("model_name", "")
            if model_name not in covered_names and model_name:
                model_identifier = record.get("model_identifier", "")
                estimator = mt.trained_model_objects.get(model_identifier)
                candidates.append({
                    "model_name": model_name,
                    "class_path": record.get("class_path", ""),
                    "model_identifier": model_identifier,
                    "estimator": estimator,
                })

        return candidates
