"""
Model Registry Agent.

Registers the best model from Model Evaluation into MLflow: logs the
fitted estimator (bundled with the fitted feature-engineering
transformers when available, so the registered model accepts raw
input directly), its best hyperparameters, and its evaluation metrics,
then registers a new version under a registered model name.

The agent NEVER trains, tunes, or evaluates models — its sole job is
packaging and registering whatever Model Evaluation already determined
was best. No LLM is used; registration is a deterministic MLflow operation.

Estimator retrieval mirrors the exact same fallback chain used by
ModelEvaluator and ExplainabilityAgent:
  1. hyperparameter_optimization.optimized_model_objects[identifier]
  2. model_training.trained_model_objects (scan by model_name)
  3. None -> registry_status = "skipped"

Failure philosophy
-------------------
A registration failure (unreachable tracking server, etc.) does NOT
halt the pipeline the way a broken train/test split would — this
agent never sets state.status = PipelineStatus.FAILED, mirroring
ExplainabilityAgent. The orchestrator surfaces registry failures as a
visible warning instead (see master_orchestrator_agent.py).
"""

from __future__ import annotations

import os
import re
import time
from typing import Any

from agents.base_agent import BaseAgent
from schemas.model_registry_schema import ModelRegistryOutput
from server.core.paths import ARTIFACTS_DIR
from state.pipeline_state import PipelineState
from tools.model_registry.feature_pipeline_replay import FeatureTransformReplay
from tools.model_registry.model_registry_tool import ModelRegistryTool
from utils.logger import logger

try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False


_MODEL_ARTIFACT_PATH = "model"


class ModelRegistryAgent(BaseAgent):
    """
    Logs and registers the best model in MLflow.
    """

    def __init__(self, default_experiment_name: str = "autonomous-mlops") -> None:
        self.default_experiment_name = default_experiment_name
        self.tool = ModelRegistryTool()

    def run(self, state: PipelineState) -> PipelineState:
        start_time = time.perf_counter()
        warnings: list[str] = []
        errors: list[str] = []

        state.current_agent = "ModelRegistryAgent"

        try:
            if not _MLFLOW_AVAILABLE:
                errors.append(
                    "mlflow package is not installed. Install it with: pip install mlflow"
                )
                return self._finalize(state, self._failed_output(errors, warnings, start_time))

            ev = state.model_evaluation
            if not ev.is_completed or not ev.best_model_name or not ev.best_model_identifier:
                warnings.append(
                    "No best model available from Model Evaluation — nothing to register."
                )
                output = ModelRegistryOutput(
                    registry_status="skipped",
                    warnings=warnings,
                    total_execution_time_seconds=round(time.perf_counter() - start_time, 4),
                    registry_summary="Model Registry skipped — no best model available.",
                )
                return self._finalize(state, output)

            estimator, _resolved_identifier = self._resolve_estimator(state, ev)
            if estimator is None:
                errors.append(
                    f"No fitted estimator found for best model '{ev.best_model_name}' "
                    "in optimized_model_objects or trained_model_objects."
                )
                return self._finalize(state, self._failed_output(errors, warnings, start_time))

            # -- Build the replay wrapper (bundle transformers if available) --
            fe_state = state.feature_engineering
            bundled_transformers = bool(fe_state.config)
            if not bundled_transformers:
                warnings.append(
                    "No feature-engineering config found on state — registering "
                    "the raw estimator only. The registered model will expect "
                    "already-transformed input, not raw data."
                )

            task_type = state.model_selection.task_type or state.model_evaluation.task_type
            target_column = state.validation.target_column

            from tools.model_registry.bundled_pipeline_model import BundledPipelineModel

            replay = FeatureTransformReplay(fe_state, target_column=target_column)
            python_model = BundledPipelineModel(
                replay=replay,
                estimator=estimator,
                task_type=task_type,
            )

            # -- Resolve tracking URI --
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
            if not tracking_uri:
                local_store = ARTIFACTS_DIR / "mlruns"
                local_store.mkdir(parents=True, exist_ok=True)
                tracking_uri = local_store.as_uri()
                warnings.append(
                    f"MLFLOW_TRACKING_URI not set — falling back to local file "
                    f"store at {tracking_uri}. MLflow's Model Registry "
                    "(versioning) requires a database-backed tracking server; "
                    "the local file store may not support registration. Set "
                    "MLFLOW_TRACKING_URI in .env to use your MLflow server."
                )

            try:
                mlflow.set_tracking_uri(tracking_uri)
            except Exception as exc:
                errors.append(f"Could not set MLflow tracking URI '{tracking_uri}': {exc}")
                return self._finalize(
                    state,
                    self._failed_output(errors, warnings, start_time, tracking_uri_used=tracking_uri),
                )

            experiment_name = self._resolve_experiment_name(state)
            try:
                mlflow.set_experiment(experiment_name)
            except Exception as exc:
                warnings.append(f"Could not set experiment '{experiment_name}': {exc}")

            registered_model_name = self._resolve_registered_model_name(state)

            # -- Gather params/metrics to log --
            hpo_record = self._find_hpo_record(state, ev.best_model_name)
            logged_params = {
                str(k): str(v)
                for k, v in (hpo_record.get("best_parameters", {}) if hpo_record else {}).items()
            }
            logged_metrics = {
                k: float(v)
                for k, v in ev.best_model_metrics.items()
                if isinstance(v, (int, float))
            }

            # -- Log + register --
            try:
                with mlflow.start_run(run_name=f"{ev.best_model_name}-{state.run_id[:8]}") as run:
                    mlflow.set_tags({
                        "pipeline_run_id": state.run_id,
                        "model_name": ev.best_model_name,
                        "task_type": task_type or "",
                        "bundled_transformers": str(bundled_transformers),
                    })
                    if logged_params:
                        mlflow.log_params(logged_params)
                    if logged_metrics:
                        mlflow.log_metrics(logged_metrics)

                    mlflow.pyfunc.log_model(
                        python_model=python_model,
                        artifact_path=_MODEL_ARTIFACT_PATH,
                    )

                    run_id = run.info.run_id

                run_model_uri = f"runs:/{run_id}/{_MODEL_ARTIFACT_PATH}"

                registered_version = mlflow.register_model(
                    model_uri=run_model_uri,
                    name=registered_model_name,
                )
                model_version = int(registered_version.version)
                model_uri = f"models:/{registered_model_name}/{model_version}"

            except Exception as exc:
                errors.append(f"MLflow logging/registration failed: {exc}")
                logger.error("[Registry] MLflow operation failed: %s", exc, exc_info=True)
                return self._finalize(
                    state,
                    self._failed_output(errors, warnings, start_time, tracking_uri_used=tracking_uri),
                )

            elapsed = time.perf_counter() - start_time
            summary = (
                f"Registered '{ev.best_model_name}' as '{registered_model_name}' "
                f"v{model_version} in MLflow (run {run_id}). "
                f"Transformers bundled: {bundled_transformers}."
            )

            output = ModelRegistryOutput(
                registry_status="completed",
                registered_model_name=registered_model_name,
                model_version=model_version,
                mlflow_run_id=run_id,
                mlflow_model_uri=model_uri,
                mlflow_run_model_uri=run_model_uri,
                tracking_uri_used=tracking_uri,
                bundled_transformers=bundled_transformers,
                logged_params=logged_params,
                logged_metrics=logged_metrics,
                total_execution_time_seconds=round(elapsed, 4),
                warnings=warnings,
                errors=errors,
                registry_summary=summary,
            )

            state.logs.append(summary)
            logger.info("[Registry] %s", summary)

            return self._finalize(state, output)

        except Exception as exc:
            # Truly unexpected failure (a bug, not an environment issue
            # handled above). Registration failing should never halt the
            # pipeline — mirrors ExplainabilityAgent's non-blocking
            # philosophy — so state.status is deliberately left untouched.
            # Matches FeatureEngineeringAgent's convention of NOT appending
            # to completed_steps when the step never meaningfully ran.
            logger.error("[Registry] Unhandled error: %s", exc, exc_info=True)
            state.logs.append(f"Model registration failed: {exc}")
            output = self._failed_output([str(exc)], warnings, start_time)
            return self.tool.execute(state, output)

    # ==========================================================
    # Internal helpers
    # ==========================================================

    def _finalize(self, state: PipelineState, output: ModelRegistryOutput) -> PipelineState:
        state = self.tool.execute(state, output)
        state.completed_steps.append("Model Registry")
        return state

    @staticmethod
    def _resolve_estimator(state: PipelineState, ev: Any) -> tuple[Any, str]:
        hpo = state.hyperparameter_optimization
        mt = state.model_training

        estimator = hpo.optimized_model_objects.get(ev.best_model_identifier)
        if estimator is not None:
            return estimator, ev.best_model_identifier

        for record in mt.trained_models:
            if record.get("model_name") == ev.best_model_name:
                identifier = record.get("model_identifier", "")
                estimator = mt.trained_model_objects.get(identifier)
                if estimator is not None:
                    return estimator, identifier

        return None, ""

    @staticmethod
    def _find_hpo_record(state: PipelineState, model_name: str) -> dict | None:
        hpo = state.hyperparameter_optimization
        for record in list(hpo.optimized_models) + list(hpo.failed_models):
            if record.get("model_name") == model_name:
                return record
        return None

    @staticmethod
    def _sanitize_mlflow_name(raw: str) -> str:
        sanitized = re.sub(r"[^A-Za-z0-9_\-\.]+", "-", raw).strip("-_.")
        return sanitized or "autonomous-mlops-model"

    def _resolve_experiment_name(self, state: PipelineState) -> str:
        dataset_name = getattr(state.dataset, "dataset_name", None)
        if dataset_name:
            return self._sanitize_mlflow_name(dataset_name)
        return self.default_experiment_name

    def _resolve_registered_model_name(self, state: PipelineState) -> str:
        dataset_name = getattr(state.dataset, "dataset_name", None)
        task_type = state.model_selection.task_type or "model"
        if dataset_name:
            return self._sanitize_mlflow_name(f"{dataset_name}-{task_type}")
        return self._sanitize_mlflow_name(f"autonomous-mlops-{task_type}")

    @staticmethod
    def _failed_output(
        errors: list[str],
        warnings: list[str],
        start_time: float,
        tracking_uri_used: str = "",
    ) -> ModelRegistryOutput:
        elapsed = time.perf_counter() - start_time
        return ModelRegistryOutput(
            registry_status="failed",
            tracking_uri_used=tracking_uri_used,
            errors=errors,
            warnings=warnings,
            total_execution_time_seconds=round(elapsed, 4),
            registry_summary=(
                f"Model registration failed: {'; '.join(errors) if errors else 'unknown error'}"
            ),
        )