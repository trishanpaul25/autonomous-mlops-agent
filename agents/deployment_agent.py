"""
Deployment Agent.

Loads the model that Model Registry just registered in MLflow and caches
it in the in-process ModelServerRegistry, behind a local FastAPI
inference route (see server/api/routes/predict.py). It never trains,
tunes, evaluates, or registers models — its sole job is taking whatever
Model Registry already produced and making it servable.

No LLM is used; loading a model URI is a deterministic MLflow operation.

Failure philosophy
-------------------
Mirrors ModelRegistryAgent: a deployment failure (model URI won't load,
tracking server unreachable, etc.) does NOT halt the pipeline the way a
broken train/test split would — this agent never sets
state.status = PipelineStatus.FAILED. The orchestrator surfaces
deployment failures as a visible warning instead (see
master_orchestrator_agent.py).
"""

from __future__ import annotations

import os
import time

from agents.base_agent import BaseAgent
from schemas.deployment_schema import DeploymentOutput
from state.pipeline_state import PipelineState
from tools.deployment.deployment_tool import DeploymentTool
from tools.deployment.model_server_registry import ModelServerRegistry
from utils.logger import logger

try:
    import mlflow
    import mlflow.pyfunc
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False


class DeploymentAgent(BaseAgent):
    """
    Loads the registered model and exposes it behind a local inference route.
    """

    def __init__(self) -> None:
        self.tool = DeploymentTool()

    def run(self, state: PipelineState) -> PipelineState:
        start_time = time.perf_counter()
        warnings: list[str] = []
        errors: list[str] = []

        state.current_agent = "DeploymentAgent"

        try:
            if not _MLFLOW_AVAILABLE:
                errors.append(
                    "mlflow package is not installed. Install it with: pip install mlflow"
                )
                return self._finalize(state, self._failed_output(errors, warnings, start_time))

            reg = state.model_registry
            if not reg.is_completed or reg.registry_status != "completed" or not reg.mlflow_model_uri:
                warnings.append(
                    "No registered model available from Model Registry — nothing to deploy."
                )
                output = DeploymentOutput(
                    deployment_status="skipped",
                    total_execution_time_seconds=round(time.perf_counter() - start_time, 4),
                    deployment_summary="Deployment skipped — no registered model available.",
                )
                return self._finalize(state, output)

            model_uri = reg.mlflow_model_uri

            # Reuse the exact tracking URI Model Registry resolved and used —
            # a "models:/..." URI only resolves against the same store it
            # was registered in.
            tracking_uri = reg.tracking_uri_used or os.getenv("MLFLOW_TRACKING_URI")
            if tracking_uri:
                try:
                    mlflow.set_tracking_uri(tracking_uri)
                except Exception as exc:
                    errors.append(f"Could not set MLflow tracking URI '{tracking_uri}': {exc}")
                    return self._finalize(state, self._failed_output(errors, warnings, start_time))

            try:
                model = mlflow.pyfunc.load_model(model_uri)
            except Exception as exc:
                errors.append(f"Failed to load model '{model_uri}': {exc}")
                logger.error("[Deployment] Model load failed: %s", exc, exc_info=True)
                return self._finalize(state, self._failed_output(errors, warnings, start_time))

            deployment_id = state.run_id
            ModelServerRegistry.register(deployment_id, model)

            endpoint = f"/predict/{deployment_id}"
            elapsed = time.perf_counter() - start_time
            summary = (
                f"Deployed '{model_uri}' locally at '{endpoint}' "
                f"(cached under deployment_id={deployment_id})."
            )

            output = DeploymentOutput(
                deployment_status="completed",
                model_uri=model_uri,
                deployment_id=deployment_id,
                endpoint=endpoint,
                total_execution_time_seconds=round(elapsed, 4),
                warnings=warnings,
                errors=errors,
                deployment_summary=summary,
            )

            state.logs.append(summary)
            logger.info("[Deployment] %s", summary)

            return self._finalize(state, output)

        except Exception as exc:
            # Truly unexpected failure (a bug, not an environment issue
            # handled above). Deployment failing should never halt the
            # pipeline — mirrors ModelRegistryAgent's non-blocking
            # philosophy — so state.status is deliberately left untouched.
            logger.error("[Deployment] Unhandled error: %s", exc, exc_info=True)
            state.logs.append(f"Deployment failed: {exc}")
            output = self._failed_output([str(exc)], warnings, start_time)
            return self.tool.execute(state, output)

    # ==========================================================
    # Internal helpers
    # ==========================================================

    def _finalize(self, state: PipelineState, output: DeploymentOutput) -> PipelineState:
        state = self.tool.execute(state, output)
        state.completed_steps.append("Deployment")
        return state

    @staticmethod
    def _failed_output(
        errors: list[str],
        warnings: list[str],
        start_time: float,
    ) -> DeploymentOutput:
        elapsed = time.perf_counter() - start_time
        return DeploymentOutput(
            deployment_status="failed",
            errors=errors,
            warnings=warnings,
            total_execution_time_seconds=round(elapsed, 4),
            deployment_summary=(
                f"Deployment failed: {'; '.join(errors) if errors else 'unknown error'}"
            ),
        )
