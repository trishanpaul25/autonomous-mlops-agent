"""
Model Registry Tool.

Transfers a ModelRegistryOutput (built by ModelRegistryAgent) into
ModelRegistryState. Pure writer — makes no decisions, calls no MLflow
APIs, performs no IO. Mirrors ModelEvaluationTool / HyperparameterOptimizationTool.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from schemas.model_registry_schema import ModelRegistryOutput
from tools.base_tool import BaseTool
from utils.logger import logger

if TYPE_CHECKING:
    from state.pipeline_state import PipelineState


class ModelRegistryTool(BaseTool):
    """
    Writes the structured registration result into PipelineState.
    """

    def execute(
        self,
        state: "PipelineState",
        output: ModelRegistryOutput,
    ) -> "PipelineState":
        mr = state.model_registry

        mr.registry_status = output.registry_status
        mr.registered_model_name = output.registered_model_name
        mr.model_version = output.model_version
        mr.mlflow_run_id = output.mlflow_run_id
        mr.mlflow_model_uri = output.mlflow_model_uri
        mr.mlflow_run_model_uri = output.mlflow_run_model_uri
        mr.tracking_uri_used = output.tracking_uri_used
        mr.bundled_transformers = output.bundled_transformers
        mr.logged_params = output.logged_params
        mr.logged_metrics = output.logged_metrics
        mr.total_execution_time_seconds = output.total_execution_time_seconds
        mr.warnings = output.warnings
        mr.errors = output.errors
        mr.summary = output.registry_summary

        # "skipped" counts as completed for pipeline-progression purposes
        # (mirrors HyperparameterOptimizationState's convention) — nothing
        # is broken if there was legitimately no model to register.
        mr.is_completed = output.registry_status in ("completed", "skipped")

        state.model_registry = mr

        # Mirror the key pointers onto PipelineState's top-level fields.
        # These already existed on PipelineState before this agent was
        # built, and orchestration_service.py already reads
        # result.mlflow_run_id directly — this is what finally populates it.
        if output.registry_status == "completed":
            state.mlflow_run_id = output.mlflow_run_id
            state.model_name = output.registered_model_name
            state.model_path = output.mlflow_model_uri or output.mlflow_run_model_uri

        logger.info(
            "ModelRegistryTool: state updated. Status: %s | Version: %s | Summary: %s",
            mr.registry_status,
            mr.model_version,
            mr.summary,
        )

        return state