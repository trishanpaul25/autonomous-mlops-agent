"""
Deployment Tool.

Transfers a DeploymentOutput (built by DeploymentAgent) into
DeploymentState. Pure writer — makes no decisions, loads no models,
performs no IO. Mirrors ModelRegistryTool.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from schemas.deployment_schema import DeploymentOutput
from tools.base_tool import BaseTool
from utils.logger import logger

if TYPE_CHECKING:
    from state.pipeline_state import PipelineState


class DeploymentTool(BaseTool):
    """
    Writes the structured deployment result into PipelineState.
    """

    def execute(
        self,
        state: "PipelineState",
        output: DeploymentOutput,
    ) -> "PipelineState":
        dep = state.deployment

        dep.deployment_status = output.deployment_status
        dep.model_uri = output.model_uri
        dep.deployment_id = output.deployment_id
        dep.endpoint = output.endpoint
        dep.total_execution_time_seconds = output.total_execution_time_seconds
        dep.warnings = output.warnings
        dep.errors = output.errors
        dep.summary = output.deployment_summary

        # "skipped" counts as completed for pipeline-progression purposes
        # (mirrors ModelRegistryState's convention) — nothing is broken if
        # there was legitimately no registered model to deploy.
        dep.is_completed = output.deployment_status in ("completed", "skipped")

        state.deployment = dep

        logger.info(
            "DeploymentTool: state updated. Status: %s | Endpoint: %s | Summary: %s",
            dep.deployment_status,
            dep.endpoint,
            dep.summary,
        )

        return state
