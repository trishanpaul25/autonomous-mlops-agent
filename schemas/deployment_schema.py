"""
Schema for the Deployment Agent.

No LLM is used — deploying is a deterministic operation: load the
registered MLflow model into memory and expose it behind a local
FastAPI inference route.
"""

from typing import Literal

from pydantic import BaseModel, Field


class DeploymentOutput(BaseModel):
    """
    Top-level structured result produced by the Deployment Agent.
    """

    deployment_status: Literal["completed", "failed", "skipped"] = Field(
        ...,
        description=(
            "'completed' — model successfully loaded and cached behind a "
            "local inference route. "
            "'failed'    — an exception occurred (e.g. model URI could not "
            "be loaded); pipeline continues, nothing was deployed. "
            "'skipped'   — no registered model available from Model "
            "Registry to deploy."
        ),
    )

    model_uri: str = Field(
        default="", description="MLflow model URI that was loaded (e.g. 'models:/titanic-survival/3')."
    )

    deployment_id: str = Field(
        default="", description="Key the model was cached under in the in-process model server registry."
    )

    endpoint: str = Field(
        default="", description="Local FastAPI inference route for this deployment, e.g. '/predict/<deployment_id>'."
    )

    total_execution_time_seconds: float = Field(default=0.0, ge=0.0)

    warnings: list[str] = Field(default_factory=list)

    errors: list[str] = Field(default_factory=list)

    deployment_summary: str = Field(..., description="Human-readable summary for logging/API display.")
