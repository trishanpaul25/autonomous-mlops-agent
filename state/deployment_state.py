"""
State used by the Deployment Agent.

Written once by DeploymentAgent and read by server/services/orchestration_service.py
to persist a Deployment row (server/models/deployment.py) once the pipeline
finishes. Mirrors ModelRegistryState in shape and conventions.
"""

from pydantic import Field

from .base_state import BaseState


class DeploymentState(BaseState):
    """
    Stores the results of the local FastAPI serving step.
    """

    # Whether deployment completed (or was validly skipped — no model to deploy)
    is_completed: bool = False

    # "completed" | "failed" | "skipped"
    deployment_status: str | None = None

    # The MLflow model URI that was loaded and deployed (from ModelRegistryState)
    model_uri: str | None = None

    # Key the model was cached under in the in-process ModelServerRegistry —
    # also the path segment used to build `endpoint`.
    deployment_id: str | None = None

    # Local inference route, e.g. "/predict/<deployment_id>"
    endpoint: str | None = None

    total_execution_time_seconds: float = 0.0

    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    summary: str | None = None
