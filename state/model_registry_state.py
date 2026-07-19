"""
State used by the Model Registry Agent.

Written once by ModelRegistryAgent and consumed read-only by
downstream agents (Deployment) and by server/services/orchestration_service.py
(which also mirrors the key pointers onto the top-level PipelineState
fields: mlflow_run_id, model_name, model_path — those already existed
on PipelineState before this agent was built).
"""

from pydantic import Field

from .base_state import BaseState


class ModelRegistryState(BaseState):
    """
    Stores the results of the model registration step.
    """

    # Whether registration completed (or was validly skipped — no model to register)
    is_completed: bool = False

    # "completed" | "failed" | "skipped"
    registry_status: str | None = None

    registered_model_name: str | None = None

    # Version MLflow assigned (auto-incremented per registered_model_name)
    model_version: int | None = None

    mlflow_run_id: str | None = None

    # "models:/<name>/<version>" — preferred, always resolves to this version
    mlflow_model_uri: str | None = None

    # "runs:/<run_id>/model" — run-relative, useful even if later archived
    mlflow_run_model_uri: str | None = None

    # The tracking URI actually used (server or local file-store fallback)
    tracking_uri_used: str | None = None

    # Whether fitted feature-engineering transformers were bundled with the
    # estimator (model accepts raw input) vs. raw-estimator-only (expects
    # already-transformed input)
    bundled_transformers: bool = False

    logged_params: dict[str, str] = Field(default_factory=dict)
    logged_metrics: dict[str, float] = Field(default_factory=dict)

    total_execution_time_seconds: float = 0.0

    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    summary: str | None = None