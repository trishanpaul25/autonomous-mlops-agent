"""
Schema for the Model Registry Agent.

No LLM is used — registration is a deterministic MLflow operation.
"""

from typing import Literal

from pydantic import BaseModel, Field


class ModelRegistryOutput(BaseModel):
    """
    Top-level structured result produced by the Model Registry Agent.
    """

    registry_status: Literal["completed", "failed", "skipped"] = Field(
        ...,
        description=(
            "'completed' — model successfully logged and registered in MLflow. "
            "'failed'    — an exception occurred (e.g. tracking server "
            "unreachable); pipeline continues, nothing was registered. "
            "'skipped'   — no best model available from Model Evaluation "
            "to register."
        ),
    )

    registered_model_name: str = Field(
        default="", description="Name the model was registered under in MLflow's Model Registry."
    )

    model_version: int | None = Field(
        default=None,
        description="Version number MLflow assigned to this registration (auto-incremented per registered_model_name).",
    )

    mlflow_run_id: str = Field(default="", description="MLflow run ID the model/metrics/params were logged under.")

    mlflow_model_uri: str = Field(
        default="",
        description="Registry URI for this exact version, e.g. 'models:/titanic-survival/3'. Preferred for downstream loading — always resolves to this specific version.",
    )

    mlflow_run_model_uri: str = Field(
        default="",
        description="Run-relative URI, e.g. 'runs:/<run_id>/model'. Useful for traceability even if the model is later archived from the registry.",
    )

    tracking_uri_used: str = Field(
        default="", description="The MLflow tracking URI actually used for this run (server or local fallback)."
    )

    bundled_transformers: bool = Field(
        default=False,
        description=(
            "True if the fitted feature-engineering transformers were "
            "bundled with the estimator (model accepts raw input). False "
            "if only the raw estimator was logged (model expects "
            "already-transformed input) — happens when no feature "
            "engineering config/fitted transformers were available."
        ),
    )

    logged_params: dict[str, str] = Field(
        default_factory=dict, description="Hyperparameters logged to the MLflow run."
    )

    logged_metrics: dict[str, float] = Field(
        default_factory=dict, description="Evaluation metrics logged to the MLflow run."
    )

    total_execution_time_seconds: float = Field(default=0.0, ge=0.0)

    warnings: list[str] = Field(default_factory=list)

    errors: list[str] = Field(default_factory=list)

    registry_summary: str = Field(..., description="Human-readable summary for logging/API display.")