"""
Global Pipeline State shared across all agents.
"""

from pydantic import Field
from .base_state import BaseState
from .dataset_state import DatasetState
from .validation_state import ValidationState
from .feature_engineering_state import FeatureEngineeringState
from .model_selection_state import ModelSelectionState
from .model_training_state import ModelTrainingState
from .hyperparameter_optimization_state import HyperparameterOptimizationState
from .model_evaluation_state import ModelEvaluationState


class PipelineState(BaseState):
    """Shared state that flows through the complete LangGraph pipeline."""

    user_prompt: str = ""
    project_id: str | None = None
    session_id: str | None = None

    dataset: DatasetState = Field(default_factory=DatasetState)
    validation: ValidationState = Field(default_factory=ValidationState)
    feature_engineering: FeatureEngineeringState = Field(default_factory=FeatureEngineeringState)
    model_selection: ModelSelectionState = Field(default_factory=ModelSelectionState)
    model_training: ModelTrainingState = Field(default_factory=ModelTrainingState)
    hyperparameter_optimization: HyperparameterOptimizationState = Field(default_factory=HyperparameterOptimizationState)
    model_evaluation: ModelEvaluationState = Field(default_factory=ModelEvaluationState)

    current_agent: str = ""
    completed_steps: list[str] = Field(default_factory=list)
    logs: list[str] = Field(default_factory=list)

    mlflow_run_id: str | None = None
    model_name: str | None = None
    model_path: str | None = None
    metrics: dict = Field(default_factory=dict)