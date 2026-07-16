from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    prompt: str
    dataset_id: str | None = None


class ModelSelectionResult(BaseModel):
    task_type: str | None = None
    primary_model_name: str | None = None
    primary_model_library: str | None = None
    ranking: list[str] = Field(default_factory=list)
    reasoning: str | None = None
    confidence: float = 0.0
    summary: str | None = None


class ModelTrainingResult(BaseModel):
    training_status: str | None = None
    train_samples: int = 0
    test_samples: int = 0
    trained_models: list[dict[str, Any]] = Field(default_factory=list)
    failed_models: list[dict[str, Any]] = Field(default_factory=list)
    total_execution_time_seconds: float = 0.0
    summary: str | None = None


class HyperparameterOptimizationResult(BaseModel):
    optimization_status: str | None = None
    scoring_metric: str | None = None
    best_overall_model_name: str | None = None
    best_overall_score: float = 0.0
    optimized_models: list[dict[str, Any]] = Field(default_factory=list)
    failed_models: list[dict[str, Any]] = Field(default_factory=list)
    total_execution_time_seconds: float = 0.0
    summary: str | None = None


class ModelEvaluationResult(BaseModel):
    evaluation_status: str | None = None
    primary_metric: str | None = None
    best_model_name: str | None = None
    best_model_metrics: dict[str, float] = Field(default_factory=dict)
    comparison_table: list[dict[str, Any]] = Field(default_factory=list)
    summary: str | None = None


class ChatResponse(BaseModel):
    user_prompt: str
    assistant_message: str | None = None

    run_id : str
    status: str
    execution_time: float | None = None
    completed_steps: list[str]
    logs: list[str]

    dataset_name: str | None = None
    rows: int | None = None
    columns: int | None = None

    problem_type: str | None = None
    target_column: str | None = None

    model_selection: ModelSelectionResult | None = None
    model_training: ModelTrainingResult | None = None
    hyperparameter_optimization: HyperparameterOptimizationResult | None = None
    model_evaluation: ModelEvaluationResult | None = None

    mlflow_run_id: str | None = None
    model_name: str | None = None
    model_path: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
