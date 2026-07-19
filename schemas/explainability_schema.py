"""
explainability_schema.py

Pydantic data contracts for the Explainability Agent.

This module contains ONLY data definitions — no computation, no I/O, no
LangGraph or LLM awareness. Every artifact produced by the Explainability
Agent (and everything a future Report Generation Agent will consume) is
defined here so the contract between agents is explicit and versioned.

All numeric payloads are stored as native Python types (float, int, list,
dict) rather than numpy/pandas/shap objects, so downstream consumers
(LangGraph state, report generation, API responses) never need
explainability-specific imports to read them.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #


class TaskType(str, Enum):
    """Mirrors the task_type already present on PipelineState."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class ImportanceType(str, Enum):
    """The distinct sources of feature importance this agent can produce."""

    SHAP = "shap"
    PERMUTATION = "permutation"
    NATIVE = "native"
    COEFFICIENT = "coefficient"


class ShapExplainerType(str, Enum):
    """Which SHAP explainer implementation was used."""

    TREE = "tree"
    LINEAR = "linear"
    KERNEL = "kernel"
    NONE = "none"


class AgentStatus(str, Enum):
    """Terminal status of the Explainability Agent run, consistent with the
    status vocabulary used by the other agents in the pipeline."""

    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"
    SKIPPED = "skipped"


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


class ExplainabilityConfig(BaseModel):
    """Tunable behavior for the Explainability Agent. Instantiate with
    defaults or override per-run without touching agent code."""

    enable_shap: bool = True
    enable_permutation_importance: bool = True
    enable_native_importance: bool = True
    enable_coefficient_importance: bool = True
    enable_partial_dependence: bool = True
    enable_llm_explanation: bool = True

    permutation_n_repeats: int = Field(default=10, ge=1, le=100)
    random_state: int = 42

    # KernelExplainer is O(n_samples * n_background) and can be extremely
    # slow on large data; these bounds keep it tractable and are the
    # trigger for graceful fallback if exceeded.
    kernel_shap_background_size: int = Field(default=50, ge=1, le=500)
    kernel_shap_max_samples: int = Field(default=100, ge=1, le=1000)
    kernel_shap_timeout_seconds: int = Field(default=120, ge=1)

    max_rows_for_shap: int = Field(default=5000, ge=1)
    top_n_features_for_pdp: int = Field(default=5, ge=1, le=50)
    top_n_features_for_local_explanation: int = Field(default=10, ge=1, le=100)

    class Config:
        frozen = False


# --------------------------------------------------------------------------- #
# Feature importance building blocks
# --------------------------------------------------------------------------- #


class FeatureImportanceScore(BaseModel):
    """A single (feature, importance) pair from one importance source."""

    feature_name: str
    importance_score: float
    importance_type: ImportanceType
    rank: Optional[int] = None


class SHAPResult(BaseModel):
    """Container for all SHAP-derived artifacts."""

    explainer_type: ShapExplainerType
    expected_value: List[float] = Field(default_factory=list)
    # global_shap_values[i][j] = SHAP value for sample i, feature j
    global_shap_values: List[List[float]] = Field(default_factory=list)
    mean_abs_shap_importance: Dict[str, float] = Field(default_factory=dict)
    feature_names: List[str] = Field(default_factory=list)
    sample_indices: List[int] = Field(default_factory=list)
    computed: bool = True
    skipped_reason: Optional[str] = None


class PermutationImportanceResult(BaseModel):
    """Result of sklearn-style permutation importance."""

    mean_importance: Dict[str, float] = Field(default_factory=dict)
    std_importance: Dict[str, float] = Field(default_factory=dict)
    n_repeats: int = 0
    computed: bool = True
    skipped_reason: Optional[str] = None


class NativeImportanceResult(BaseModel):
    """Result of model.feature_importances_ (tree ensembles)."""

    importance: Dict[str, float] = Field(default_factory=dict)
    computed: bool = True
    skipped_reason: Optional[str] = None


class CoefficientImportanceResult(BaseModel):
    """Result of model.coef_ (linear models), normalized to [0, 1]."""

    raw_coefficients: Dict[str, float] = Field(default_factory=dict)
    normalized_importance: Dict[str, float] = Field(default_factory=dict)
    computed: bool = True
    skipped_reason: Optional[str] = None


class PartialDependenceResult(BaseModel):
    """Partial dependence curve for a single feature."""

    feature_name: str
    grid_values: List[float]
    pd_values: List[float]


class PartialDependenceCollection(BaseModel):
    results: List[PartialDependenceResult] = Field(default_factory=list)
    computed: bool = True
    skipped_reason: Optional[str] = None


class UnifiedFeatureRanking(BaseModel):
    """One row of the unified, cross-method feature ranking table."""

    feature_name: str
    shap_score: Optional[float] = None
    permutation_importance: Optional[float] = None
    native_importance: Optional[float] = None
    coefficient_importance: Optional[float] = None
    overall_score: float
    overall_rank: int


# --------------------------------------------------------------------------- #
# Explanations
# --------------------------------------------------------------------------- #


class GlobalExplanation(BaseModel):
    """Model-level narrative summary, computed purely from numeric results
    (no LLM involved in producing these fields)."""

    most_important_features: List[str] = Field(default_factory=list)
    least_important_features: List[str] = Field(default_factory=list)
    positively_influential_features: List[str] = Field(default_factory=list)
    negatively_influential_features: List[str] = Field(default_factory=list)
    summary: str = ""


class LocalExplanationRequest(BaseModel):
    sample_index: int = Field(ge=0)


class LocalExplanationResult(BaseModel):
    """Explanation for a single prediction."""

    sample_index: int
    predicted_value: float
    predicted_label: Optional[str] = None
    top_contributing_features: List[str] = Field(default_factory=list)
    contribution_values: Dict[str, float] = Field(default_factory=dict)
    prediction_explanation: str = ""


class LLMExplanationOutput(BaseModel):
    """Human-readable narration of already-computed explanations. The LLM
    is a formatter here, never a source of numeric truth."""

    technical_explanation: str = ""
    business_explanation: str = ""
    non_technical_explanation: str = ""
    generated: bool = True
    skipped_reason: Optional[str] = None


# --------------------------------------------------------------------------- #
# Visualization payloads (data only — no rendering)
# --------------------------------------------------------------------------- #


class VisualizationData(BaseModel):
    """Pre-computed, JSON-serializable data for each plot type. Consumers
    (a UI, a report agent) render these; this agent never plots."""

    shap_summary_plot: Optional[Dict[str, Any]] = None
    shap_beeswarm_plot: Optional[Dict[str, Any]] = None
    shap_waterfall_plot: Optional[Dict[str, Any]] = None
    shap_force_plot: Optional[Dict[str, Any]] = None
    feature_importance_bar_chart: Optional[Dict[str, Any]] = None
    permutation_importance_plot: Optional[Dict[str, Any]] = None
    partial_dependence_plot: Optional[Dict[str, Any]] = None


# --------------------------------------------------------------------------- #
# Top-level agent output
# --------------------------------------------------------------------------- #


class ExplainabilityResults(BaseModel):
    """The single object written to PipelineState.explainability_results.
    Everything else on PipelineState (feature_importance, shap_values, ...)
    is a convenience projection of fields already present here — see
    pipeline_state_patch.py for how those projections are populated.
    """

    agent_status: AgentStatus
    task_type: TaskType

    feature_ranking: List[UnifiedFeatureRanking] = Field(default_factory=list)

    shap_result: Optional[SHAPResult] = None
    permutation_result: Optional[PermutationImportanceResult] = None
    native_importance_result: Optional[NativeImportanceResult] = None
    coefficient_result: Optional[CoefficientImportanceResult] = None
    partial_dependence: Optional[PartialDependenceCollection] = None

    global_explanation: GlobalExplanation = Field(default_factory=GlobalExplanation)
    local_explanations: List[LocalExplanationResult] = Field(default_factory=list)
    llm_explanation: LLMExplanationOutput = Field(default_factory=LLMExplanationOutput)

    visualization_data: VisualizationData = Field(default_factory=VisualizationData)

    execution_logs: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)

    execution_time_seconds: float = 0.0
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("feature_ranking")
    @classmethod
    def _ranking_sorted(cls, v: List[UnifiedFeatureRanking]) -> List[UnifiedFeatureRanking]:
        return sorted(v, key=lambda r: r.overall_rank)


class ExplainabilityAgentOutput(BaseModel):
    """Flat, report-friendly view matching the OUTPUT SCHEMA requested in
    the spec. Built from ExplainabilityResults via
    ExplainabilityAgentOutput.from_results(). Kept separate from
    ExplainabilityResults so PipelineState always carries the rich object
    while lightweight consumers (e.g. a REST response) can use this one.
    """

    feature_name: List[str]
    importance_score: List[float]
    importance_type: List[str]
    ranking: List[int]
    summary: str
    technical_explanation: str
    business_explanation: str
    local_explanations: List[LocalExplanationResult]
    visualization_data: VisualizationData
    execution_time: float

    @classmethod
    def from_results(cls, results: ExplainabilityResults) -> "ExplainabilityAgentOutput":
        return cls(
            feature_name=[r.feature_name for r in results.feature_ranking],
            importance_score=[r.overall_score for r in results.feature_ranking],
            importance_type=["combined" for _ in results.feature_ranking],
            ranking=[r.overall_rank for r in results.feature_ranking],
            summary=results.global_explanation.summary,
            technical_explanation=results.llm_explanation.technical_explanation,
            business_explanation=results.llm_explanation.business_explanation,
            local_explanations=results.local_explanations,
            visualization_data=results.visualization_data,
            execution_time=results.execution_time_seconds,
        )