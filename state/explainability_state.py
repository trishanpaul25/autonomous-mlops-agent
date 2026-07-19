"""
State used by the Explainability Agent.

Stores every artifact produced during the explainability step:
  - Unified, cross-method feature importance ranking
  - Raw SHAP artifacts and which explainer was used
  - Permutation / native / coefficient importance results
  - Partial dependence curves
  - Global (model-level) and local (per-sample) explanations
  - LLM-generated narratives (technical / business / non-technical)
  - Visualization-ready data structures (JSON-serializable)

Design notes
------------
Mirrors ModelEvaluationState: everything is stored as plain,
JSON-serializable dicts/lists (no numpy arrays, no shap objects, no
fitted estimators) so the state remains safely serializable across the
LangGraph checkpointer / API layer. Downstream consumers (API layer,
dashboards, Report Generation Agent) reconstruct arrays as needed.

The fitted "best model" itself is NOT stored here — the Explainability
Agent only reads it (from wherever HyperparameterOptimizationState /
ModelEvaluationState keep it) and never writes it back.

This state is written once by ExplainabilityAgent and consumed
read-only by downstream agents (Report Generation).
"""

from typing import Any

from pydantic import Field

from .base_state import BaseState


class ExplainabilityState(BaseState):
    """
    Stores the results of the explainability step.

    Written once by the ExplainabilityAgent and consumed read-only
    by downstream agents (Report Generation).
    """

    # Whether explainability completed (at least one technique succeeded)
    is_completed: bool = False

    # Overall explainability outcome
    # "completed" — every requested technique succeeded
    # "partial"   — at least one technique succeeded, at least one skipped/failed
    # "failed"    — no importance method could be computed at all
    explainability_status: str | None = None

    # ML task type used for method selection (mirrors model_evaluation.task_type)
    task_type: str | None = None

    # Which importance sources actually produced results
    shap_computed: bool = False
    permutation_computed: bool = False
    native_importance_computed: bool = False
    coefficient_computed: bool = False
    partial_dependence_computed: bool = False

    # Which SHAP explainer was used: "tree" | "linear" | "kernel" | "none"
    shap_explainer_type: str | None = None

    # method_name -> human-readable reason it was skipped or failed
    # e.g. {"shap": "shap package is not installed",
    #       "coefficient": "model does not expose coef_"}
    skipped_methods: dict[str, str] = Field(default_factory=dict)

    # Unified, cross-method feature ranking, ordered by overall_rank ascending.
    # Each entry:
    # {"feature_name": str, "shap_score": float | None,
    #  "permutation_importance": float | None, "native_importance": float | None,
    #  "coefficient_importance": float | None, "overall_score": float,
    #  "overall_rank": int}
    feature_ranking: list[dict[str, Any]] = Field(default_factory=list)

    # Raw SHAP artifacts.
    # {"explainer_type": str, "expected_value": [...], "feature_names": [...],
    #  "global_shap_values": [[...], ...], "mean_abs_shap_importance": {...},
    #  "sample_indices": [...]}
    shap_values: dict[str, Any] = Field(default_factory=dict)

    # {"mean_importance": {...}, "std_importance": {...}, "n_repeats": int}
    permutation_importance: dict[str, Any] = Field(default_factory=dict)

    # {"importance": {feature_name: value, ...}}
    native_feature_importance: dict[str, Any] = Field(default_factory=dict)

    # {"raw_coefficients": {...}, "normalized_importance": {...}}
    coefficient_importance: dict[str, Any] = Field(default_factory=dict)

    # {"curves": [{"feature_name": str, "grid_values": [...], "pd_values": [...]}]}
    partial_dependence: dict[str, Any] = Field(default_factory=dict)

    # {"most_important_features": [...], "least_important_features": [...],
    #  "positively_influential_features": [...],
    #  "negatively_influential_features": [...], "summary": str}
    global_explanation: dict[str, Any] = Field(default_factory=dict)

    # Per-sample explanations.
    # Each entry: {"sample_index": int, "predicted_value": float,
    #              "predicted_label": str | None,
    #              "top_contributing_features": [...],
    #              "contribution_values": {...},
    #              "prediction_explanation": str}
    local_explanations: list[dict[str, Any]] = Field(default_factory=list)

    # Visualization-ready data structures (JSON-serializable dicts of lists),
    # mirroring ModelEvaluationState.visualization_data:
    # {
    #   "shap_summary_plot": {...} | None,
    #   "shap_beeswarm_plot": {...} | None,
    #   "shap_waterfall_plot": {...} | None,
    #   "shap_force_plot": {...} | None,
    #   "feature_importance_bar_chart": {...} | None,
    #   "permutation_importance_plot": {...} | None,
    #   "partial_dependence_plot": {...} | None,
    # }
    visualization_data: dict[str, Any] = Field(default_factory=dict)

    # LLM-generated narratives (None when no LLM/API key is configured)
    technical_explanation: str | None = None
    business_explanation: str | None = None
    non_technical_explanation: str | None = None

    # Wall-clock time for the entire explainability step (seconds)
    total_execution_time_seconds: float = 0.0

    # Collected error messages (blocking failures)
    errors: list[str] = Field(default_factory=list)

    # Non-blocking warnings (e.g. a technique was skipped/fell back)
    warnings: list[str] = Field(default_factory=list)

    # Short human-readable summary for logging and API display
    summary: str | None = None