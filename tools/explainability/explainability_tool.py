"""
explainability_tool.py

Pure computation layer for the Explainability Agent.

Design intent (SRP / Clean Architecture):
    - This module knows how to COMPUTE explainability artifacts from a
      trained model and data. It knows nothing about PipelineState,
      LangGraph, or LLMs.
    - Every public function returns a schema object from
      explainability_schema.py, or None-ish "not computed" variants with a
      `skipped_reason`, so the orchestrating agent can implement fallback
      policy without this module needing to know about that policy.
    - Third-party ML libraries (shap, sklearn) are imported lazily inside
      the functions that need them, so importing this module never fails
      just because an optional dependency is missing.

All heavy numeric results are converted to native Python types via
explainability_utils.to_jsonable before being placed on schema objects.
"""

from __future__ import annotations

import concurrent.futures
import logging
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence
from schemas.explainability_schema import CoefficientImportanceResult,ExplainabilityConfig,GlobalExplanation,ImportanceType,LocalExplanationResult,NativeImportanceResult,PartialDependenceCollection,PartialDependenceResult,PermutationImportanceResult,SHAPResult,ShapExplainerType,TaskType,UnifiedFeatureRanking,VisualizationData
from utils.explainability_utils import normalize_to_unit_range, safe_mean, to_jsonable


# --------------------------------------------------------------------------- #
# Model capability detection
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ModelCapabilities:
    """Internal capability flags used to route to the right explainability
    technique. Not exposed on PipelineState — only the resulting artifacts
    are."""

    is_tree_based: bool
    is_linear: bool
    has_feature_importances: bool
    has_coef: bool
    has_predict_proba: bool


_TREE_MODULE_HINTS = (
    "sklearn.tree",
    "sklearn.ensemble",
    "xgboost",
    "lightgbm",
    "catboost",
)
_TREE_ATTR_HINTS = ("estimators_", "tree_", "booster_", "get_booster")


def detect_model_capabilities(model: Any) -> ModelCapabilities:
    """Inspect a fitted model and determine which explainability paths are
    viable. Uses duck-typing (attribute presence + module name heuristics)
    rather than isinstance checks against every possible library, so it
    works across sklearn, XGBoost, LightGBM, and CatBoost without hard
    dependencies on any of them.
    """
    module_name = type(model).__module__.lower()

    is_tree_based = any(hint in module_name for hint in _TREE_MODULE_HINTS) or any(
        hasattr(model, attr) for attr in _TREE_ATTR_HINTS
    )
    has_feature_importances = hasattr(model, "feature_importances_")
    has_coef = hasattr(model, "coef_")
    # A model can technically expose both coef_ and tree attributes only in
    # pathological wrapper cases; treat explicit tree evidence as authoritative.
    is_linear = has_coef and not is_tree_based
    has_predict_proba = hasattr(model, "predict_proba")

    return ModelCapabilities(
        is_tree_based=is_tree_based,
        is_linear=is_linear,
        has_feature_importances=has_feature_importances,
        has_coef=has_coef,
        has_predict_proba=has_predict_proba,
    )


# --------------------------------------------------------------------------- #
# SHAP
# --------------------------------------------------------------------------- #


def compute_shap(
    model: Any,
    X_train: Any,
    X_test: Any,
    feature_names: Sequence[str],
    capabilities: ModelCapabilities,
    config: ExplainabilityConfig,
    logger: logging.Logger,
) -> SHAPResult:
    """Compute SHAP values, automatically selecting TreeExplainer,
    LinearExplainer, or KernelExplainer based on model type. Always
    returns a SHAPResult; on any failure `computed=False` with a
    human-readable `skipped_reason` is set so the caller can fall back.
    """
    try:
        import shap  # noqa: F401  (import guarded — optional dependency)
    except ImportError:
        logger.warning("SHAP package not installed — skipping SHAP computation.")
        return SHAPResult(explainer_type=ShapExplainerType.NONE, computed=False,
                           skipped_reason="shap package is not installed")

    X_test_capped = X_test[: config.max_rows_for_shap]

    try:
        if capabilities.is_tree_based:
            logger.info("Detected tree-based model — using TreeExplainer.")
            explainer = shap.TreeExplainer(model)
            explainer_type = ShapExplainerType.TREE
            raw_values = explainer.shap_values(X_test_capped)
        elif capabilities.is_linear:
            logger.info("Detected linear model — using LinearExplainer.")
            explainer = shap.LinearExplainer(model, X_train)
            explainer_type = ShapExplainerType.LINEAR
            raw_values = explainer.shap_values(X_test_capped)
        else:
            logger.info("Model type not tree/linear — using KernelExplainer.")
            explainer_type = ShapExplainerType.KERNEL
            raw_values = _compute_kernel_shap(
                model, X_train, X_test_capped, config, logger
            )
            if raw_values is None:
                return SHAPResult(
                    explainer_type=ShapExplainerType.KERNEL,
                    computed=False,
                    skipped_reason=(
                        f"KernelExplainer exceeded "
                        f"{config.kernel_shap_timeout_seconds}s timeout"
                    ),
                )
    except Exception as exc:  # noqa: BLE001 — must never crash the pipeline
        logger.error("SHAP computation failed: %s", exc)
        return SHAPResult(
            explainer_type=ShapExplainerType.NONE,
            computed=False,
            skipped_reason=f"SHAP computation raised: {exc}",
        )

    logger.info("Computed SHAP values using %s.", explainer_type.value)

    # Multiclass classifiers return a list of arrays (one per class); binary
    # classification / regression return a single array. Normalize to a
    # single [n_samples, n_features] matrix by averaging class-wise
    # magnitude, which keeps downstream shapes consistent.
    values_matrix = _normalize_shap_output(raw_values)
    values_matrix = to_jsonable(values_matrix)

    expected_value = getattr(explainer, "expected_value", None)
    if isinstance(expected_value, (list, tuple)):
        expected_value_list = [float(v) for v in to_jsonable(expected_value)]
    elif expected_value is None:
        expected_value_list = []
    else:
        expected_value_list = [float(to_jsonable(expected_value))]

    mean_abs_importance = _mean_abs_importance(values_matrix, feature_names)

    return SHAPResult(
        explainer_type=explainer_type,
        expected_value=expected_value_list,
        global_shap_values=values_matrix,
        mean_abs_shap_importance=mean_abs_importance,
        feature_names=list(feature_names),
        sample_indices=list(range(len(values_matrix))),
        computed=True,
    )


def _compute_kernel_shap(
    model: Any,
    X_train: Any,
    X_test_capped: Any,
    config: ExplainabilityConfig,
    logger: logging.Logger,
) -> Optional[Any]:
    """Run KernelExplainer with a bounded background set, a sample cap, and
    a soft wall-clock timeout (KernelExplainer has no native timeout and
    can be extremely slow on wide/large data).
    """
    import shap

    background = shap.sample(X_train, config.kernel_shap_background_size)
    samples = X_test_capped[: config.kernel_shap_max_samples]

    predict_fn = model.predict_proba if hasattr(model, "predict_proba") else model.predict
    explainer = shap.KernelExplainer(predict_fn, background)

    def _run() -> Any:
        return explainer.shap_values(samples)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_run)
        try:
            return future.result(timeout=config.kernel_shap_timeout_seconds)
        except concurrent.futures.TimeoutError:
            logger.warning(
                "KernelExplainer timed out after %ss.",
                config.kernel_shap_timeout_seconds,
            )
            return None


def _normalize_shap_output(raw_values: Any) -> List[List[float]]:
    """Collapse multiclass SHAP output into a single [n_samples, n_features]
    matrix by averaging absolute magnitude across classes; pass single-array
    output (binary/regression) through unchanged (as nested lists).

    Handles both SHAP output conventions seen across versions:
      - Older/list convention: a list of length n_classes, each element an
        (n_samples, n_features) array.
      - Newer/ndarray convention (SHAP >= ~0.42 for many classifiers): a
        single (n_samples, n_features, n_classes) array.
    Both are collapsed identically, by averaging |value| over the class axis.
    """
    if isinstance(raw_values, list):
        arrays = [to_jsonable(a) for a in raw_values]
        n_samples = len(arrays[0])
        n_features = len(arrays[0][0])
        collapsed: List[List[float]] = []
        for i in range(n_samples):
            row = []
            for j in range(n_features):
                row.append(safe_mean(abs(a[i][j]) for a in arrays))
            collapsed.append(row)
        return collapsed

    values = to_jsonable(raw_values)

    # Newer SHAP convention: single array shaped (n_samples, n_features, n_classes).
    if (
        isinstance(values, list)
        and values
        and isinstance(values[0], list)
        and values[0]
        and isinstance(values[0][0], list)
    ):
        n_samples = len(values)
        n_features = len(values[0])
        n_classes = len(values[0][0])
        collapsed = []
        for i in range(n_samples):
            row = []
            for j in range(n_features):
                row.append(
                    safe_mean(abs(values[i][j][c]) for c in range(n_classes))
                )
            collapsed.append(row)
        return collapsed

    return values


def _mean_abs_importance(
    values_matrix: List[List[float]], feature_names: Sequence[str]
) -> dict:
    if not values_matrix:
        return {name: 0.0 for name in feature_names}
    n_features = len(feature_names)
    sums = [0.0] * n_features
    for row in values_matrix:
        for j in range(n_features):
            sums[j] += abs(row[j])
    n = len(values_matrix)
    return {feature_names[j]: sums[j] / n for j in range(n_features)}


# --------------------------------------------------------------------------- #
# Permutation importance
# --------------------------------------------------------------------------- #


def compute_permutation_importance(
    model: Any,
    X_test: Any,
    y_test: Any,
    feature_names: Sequence[str],
    task_type: TaskType,
    config: ExplainabilityConfig,
    logger: logging.Logger,
) -> PermutationImportanceResult:
    try:
        from sklearn.inspection import permutation_importance
    except ImportError:
        logger.warning("scikit-learn not available — skipping permutation importance.")
        return PermutationImportanceResult(
            computed=False, skipped_reason="scikit-learn is not installed"
        )

    scoring = "accuracy" if task_type == TaskType.CLASSIFICATION else "r2"

    try:
        result = permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=config.permutation_n_repeats,
            random_state=config.random_state,
            scoring=scoring,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Permutation importance failed: %s", exc)
        return PermutationImportanceResult(
            computed=False, skipped_reason=f"permutation_importance raised: {exc}"
        )

    logger.info("Computed permutation importance.")

    means = to_jsonable(result.importances_mean)
    stds = to_jsonable(result.importances_std)

    return PermutationImportanceResult(
        mean_importance={feature_names[i]: means[i] for i in range(len(feature_names))},
        std_importance={feature_names[i]: stds[i] for i in range(len(feature_names))},
        n_repeats=config.permutation_n_repeats,
        computed=True,
    )


# --------------------------------------------------------------------------- #
# Native importance (tree ensembles)
# --------------------------------------------------------------------------- #


def compute_native_importance(
    model: Any,
    feature_names: Sequence[str],
    capabilities: ModelCapabilities,
    logger: logging.Logger,
) -> NativeImportanceResult:
    if not capabilities.has_feature_importances:
        return NativeImportanceResult(
            computed=False,
            skipped_reason="model does not expose feature_importances_",
        )

    try:
        raw = to_jsonable(model.feature_importances_)
    except Exception as exc:  # noqa: BLE001
        logger.error("Reading feature_importances_ failed: %s", exc)
        return NativeImportanceResult(
            computed=False, skipped_reason=f"feature_importances_ access raised: {exc}"
        )

    logger.info("Collected native feature_importances_.")
    return NativeImportanceResult(
        importance={feature_names[i]: raw[i] for i in range(len(feature_names))},
        computed=True,
    )


# --------------------------------------------------------------------------- #
# Coefficient importance (linear models)
# --------------------------------------------------------------------------- #


def compute_coefficient_importance(
    model: Any,
    feature_names: Sequence[str],
    capabilities: ModelCapabilities,
    logger: logging.Logger,
) -> CoefficientImportanceResult:
    if not capabilities.has_coef:
        return CoefficientImportanceResult(
            computed=False, skipped_reason="model does not expose coef_"
        )

    try:
        coef = to_jsonable(model.coef_)
    except Exception as exc:  # noqa: BLE001
        logger.error("Reading coef_ failed: %s", exc)
        return CoefficientImportanceResult(
            computed=False, skipped_reason=f"coef_ access raised: {exc}"
        )

    # Multiclass linear models expose coef_ as [n_classes, n_features];
    # collapse to a single per-feature value by averaging magnitude.
    if coef and isinstance(coef[0], list):
        n_features = len(coef[0])
        flat = [safe_mean(abs(coef[c][j]) for c in range(len(coef))) for j in range(n_features)]
        raw_signed = [safe_mean(coef[c][j] for c in range(len(coef))) for j in range(n_features)]
    else:
        flat = [abs(v) for v in coef]
        raw_signed = list(coef)

    normalized = normalize_to_unit_range(flat)

    logger.info("Collected and normalized coefficient importance.")
    return CoefficientImportanceResult(
        raw_coefficients={feature_names[i]: raw_signed[i] for i in range(len(feature_names))},
        normalized_importance={feature_names[i]: normalized[i] for i in range(len(feature_names))},
        computed=True,
    )


# --------------------------------------------------------------------------- #
# Partial dependence
# --------------------------------------------------------------------------- #


def compute_partial_dependence(
    model: Any,
    X_train: Any,
    feature_names: Sequence[str],
    top_features: Sequence[str],
    config: ExplainabilityConfig,
    logger: logging.Logger,
) -> PartialDependenceCollection:
    try:
        from sklearn.inspection import partial_dependence
    except ImportError:
        logger.warning("scikit-learn not available — skipping partial dependence.")
        return PartialDependenceCollection(
            computed=False, skipped_reason="scikit-learn is not installed"
        )

    results: List[PartialDependenceResult] = []
    failures: List[str] = []

    for feature_name in top_features[: config.top_n_features_for_pdp]:
        try:
            feature_index = list(feature_names).index(feature_name)
            pd_result = partial_dependence(model, X_train, [feature_index], kind="average")
            grid = to_jsonable(pd_result["grid_values"][0])
            values = to_jsonable(pd_result["average"][0])
            results.append(
                PartialDependenceResult(
                    feature_name=feature_name, grid_values=grid, pd_values=values
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Partial dependence failed for '%s': %s", feature_name, exc)
            failures.append(feature_name)

    if not results:
        return PartialDependenceCollection(
            computed=False,
            skipped_reason=f"partial dependence failed for all requested features: {failures}",
        )

    logger.info("Computed partial dependence for %d feature(s).", len(results))
    return PartialDependenceCollection(results=results, computed=True)


# --------------------------------------------------------------------------- #
# Unified ranking
# --------------------------------------------------------------------------- #


def build_unified_ranking(
    feature_names: Sequence[str],
    shap_result: Optional[SHAPResult],
    permutation_result: Optional[PermutationImportanceResult],
    native_result: Optional[NativeImportanceResult],
    coefficient_result: Optional[CoefficientImportanceResult],
) -> List[UnifiedFeatureRanking]:
    """Combine every available importance source into one ranking. Each
    source is min-max normalized to [0, 1] independently before averaging,
    so no single method dominates due to differing scales. Sources that
    were not computed are simply omitted from a feature's average (not
    treated as zero).
    """
    shap_norm = _normalized_or_empty(
        shap_result.mean_abs_shap_importance if shap_result and shap_result.computed else {},
        feature_names,
    )
    perm_norm = _normalized_or_empty(
        permutation_result.mean_importance
        if permutation_result and permutation_result.computed
        else {},
        feature_names,
    )
    native_norm = _normalized_or_empty(
        native_result.importance if native_result and native_result.computed else {},
        feature_names,
    )
    coef_norm = _normalized_or_empty(
        coefficient_result.normalized_importance
        if coefficient_result and coefficient_result.computed
        else {},
        feature_names,
    )

    rows: List[UnifiedFeatureRanking] = []
    for name in feature_names:
        available_scores = [
            d[name] for d in (shap_norm, perm_norm, native_norm, coef_norm) if name in d
        ]
        overall_score = safe_mean(available_scores) if available_scores else 0.0
        rows.append(
            UnifiedFeatureRanking(
                feature_name=name,
                shap_score=shap_norm.get(name),
                permutation_importance=perm_norm.get(name),
                native_importance=native_norm.get(name),
                coefficient_importance=coef_norm.get(name),
                overall_score=overall_score,
                overall_rank=0,  # assigned below
            )
        )

    rows.sort(key=lambda r: r.overall_score, reverse=True)
    for i, row in enumerate(rows, start=1):
        row.overall_rank = i

    return rows


def _normalized_or_empty(raw: dict, feature_names: Sequence[str]) -> dict:
    if not raw:
        return {}
    ordered = [raw.get(name, 0.0) for name in feature_names]
    normalized = normalize_to_unit_range(ordered)
    return {name: normalized[i] for i, name in enumerate(feature_names)}


# --------------------------------------------------------------------------- #
# Global explanation (numeric — no LLM)
# --------------------------------------------------------------------------- #


def compute_global_explanation(
    ranking: List[UnifiedFeatureRanking],
    shap_result: Optional[SHAPResult],
    coefficient_result: Optional[CoefficientImportanceResult],
    top_n: int = 5,
) -> GlobalExplanation:
    if not ranking:
        return GlobalExplanation(summary="No feature importance could be computed.")

    most_important = [r.feature_name for r in ranking[:top_n]]
    least_important = [r.feature_name for r in ranking[-top_n:]]

    positive, negative = _infer_directionality(ranking, shap_result, coefficient_result)

    summary = (
        f"The model relies most heavily on {', '.join(most_important)}. "
        f"{', '.join(least_important)} contribute the least to predictions."
    )

    return GlobalExplanation(
        most_important_features=most_important,
        least_important_features=least_important,
        positively_influential_features=positive,
        negatively_influential_features=negative,
        summary=summary,
    )


def _infer_directionality(
    ranking: List[UnifiedFeatureRanking],
    shap_result: Optional[SHAPResult],
    coefficient_result: Optional[CoefficientImportanceResult],
):
    """Determine directional influence (raises vs lowers the prediction),
    preferring signed coefficients when available, falling back to signed
    mean SHAP value, and omitting directionality entirely if neither
    signed source is available (native/permutation importance is
    magnitude-only and cannot indicate direction)."""
    positive: List[str] = []
    negative: List[str] = []

    if coefficient_result and coefficient_result.computed:
        for name, value in coefficient_result.raw_coefficients.items():
            (positive if value > 0 else negative).append(name)
        return _top_n_by_rank(positive, ranking), _top_n_by_rank(negative, ranking)

    if shap_result and shap_result.computed and shap_result.global_shap_values:
        n_features = len(shap_result.feature_names)
        for j in range(n_features):
            column_mean = safe_mean(row[j] for row in shap_result.global_shap_values)
            name = shap_result.feature_names[j]
            (positive if column_mean > 0 else negative).append(name)
        return _top_n_by_rank(positive, ranking), _top_n_by_rank(negative, ranking)

    return [], []


def _top_n_by_rank(names: List[str], ranking: List[UnifiedFeatureRanking], n: int = 5) -> List[str]:
    rank_lookup = {r.feature_name: r.overall_rank for r in ranking}
    return sorted(names, key=lambda n_: rank_lookup.get(n_, 10**9))[:n]


# --------------------------------------------------------------------------- #
# Local explanation
# --------------------------------------------------------------------------- #


def explain_local(
    sample_index: int,
    y_pred_value: float,
    predicted_label: Optional[str],
    feature_names: Sequence[str],
    shap_result: Optional[SHAPResult],
    ranking: List[UnifiedFeatureRanking],
    config: ExplainabilityConfig,
    logger: logging.Logger,
) -> LocalExplanationResult:
    """Explain a single prediction. Prefers per-instance SHAP contributions
    (exact, signed); falls back to the global ranking (magnitude-only,
    clearly labeled as such) if SHAP is unavailable for this sample.
    """
    if (
        shap_result
        and shap_result.computed
        and sample_index in shap_result.sample_indices
    ):
        row = shap_result.global_shap_values[shap_result.sample_indices.index(sample_index)]
        contributions = {feature_names[j]: row[j] for j in range(len(feature_names))}
        top_features = sorted(
            contributions, key=lambda name: abs(contributions[name]), reverse=True
        )[: config.top_n_features_for_local_explanation]

        direction_phrases = [
            f"{name} ({'+' if contributions[name] >= 0 else ''}{contributions[name]:.4f})"
            for name in top_features
        ]
        explanation = (
            f"For sample {sample_index}, the prediction was most influenced by: "
            f"{', '.join(direction_phrases)}."
        )
        logger.info("Generated local SHAP-based explanation for sample %d.", sample_index)
        return LocalExplanationResult(
            sample_index=sample_index,
            predicted_value=y_pred_value,
            predicted_label=predicted_label,
            top_contributing_features=top_features,
            contribution_values=contributions,
            prediction_explanation=explanation,
        )

    # Fallback: no per-instance SHAP row available for this sample.
    logger.warning(
        "No SHAP row available for sample %d — falling back to global ranking "
        "(magnitude-only, not instance-specific).",
        sample_index,
    )
    top_features = [r.feature_name for r in ranking[: config.top_n_features_for_local_explanation]]
    explanation = (
        f"Per-instance SHAP values were unavailable for sample {sample_index}; "
        f"showing the globally most important features instead: {', '.join(top_features)}."
    )
    return LocalExplanationResult(
        sample_index=sample_index,
        predicted_value=y_pred_value,
        predicted_label=predicted_label,
        top_contributing_features=top_features,
        contribution_values={},
        prediction_explanation=explanation,
    )


# --------------------------------------------------------------------------- #
# Visualization data assembly (data only, no rendering)
# --------------------------------------------------------------------------- #


def build_visualization_data(
    feature_names: Sequence[str],
    shap_result: Optional[SHAPResult],
    permutation_result: Optional[PermutationImportanceResult],
    ranking: List[UnifiedFeatureRanking],
    partial_dependence: Optional[PartialDependenceCollection],
    sample_local_explanation: Optional[LocalExplanationResult],
) -> VisualizationData:
    viz = VisualizationData()

    if shap_result and shap_result.computed:
        viz.shap_summary_plot = {
            "feature_names": shap_result.feature_names,
            "mean_abs_importance": shap_result.mean_abs_shap_importance,
        }
        viz.shap_beeswarm_plot = {
            "feature_names": shap_result.feature_names,
            "shap_values": shap_result.global_shap_values,
        }
        if sample_local_explanation:
            viz.shap_waterfall_plot = {
                "sample_index": sample_local_explanation.sample_index,
                "base_value": shap_result.expected_value,
                "contributions": sample_local_explanation.contribution_values,
            }
            viz.shap_force_plot = {
                "sample_index": sample_local_explanation.sample_index,
                "base_value": shap_result.expected_value,
                "contributions": sample_local_explanation.contribution_values,
            }

    viz.feature_importance_bar_chart = {
        "feature_names": [r.feature_name for r in ranking],
        "scores": [r.overall_score for r in ranking],
    }

    if permutation_result and permutation_result.computed:
        viz.permutation_importance_plot = {
            "feature_names": list(permutation_result.mean_importance.keys()),
            "mean_importance": list(permutation_result.mean_importance.values()),
            "std_importance": list(permutation_result.std_importance.values()),
        }

    if partial_dependence and partial_dependence.computed:
        viz.partial_dependence_plot = {
            "curves": [
                {
                    "feature_name": r.feature_name,
                    "grid_values": r.grid_values,
                    "pd_values": r.pd_values,
                }
                for r in partial_dependence.results
            ]
        }

    return viz