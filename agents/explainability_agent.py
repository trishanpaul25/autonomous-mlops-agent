"""
explainability_agent.py

Orchestrator for the Explainability Agent (SRP root of this module).

Responsibilities:
    - Resolve the fitted "best model" and its training/test data from the
      composed PipelineState (see `_resolve_inputs` for the exact
      resolution order, which mirrors the fallback rule documented in
      HyperparameterOptimizationState / ModelEvaluationState).
    - Call explainability_tool.py functions in the right order, applying
      the fallback policy described in the spec (SHAP unavailable ->
      permutation importance -> native importance).
    - Call explainability_prompt.py + an injected LLM client to narrate
      the already-computed results.
    - Package everything into an ExplainabilityResults object, project it
      into ExplainabilityState, and write it to `state.explainability`.

This agent does NOT train, tune, evaluate, deploy, or register models —
it only reads a fitted model and already-split data and produces
explanations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Protocol, Tuple

import numpy as np

from prompts.explainability_prompt import (
    build_business_explanation_prompt,
    build_non_technical_explanation_prompt,
    build_technical_explanation_prompt,
)
from schemas.explainability_schema import (
    AgentStatus,
    ExplainabilityConfig,
    ExplainabilityResults,
    LLMExplanationOutput,
    LocalExplanationResult,
    TaskType,
)
from state.explainability_state import ExplainabilityState
from tools.explainability.explainability_tool import (
    build_unified_ranking,
    build_visualization_data,
    compute_coefficient_importance,
    compute_global_explanation,
    compute_native_importance,
    compute_partial_dependence,
    compute_permutation_importance,
    compute_shap,
    detect_model_capabilities,
    explain_local,
)
from utils.explainability_utils import get_agent_logger


class LLMClient(Protocol):
    """Minimal interface the Explainability Agent needs from an LLM
    client. Any LangChain chat model can be adapted to this with a
    one-line wrapper (see LangChainLLMAdapter below), keeping this agent
    decoupled from any specific LLM SDK (Dependency Inversion).
    """

    def generate(self, prompt: str) -> str:
        ...


class LangChainLLMAdapter:
    """Adapts a LangChain-compatible chat model (anything exposing
    `.invoke(str) -> BaseMessage` or `.predict(str) -> str`) to the
    LLMClient protocol expected by this agent.
    """

    def __init__(self, langchain_model: Any):
        self._model = langchain_model

    def generate(self, prompt: str) -> str:
        if hasattr(self._model, "invoke"):
            response = self._model.invoke(prompt)
            return getattr(response, "content", str(response))
        if hasattr(self._model, "predict"):
            return self._model.predict(prompt)
        raise TypeError(
            "langchain_model must expose either .invoke() or .predict()"
        )


# --------------------------------------------------------------------------- #
# Input resolution
# --------------------------------------------------------------------------- #


@dataclass
class ResolvedInputs:
    """Everything the agent needs, pulled out of the composed
    PipelineState and normalized (lists -> numpy arrays)."""

    best_model: Any
    best_model_identifier: str
    feature_names: List[str]
    target_column: Optional[str]
    task_type: TaskType
    X_train: Any
    X_test: Any
    y_train: Any
    y_test: Any
    resolution_notes: List[str] = field(default_factory=list)


# Model Selection's TaskType (tools/model_selection/model_registry.py) uses
# finer-grained values (binary_classification, multiclass_classification,
# clustering, time_series, ...) than explainability_schema.TaskType, which
# only distinguishes classification vs. regression (explainability doesn't
# need finer granularity — permutation scoring, SHAP routing, etc. only
# care about the classification/regression split). This maps the former
# onto the latter instead of constructing TaskType(task_type_raw) directly,
# which would raise ValueError for every classification run in practice.
_CLASSIFICATION_TASK_TYPE_VALUES = {
    "classification",
    "binary_classification",
    "multiclass_classification",
}
_REGRESSION_TASK_TYPE_VALUES = {"regression"}

# Valid model_selection task types that explainability does not support by
# design (SHAP / permutation-on-a-test-set explanations don't apply the same
# way to clustering or time series). These should produce a SKIPPED status,
# not FAILED — mirroring HyperparameterOptimizationAgent's handling of
# unsupported task types — since nothing is actually broken here.
_UNSUPPORTED_TASK_TYPES = {"clustering", "time_series"}


def _normalize_task_type(task_type_raw: str) -> Optional[TaskType]:
    normalized = task_type_raw.strip().lower()
    if normalized in _CLASSIFICATION_TASK_TYPE_VALUES:
        return TaskType.CLASSIFICATION
    if normalized in _REGRESSION_TASK_TYPE_VALUES:
        return TaskType.REGRESSION
    return None


def _resolve_inputs(state: Any) -> Tuple[Optional[ResolvedInputs], List[str]]:
    """Resolve model + data inputs from the composed PipelineState.

    Returns (resolved, errors). `resolved` is None if resolution failed,
    in which case `errors` explains exactly what was missing.
    """
    errors: List[str] = []
    notes: List[str] = []

    model_training = getattr(state, "model_training", None)
    model_evaluation = getattr(state, "model_evaluation", None)
    hyperparameter_optimization = getattr(state, "hyperparameter_optimization", None)
    feature_engineering = getattr(state, "feature_engineering", None)

    if model_training is None:
        errors.append("PipelineState.model_training is missing")
        return None, errors
    if model_evaluation is None:
        errors.append("PipelineState.model_evaluation is missing")
        return None, errors

    best_model_identifier = getattr(model_evaluation, "best_model_identifier", None)
    if not best_model_identifier:
        errors.append("model_evaluation.best_model_identifier is not set")
        return None, errors

    best_model = None
    if hyperparameter_optimization is not None:
        best_model = hyperparameter_optimization.optimized_model_objects.get(
            best_model_identifier
        )
        if best_model is not None:
            notes.append(
                f"Resolved best model '{best_model_identifier}' from "
                "hyperparameter_optimization.optimized_model_objects"
            )
    if best_model is None:
        best_model = model_training.trained_model_objects.get(best_model_identifier)
        if best_model is not None:
            notes.append(
                f"Resolved best model '{best_model_identifier}' from "
                "model_training.trained_model_objects (HPO fallback, per "
                "HyperparameterOptimizationState's documented fallback rule)"
            )

    if best_model is None:
        errors.append(
            f"Could not find a fitted estimator for best_model_identifier="
            f"'{best_model_identifier}' in either "
            "hyperparameter_optimization.optimized_model_objects or "
            "model_training.trained_model_objects"
        )
        return None, errors

    feature_names = list(model_training.feature_columns) if model_training.feature_columns else []
    if not feature_names and feature_engineering is not None:
        feature_names = list(feature_engineering.final_feature_columns or [])
        if feature_names:
            notes.append(
                "feature_names resolved from feature_engineering.final_feature_columns "
                "(model_training.feature_columns was empty)"
            )
    if not feature_names:
        errors.append(
            "No feature names found on model_training.feature_columns or "
            "feature_engineering.final_feature_columns"
        )

    task_type: Optional[TaskType] = None
    task_type_raw = getattr(model_evaluation, "task_type", None)
    if not task_type_raw:
        errors.append("model_evaluation.task_type is not set")
    else:
        task_type = _normalize_task_type(str(task_type_raw))
        if task_type is None:
            errors.append(
                f"Unrecognized task_type '{task_type_raw}' on model_evaluation "
                f"(expected a classification or regression variant, e.g. "
                "'binary_classification', 'multiclass_classification', "
                "'classification', 'regression')"
            )

    for split_name in ("X_train", "X_test", "y_train", "y_test"):
        if getattr(model_training, split_name, None) is None:
            errors.append(f"model_training.{split_name} is missing")

    if errors:
        return None, errors

    return (
        ResolvedInputs(
            best_model=best_model,
            best_model_identifier=best_model_identifier,
            feature_names=feature_names,
            target_column=model_training.target_column,
            task_type=task_type,
            X_train=np.array(model_training.X_train),
            X_test=np.array(model_training.X_test),
            y_train=np.array(model_training.y_train),
            y_test=np.array(model_training.y_test),
            resolution_notes=notes,
        ),
        [],
    )


# --------------------------------------------------------------------------- #
# Status mapping (ExplainabilityResults.AgentStatus -> state string convention)
# --------------------------------------------------------------------------- #

_STATUS_TO_STATE_STRING = {
    AgentStatus.SUCCESS: "completed",
    AgentStatus.PARTIAL_SUCCESS: "partial",
    AgentStatus.FAILED: "failed",
    AgentStatus.SKIPPED: "skipped",
}


def _collect_skipped_methods(results: ExplainabilityResults) -> dict:
    """method_name -> reason, for every technique that did not compute."""
    skipped: dict = {}
    if results.shap_result and not results.shap_result.computed:
        skipped["shap"] = results.shap_result.skipped_reason or "unknown"
    if results.permutation_result and not results.permutation_result.computed:
        skipped["permutation"] = results.permutation_result.skipped_reason or "unknown"
    if results.native_importance_result and not results.native_importance_result.computed:
        skipped["native"] = results.native_importance_result.skipped_reason or "unknown"
    if results.coefficient_result and not results.coefficient_result.computed:
        skipped["coefficient"] = results.coefficient_result.skipped_reason or "unknown"
    if results.partial_dependence and not results.partial_dependence.computed:
        skipped["partial_dependence"] = results.partial_dependence.skipped_reason or "unknown"
    return skipped


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #


class ExplainabilityAgent:
    """Single-responsibility agent: explain the already-trained,
    already-evaluated best model. Stateless across runs — all state is
    read from / written to the PipelineState instance passed to `run`.
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        config: Optional[ExplainabilityConfig] = None,
    ):
        self._llm_client = llm_client
        self._config = config or ExplainabilityConfig()
        self._logger = get_agent_logger()

    def run(self, state: Any) -> Any:
        """Execute the full explainability pipeline against `state` and
        return the (mutated) PipelineState with `state.explainability`
        populated. Never raises — on unrecoverable failure, writes a
        FAILED ExplainabilityState instead, so a single agent failure
        does not crash the overall LangGraph run.
        """
        start_time = time.monotonic()
        logs: List[str] = []
        warnings: List[str] = []
        errors: List[str] = []

        def log(message: str) -> None:
            self._logger.info(message)
            logs.append(message)

        log("Explainability Started")

        resolved, resolution_errors = _resolve_inputs(state)
        if resolved is None:
            for err in resolution_errors:
                self._logger.error(err)
            errors.extend(resolution_errors)

            raw_task_type = str(
                getattr(getattr(state, "model_evaluation", None), "task_type", "") or ""
            ).strip().lower()
            is_unsupported_task_type = raw_task_type in _UNSUPPORTED_TASK_TYPES

            if is_unsupported_task_type:
                log(
                    f"Explainability skipped — task type '{raw_task_type}' is "
                    "not supported (SHAP/permutation explanations require a "
                    "classification or regression model)."
                )
                results = ExplainabilityResults(
                    agent_status=AgentStatus.SKIPPED,
                    task_type=None,
                    execution_logs=logs,
                    warnings=[
                        f"Skipped: task type '{raw_task_type}' is not "
                        "supported by the Explainability Agent."
                    ],
                    execution_time_seconds=time.monotonic() - start_time,
                )
                return self._write_to_state(state, results)

            results = ExplainabilityResults(
                agent_status=AgentStatus.FAILED,
                task_type=None,
                execution_logs=logs,
                errors=errors,
                execution_time_seconds=time.monotonic() - start_time,
            )
            return self._write_to_state(state, results)

        for note in resolved.resolution_notes:
            log(note)

        best_model = resolved.best_model
        feature_names = resolved.feature_names
        task_type = resolved.task_type
        X_train, X_test = resolved.X_train, resolved.X_test
        y_train, y_test = resolved.y_train, resolved.y_test

        capabilities = detect_model_capabilities(best_model)

        # --- SHAP (with automatic fallback if unavailable/failed) ------- #
        shap_result = None
        if self._config.enable_shap:
            shap_result = compute_shap(
                best_model, X_train, X_test, feature_names, capabilities,
                self._config, self._logger,
            )
            logs.append(f"SHAP computed: {shap_result.computed}")
            if not shap_result.computed:
                warnings.append(f"SHAP skipped: {shap_result.skipped_reason}")
                log("SHAP unavailable — falling back to permutation/native importance")

        # --- Permutation importance --------------------------------------- #
        permutation_result = None
        if self._config.enable_permutation_importance:
            permutation_result = compute_permutation_importance(
                best_model, X_test, y_test, feature_names, task_type,
                self._config, self._logger,
            )
            if not permutation_result.computed:
                warnings.append(f"Permutation importance skipped: {permutation_result.skipped_reason}")

        # --- Native importance --------------------------------------------- #
        native_result = None
        if self._config.enable_native_importance:
            native_result = compute_native_importance(
                best_model, feature_names, capabilities, self._logger
            )
            if not native_result.computed:
                warnings.append(f"Native importance skipped: {native_result.skipped_reason}")

        # --- Coefficient importance ----------------------------------------- #
        coefficient_result = None
        if self._config.enable_coefficient_importance:
            coefficient_result = compute_coefficient_importance(
                best_model, feature_names, capabilities, self._logger
            )
            if not coefficient_result.computed:
                warnings.append(f"Coefficient importance skipped: {coefficient_result.skipped_reason}")

        # If every importance source failed, this run cannot proceed meaningfully.
        all_failed = all(
            r is None or not r.computed
            for r in (shap_result, permutation_result, native_result, coefficient_result)
        )
        if all_failed:
            error_msg = (
                "All importance computation methods failed or were disabled "
                "(SHAP, permutation, native, coefficient). Cannot produce "
                "feature ranking or explanations."
            )
            errors.append(error_msg)
            self._logger.error(error_msg)
            results = ExplainabilityResults(
                agent_status=AgentStatus.FAILED,
                task_type=task_type,
                execution_logs=logs,
                warnings=warnings,
                errors=errors,
                execution_time_seconds=time.monotonic() - start_time,
            )
            return self._write_to_state(state, results)

        # --- Unified ranking -------------------------------------------- #
        ranking = build_unified_ranking(
            feature_names, shap_result, permutation_result, native_result, coefficient_result
        )

        # --- Partial dependence ------------------------------------------ #
        pdp_result = None
        if self._config.enable_partial_dependence:
            top_feature_names = [r.feature_name for r in ranking[: self._config.top_n_features_for_pdp]]
            pdp_result = compute_partial_dependence(
                best_model, X_train, feature_names, top_feature_names, self._config, self._logger
            )
            if not pdp_result.computed:
                warnings.append(f"Partial dependence skipped: {pdp_result.skipped_reason}")

        # --- Global explanation (numeric, no LLM) ------------------------- #
        global_explanation = compute_global_explanation(ranking, shap_result, coefficient_result)
        log("Generated Global Explanation")

        # --- Local explanation for a representative sample ---------------- #
        local_explanations: List[LocalExplanationResult] = []
        sample_local_explanation = None
        try:
            if len(X_test) > 0:
                sample_index = 0
                pred = best_model.predict(X_test[sample_index : sample_index + 1])
                predicted_value = float(pred[0])
                predicted_label = str(pred[0]) if task_type == TaskType.CLASSIFICATION else None
                sample_local_explanation = explain_local(
                    sample_index, predicted_value, predicted_label, feature_names,
                    shap_result, ranking, self._config, self._logger,
                )
                local_explanations.append(sample_local_explanation)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Local explanation for sample 0 failed: {exc}")
            self._logger.warning("Local explanation failed: %s", exc)

        # --- Visualization data ------------------------------------------ #
        visualization_data = build_visualization_data(
            feature_names, shap_result, permutation_result, ranking, pdp_result,
            sample_local_explanation,
        )

        # --- LLM narration -------------------------------------------------- #
        llm_output = self._generate_llm_explanations(ranking, global_explanation, task_type, warnings)
        if llm_output.generated:
            log("Generated Business Explanation")

        log("Explainability Completed")

        status = AgentStatus.SUCCESS if not warnings else AgentStatus.PARTIAL_SUCCESS

        results = ExplainabilityResults(
            agent_status=status,
            task_type=task_type,
            feature_ranking=ranking,
            shap_result=shap_result,
            permutation_result=permutation_result,
            native_importance_result=native_result,
            coefficient_result=coefficient_result,
            partial_dependence=pdp_result,
            global_explanation=global_explanation,
            local_explanations=local_explanations,
            llm_explanation=llm_output,
            visualization_data=visualization_data,
            execution_logs=logs,
            warnings=warnings,
            errors=errors,
            execution_time_seconds=time.monotonic() - start_time,
        )
        return self._write_to_state(state, results)

    def explain_sample(self, state: Any, sample_index: int) -> LocalExplanationResult:
        """Public entry point for on-demand local explanation of an
        arbitrary sample index, independent of the main `run` flow (e.g.
        called later by an API endpoint or the Report Generation Agent).
        Reuses the SHAP result and ranking already stored on
        `state.explainability` if present, avoiding recomputation.
        """
        resolved, errors = _resolve_inputs(state)
        if resolved is None:
            raise ValueError(f"Cannot resolve inputs for local explanation: {errors}")

        existing = getattr(state, "explainability", None)
        shap_values_dict = existing.shap_values if existing and existing.shap_computed else None
        ranking_dicts = existing.feature_ranking if existing else []

        # Rehydrate the minimal pieces explain_local needs from the stored
        # plain-dict state representation.
        from schemas.explainability_schema import SHAPResult, ShapExplainerType, UnifiedFeatureRanking

        shap_result = None
        if shap_values_dict:
            shap_result = SHAPResult(
                explainer_type=ShapExplainerType(shap_values_dict.get("explainer_type", "none")),
                expected_value=shap_values_dict.get("expected_value", []),
                global_shap_values=shap_values_dict.get("global_shap_values", []),
                mean_abs_shap_importance=shap_values_dict.get("mean_abs_shap_importance", {}),
                feature_names=shap_values_dict.get("feature_names", []),
                sample_indices=shap_values_dict.get("sample_indices", []),
                computed=True,
            )
        ranking = [UnifiedFeatureRanking(**r) for r in ranking_dicts]

        row = resolved.X_test[sample_index : sample_index + 1]
        pred = resolved.best_model.predict(row)
        predicted_value = float(pred[0])
        predicted_label = (
            str(pred[0]) if resolved.task_type == TaskType.CLASSIFICATION else None
        )

        return explain_local(
            sample_index, predicted_value, predicted_label, resolved.feature_names,
            shap_result, ranking, self._config, self._logger,
        )

    def _generate_llm_explanations(
        self, ranking, global_explanation, task_type, warnings: List[str]
    ) -> LLMExplanationOutput:
        if not self._config.enable_llm_explanation:
            return LLMExplanationOutput(generated=False, skipped_reason="LLM explanation disabled by config")
        if self._llm_client is None:
            warnings.append("LLM explanation skipped: no llm_client provided to ExplainabilityAgent")
            return LLMExplanationOutput(generated=False, skipped_reason="no llm_client configured")

        try:
            technical = self._llm_client.generate(
                build_technical_explanation_prompt(ranking, global_explanation, task_type)
            )
            business = self._llm_client.generate(
                build_business_explanation_prompt(ranking, global_explanation, task_type)
            )
            non_technical = self._llm_client.generate(
                build_non_technical_explanation_prompt(global_explanation, task_type)
            )
            return LLMExplanationOutput(
                technical_explanation=technical.strip(),
                business_explanation=business.strip(),
                non_technical_explanation=non_technical.strip(),
                generated=True,
            )
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"LLM explanation generation failed: {exc}")
            self._logger.error("LLM explanation generation failed: %s", exc)
            return LLMExplanationOutput(generated=False, skipped_reason=f"LLM call raised: {exc}")

    @staticmethod
    def _write_to_state(state: Any, results: ExplainabilityResults) -> Any:
        """Project ExplainabilityResults into an ExplainabilityState and
        attach it at `state.explainability`, matching the composed
        PipelineState pattern used by every other agent in the pipeline.
        """
        status_string = _STATUS_TO_STATE_STRING[results.agent_status]

        explainability_state = ExplainabilityState(
            status=status_string,
            error=results.errors[0] if results.errors else None,
            is_completed=results.agent_status
            in (AgentStatus.SUCCESS, AgentStatus.PARTIAL_SUCCESS, AgentStatus.SKIPPED),
            explainability_status=status_string,
            task_type=results.task_type.value if results.task_type else None,
            shap_computed=bool(results.shap_result and results.shap_result.computed),
            permutation_computed=bool(results.permutation_result and results.permutation_result.computed),
            native_importance_computed=bool(
                results.native_importance_result and results.native_importance_result.computed
            ),
            coefficient_computed=bool(results.coefficient_result and results.coefficient_result.computed),
            partial_dependence_computed=bool(
                results.partial_dependence and results.partial_dependence.computed
            ),
            shap_explainer_type=(
                results.shap_result.explainer_type.value if results.shap_result else None
            ),
            skipped_methods=_collect_skipped_methods(results),
            feature_ranking=[r.model_dump() for r in results.feature_ranking],
            shap_values=(
                results.shap_result.model_dump()
                if results.shap_result and results.shap_result.computed
                else {}
            ),
            permutation_importance=(
                results.permutation_result.model_dump()
                if results.permutation_result and results.permutation_result.computed
                else {}
            ),
            native_feature_importance=(
                results.native_importance_result.model_dump()
                if results.native_importance_result and results.native_importance_result.computed
                else {}
            ),
            coefficient_importance=(
                results.coefficient_result.model_dump()
                if results.coefficient_result and results.coefficient_result.computed
                else {}
            ),
            partial_dependence=(
                results.partial_dependence.model_dump()
                if results.partial_dependence and results.partial_dependence.computed
                else {}
            ),
            global_explanation=results.global_explanation.model_dump(),
            local_explanations=[le.model_dump() for le in results.local_explanations],
            visualization_data=results.visualization_data.model_dump(),
            technical_explanation=results.llm_explanation.technical_explanation or None,
            business_explanation=results.llm_explanation.business_explanation or None,
            non_technical_explanation=results.llm_explanation.non_technical_explanation or None,
            total_execution_time_seconds=results.execution_time_seconds,
            errors=results.errors,
            warnings=results.warnings,
            summary=results.global_explanation.summary or None,
        )

        state.explainability = explainability_state
        state.logs = list(getattr(state, "logs", [])) + results.execution_logs

        if explainability_state.is_completed:
            completed_steps = list(getattr(state, "completed_steps", []))
            if "explainability" not in completed_steps:
                completed_steps.append("explainability")
            state.completed_steps = completed_steps

        return state