"""
Model Selection Agent.

Responsible for analysing the post-feature-engineering PipelineState,
detecting the ML task type, recommending ranked candidate models, and
storing the selection results in ModelSelectionState.

The agent NEVER trains models, tunes hyperparameters, or evaluates
model performance. Its sole responsibility is algorithm selection.

Architecture
------------
Two execution paths share the same output contract (ModelSelectionOutput):

1. LLM path (primary):
   DatasetProfiler → prompt variables → LLM → ModelSelectionOutput

2. Heuristic path (fallback when no API key is set):
   DatasetProfiler → ModelRegistry.get_candidates() → HeuristicBuilder
   → ModelSelectionOutput

Both paths write their result through ModelSelectionTool, ensuring the
PipelineState update logic lives in exactly one place.
"""

from __future__ import annotations

import os

from agents.base_agent import BaseAgent

from prompts.model_selection_prompt import model_selection_prompt

from schemas.model_selection_schema import (
    CandidateModel,
    ModelSelectionOutput,
)

from services.llm_service import LLMService

from state.pipeline_state import PipelineState

from tools.model_selection.dataset_profiler import DatasetProfiler, DatasetProfile
from tools.model_selection.model_registry import ModelRegistry, TaskType, _compute_score
from tools.model_selection.model_selection_tool import ModelSelectionTool

from utils.logger import logger
from server.core.constants import PipelineStatus

class ModelSelectionAgent(BaseAgent):
    """
    Analyses the dataset and recommends appropriate ML algorithms.

    Constructor wires up:
      - LLM chain (prompt | structured LLM) if an API key is available
      - DatasetProfiler for quantitative dataset analysis
      - ModelRegistry as the algorithm knowledge base
      - ModelSelectionTool for writing results back to PipelineState
    """

    # Maximum candidates to consider and surface to the LLM / fallback
    _MAX_CANDIDATES: int = 7
    _MIN_CANDIDATES: int = 3

    def __init__(self) -> None:

        self.chain = None

        if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):

            self.llm = LLMService.get_structured_llm(ModelSelectionOutput)

            self.chain = model_selection_prompt | self.llm

        self.profiler = DatasetProfiler()
        self.registry = ModelRegistry()
        self.tool = ModelSelectionTool()

    def _select_without_llm(
        self,
        profile: DatasetProfile,
    ) -> ModelSelectionOutput:
        """
        Build a ModelSelectionOutput using only the ModelRegistry and
        dataset profile — no LLM call required.

        The heuristic guarantees at least _MIN_CANDIDATES candidates.
        If the registry returns fewer than _MIN_CANDIDATES for the
        exact task type, broader types are used as a safety net.
        """
        candidates = self.registry.get_candidates(
            task_type=profile.task_type,
            profile=profile,
            max_candidates=self._MAX_CANDIDATES,
        )

        if len(candidates) < self._MIN_CANDIDATES:
            logger.warning(
                "ModelRegistry returned only %d candidates for task '%s'. "
                "Expanding search.",
                len(candidates),
                profile.task_type.value,
            )

        # Build CandidateModel objects from ModelDescriptor + computed score
        candidate_models: list[CandidateModel] = []

        for rank, descriptor in enumerate(candidates, start=1):
            score = self.registry.get_score(descriptor, profile.task_type, profile)

            strengths: list[str] = []
            limitations: list[str] = []

            # Derive strengths from descriptor metadata
            if descriptor.interpretability >= 4:
                strengths.append("High interpretability — decisions are transparent")
            if descriptor.scalability >= 4:
                strengths.append("Scales well to large datasets")
            if descriptor.handles_missing:
                strengths.append("Natively handles missing values")
            if descriptor.handles_categorical:
                strengths.append("Natively handles categorical features")
            if descriptor.handles_imbalance:
                strengths.append("Supports class imbalance weighting")

            # Derive limitations from descriptor metadata
            if descriptor.interpretability <= 2:
                limitations.append("Low interpretability — requires SHAP/LIME for explanations")
            if descriptor.scalability <= 2:
                limitations.append("May be slow on large datasets")
            if not descriptor.handles_missing:
                limitations.append("Missing values must be imputed before use")
            if descriptor.max_rows > 0 and profile.num_rows > descriptor.max_rows * 0.8:
                limitations.append(
                    f"Approaching row limit ({descriptor.max_rows:,} rows recommended max)"
                )

            rationale = (
                f"{descriptor.description} "
                f"Selected for '{profile.task_type.value}' task with "
                f"{profile.num_rows:,} rows and {profile.num_feature_cols} features."
            )

            candidate_models.append(
                CandidateModel(
                    name=descriptor.name,
                    library=descriptor.library,
                    class_path=descriptor.class_path,
                    rank=rank,
                    suitability_score=score,
                    rationale=rationale,
                    strengths=strengths,
                    limitations=limitations,
                )
            )

        primary = candidate_models[0] if candidate_models else self._emergency_fallback(profile)

        ranking_criteria = [
            "Suitability score derived from dataset size, feature types, and class distribution",
            "Scalability to the current dataset dimensions",
            "Native support for categorical features and missing values",
            "Class imbalance handling capability",
            "Interpretability requirement based on dataset size",
        ]

        assumptions: list[str] = profile.warnings + [
            "No LLM available — selection based on heuristic registry scoring only.",
            "Hyperparameter tuning has not been considered.",
            "Model evaluation metrics are not part of this recommendation.",
        ]

        reasoning = (
            f"Heuristic model selection for task '{profile.task_type.value}'. "
            f"Dataset: {profile.num_rows:,} rows × {profile.num_feature_cols} features "
            f"({profile.dataset_size_category.value} size category). "
            f"The registry was queried for models supporting this task type and "
            f"scored against the dataset profile. "
            f"Primary model '{primary.name}' achieved the highest composite score "
            f"({primary.suitability_score:.2f})."
        )

        return ModelSelectionOutput(
            task_type=profile.task_type.value,  # type: ignore[arg-type]
            primary_model=primary,
            candidate_models=candidate_models,
            ranking_criteria=ranking_criteria,
            reasoning=reasoning,
            assumptions=assumptions,
            confidence=0.65,
            needs_clarification=False,
            clarification_question=None,
        )

    def _emergency_fallback(self, profile: DatasetProfile) -> CandidateModel:
        """
        Returns a safe last-resort model when the registry has no candidates.
        This should never happen in production but guards against bad state.
        """
        from schemas.model_selection_schema import CandidateModel

        task_map = {
            TaskType.BINARY_CLASSIFICATION: (
                "Logistic Regression",
                "sklearn",
                "sklearn.linear_model.LogisticRegression",
            ),
            TaskType.MULTICLASS_CLASSIFICATION: (
                "Random Forest Classifier",
                "sklearn",
                "sklearn.ensemble.RandomForestClassifier",
            ),
            TaskType.REGRESSION: (
                "Ridge Regression",
                "sklearn",
                "sklearn.linear_model.Ridge",
            ),
            TaskType.CLUSTERING: (
                "K-Means",
                "sklearn",
                "sklearn.cluster.KMeans",
            ),
            TaskType.TIME_SERIES: (
                "ARIMA",
                "statsmodels",
                "statsmodels.tsa.arima.model.ARIMA",
            ),
        }

        name, library, class_path = task_map.get(
            profile.task_type,
            ("Logistic Regression", "sklearn", "sklearn.linear_model.LogisticRegression"),
        )

        return CandidateModel(
            name=name,
            library=library,
            class_path=class_path,
            rank=1,
            suitability_score=0.5,
            rationale="Emergency fallback — registry returned no candidates.",
            strengths=["Safe default for the detected task type"],
            limitations=["Selected without dataset-specific scoring"],
        )

    def run(self, state: PipelineState) -> PipelineState:
        """
        Execute the Model Selection Agent.

        Execution flow
        --------------
        1. Guard: verify feature engineering completed.
        2. Build DatasetProfile.
        3. Invoke LLM or heuristic to produce ModelSelectionOutput.
        4. Handle clarification request if raised.
        5. Write output to PipelineState via ModelSelectionTool.
        6. Append step and logs.

        Parameters
        ----------
        state : PipelineState
            Shared pipeline state after Feature Engineering.

        Returns
        -------
        PipelineState
            Updated state with model_selection populated.
        """
        try:
            state.current_agent = "ModelSelectionAgent"

            if not state.feature_engineering.is_completed:
                raise ValueError(
                    "Model Selection Agent requires Feature Engineering to have "
                    "completed successfully before it can run."
                )

            logger.info(
                "ModelSelectionAgent: building dataset profile..."
            )

            profile: DatasetProfile = self.profiler.build(state)

            if profile.warnings:
                for warning in profile.warnings:
                    logger.warning("DatasetProfiler: %s", warning)
                state.model_selection.warnings.extend(profile.warnings)

            logger.info(
                "DatasetProfile built. Task: %s | Rows: %d | Features: %d | "
                "Size category: %s",
                profile.task_type.value,
                profile.num_rows,
                profile.num_feature_cols,
                profile.dataset_size_category.value,
            )

            if self.chain is None:

                logger.info(
                    "No LLM available. Using heuristic model selection."
                )

                ms_output: ModelSelectionOutput = self._select_without_llm(profile)

            else:

                logger.info(
                    "Invoking LLM for model selection (task: %s)...",
                    profile.task_type.value,
                )

                ms_output = self.chain.invoke(
                    {
                        "user_prompt": state.user_prompt,
                        "problem_type": profile.task_type.value,
                        "target_column": profile.target_column,
                        "target_dtype": profile.target_dtype,
                        "num_rows": profile.num_rows,
                        "num_feature_cols": profile.num_feature_cols,
                        "numerical_features": profile.numerical_features,
                        "categorical_features": profile.categorical_features,
                        "class_distribution": profile.class_distribution,
                        "has_missing": profile.has_missing_after_engineering,
                        "dataset_size_category": profile.dataset_size_category.value,
                        "transformations_applied": profile.transformations_applied,
                        "encoded_columns": profile.encoded_columns,
                        "scaled_columns": profile.scaled_columns,
                        "dropped_columns": profile.dropped_columns,
                        "fe_summary": profile.fe_summary,
                        "metadata": profile.metadata,
                    }
                )
            if ms_output.needs_clarification:

                state.status = PipelineStatus.WAITING_FOR_USER

                logger.warning(
                    "ModelSelectionAgent requires clarification: %s",
                    ms_output.clarification_question,
                )

                state.logs.append(ms_output.clarification_question)

                return state

            state = self.tool.execute(
                state,
                ms_output,
                dataset_profile_dict=profile.to_dict(),
            )

            state.completed_steps.append("Model Selection")

            logger.info(
                "ModelSelectionAgent completed. "
                "Task: %s | Primary model: %s | Candidates: %d | "
                "Confidence: %.2f",
                ms_output.task_type,
                ms_output.primary_model.name,
                len(ms_output.candidate_models),
                ms_output.confidence,
            )

            state.logs.append(
                f"Model selection completed. "
                f"Task: {ms_output.task_type} | "
                f"Primary model: {ms_output.primary_model.name} | "
                f"Candidates: {len(ms_output.candidate_models)}."
            )

            return state

        except Exception as e:

            state.status = PipelineStatus.FAILED

            state.error = str(e)

            logger.error(
                "ModelSelectionAgent failed: %s", e, exc_info=True
            )

            state.logs.append(f"Model selection failed: {e}")

            return state
