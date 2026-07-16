"""
Feature Engineering Agent — decides and applies feature engineering steps.
"""

import os

import pandas as pd

from agents.base_agent import BaseAgent
from prompts.feature_engineering_prompt import feature_engineering_prompt
from schemas.feature_engineering_schema import (
    DerivedFeatureSpec,
    FeatureEngineeringOutput,
)
from services.llm_service import LLMService
from state.pipeline_state import PipelineState
from tools.feature_engineering.feature_engineering_tool import FeatureEngineeringTool
from utils.logger import logger

from server.core.constants import PipelineStatus

class FeatureEngineeringAgent(BaseAgent):

    def __init__(self):
        self.chain = None
        if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
            self.llm = LLMService.get_structured_llm(FeatureEngineeringOutput)
            self.chain = feature_engineering_prompt | self.llm
        self.feature_engineering_tool = FeatureEngineeringTool()

    @staticmethod
    def _detect_identifier_columns(state: PipelineState) -> list[str]:
        """
        Heuristically detects identifier-like or free-text columns
        (e.g. IDs, names, ticket numbers) that carry no generalizable signal.
        """
        df = state.dataset.dataframe
        target_column = state.validation.target_column
        if df is None or len(df) == 0:
            return []

        candidates = []
        for col in df.columns:
            if col == target_column:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                if col.lower() in ("id", "index", "unnamed: 0") or (
                    col.lower().endswith("id") and df[col].nunique() == len(df)
                ):
                    candidates.append(col)
            else:
                if df[col].nunique() / len(df) > 0.9:
                    candidates.append(col)
        return candidates

    @staticmethod
    def _detect_derived_features(state: PipelineState) -> list[DerivedFeatureSpec]:
        """
        Heuristically proposes generic derived features by looking at
        column shape and naming patterns, not by hardcoding any one
        dataset's column names. Used when no LLM is available.
        """
        df = state.dataset.dataframe
        target_column = state.validation.target_column
        if df is None or len(df) == 0:
            return []

        specs: list[DerivedFeatureSpec] = []

        # A free-text column where most values match "<text>, <word>." —
        # e.g. names formatted as "Last, Title. First" — likely has an
        # extractable title/prefix worth keeping before the column itself
        # gets dropped as an identifier.
        title_pattern = r",\s*([A-Za-z]+)\."
        title_detect_pattern = r",\s*[A-Za-z]+\."
        for col in df.select_dtypes(include=["object", "string"]).columns:
            if col == target_column:
                continue
            sample = df[col].dropna().astype(str)
            if sample.empty:
                continue
            if sample.str.contains(title_detect_pattern, regex=True).mean() > 0.8:
                specs.append(DerivedFeatureSpec(
                    new_column=f"{col.lower()}_title",
                    operation="regex_extract",
                    source_columns=[col],
                    pattern=title_pattern,
                    fillna="Rare",
                ))
                break

        # Two numeric columns that both look like relative-count columns
        # (siblings/spouse and parents/children) combine into a family size.
        sib_col = next(
            (c for c in df.columns if any(k in c.lower() for k in ("sibsp", "sibling", "spouse"))),
            None,
        )
        parch_col = next(
            (c for c in df.columns if any(k in c.lower() for k in ("parch", "parent", "child"))),
            None,
        )
        if sib_col and parch_col:
            specs.append(DerivedFeatureSpec(
                new_column="family_size",
                operation="sum_columns",
                source_columns=[sib_col, parch_col],
                constant=1,
            ))

        # A mostly-missing, high-cardinality categorical column would
        # otherwise be dropped outright as an identifier; a coarse
        # first-character category can still carry signal.
        for col in df.select_dtypes(include=["object", "string"]).columns:
            if col == target_column:
                continue
            non_null = df[col].dropna()
            if non_null.empty:
                continue
            missing_ratio = 1 - (len(non_null) / len(df))
            if missing_ratio > 0.5 and non_null.nunique() / len(non_null) > 0.3:
                specs.append(DerivedFeatureSpec(
                    new_column=f"{col.lower()}_category",
                    operation="first_char",
                    source_columns=[col],
                    fillna="Unknown",
                ))

        return specs

    def _plan_without_llm(self, state: PipelineState) -> FeatureEngineeringOutput:
        """Fallback feature engineering plan when no LLM is available."""
        has_missing = any(v > 0 for v in (state.validation.missing_values or {}).values())
        return FeatureEngineeringOutput(
            derived_features=self._detect_derived_features(state),
            drop_columns=self._detect_identifier_columns(state),
            handle_missing_values=has_missing,
            numeric_missing_strategy="mean",
            categorical_missing_strategy="mode",
            handle_outliers=False,
            outlier_method="none",
            encode_categorical=True,
            encoding_method="onehot",
            scale_numerical=True,
            scaling_method="standard",
            reasoning="Using default feature engineering steps because no LLM is available.",
            confidence=0.5,
            needs_clarification=False,
            clarification_question=None,
        )

    def run(self, state: PipelineState) -> PipelineState:
        try:
            state.current_agent = "FeatureEngineeringAgent"

            if state.dataset.dataframe is None:
                raise ValueError("No dataset available for feature engineering.")

            if self.chain is None:
                logger.info("No LLM available. Using default feature engineering plan.")
                fe_output = self._plan_without_llm(state)
            else:
                logger.info("Invoking LLM to determine feature engineering steps...")
                fe_output = self.chain.invoke({
                    "user_prompt": state.user_prompt,
                    "metadata": state.dataset.metadata,
                    "target_column": state.validation.target_column,
                    "problem_type": state.validation.problem_type,
                    "missing_values": state.validation.missing_values,
                    "duplicate_rows": state.validation.duplicate_rows,
                    "data_types": state.validation.data_types,
                })

            if fe_output.needs_clarification:

                state.status = PipelineStatus.WAITING_FOR_USER

                logger.warning(
                    "Feature engineering requires clarification: %s",
                    fe_output.clarification_question,
                )
                state.logs.append(fe_output.clarification_question)
                return state

            state = self.feature_engineering_tool.execute(state, fe_output)
            state.completed_steps.append("Feature Engineering")
            logger.info(
                "Feature engineering completed. Final shape: %s",
                state.feature_engineering.final_shape,
            )
            state.logs.append("Feature engineering completed successfully.")
            return state

        except Exception as e:

            state.status = PipelineStatus.FAILED

            state.error = str(e)
            logger.error("Feature engineering failed: %s", e, exc_info=True)
            state.logs.append(f"Feature engineering failed: {e}")
            return state