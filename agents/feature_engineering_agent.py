"""
Feature Engineering Agent.

Responsible for analyzing the dataset metadata and validation
results, deciding which feature engineering steps should be
executed, and updating the FeatureEngineeringState.
"""

import os

import pandas as pd

from agents.base_agent import BaseAgent

from prompts.feature_engineering_prompt import feature_engineering_prompt

from services.llm_service import LLMService

from schemas.feature_engineering_schema import FeatureEngineeringOutput

from tools.feature_engineering.feature_engineering_tool import (
    FeatureEngineeringTool,
)

from state.pipeline_state import PipelineState

from utils.logger import logger


class FeatureEngineeringAgent(BaseAgent):

    def __init__(self):

        self.chain = None

        if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):

            self.llm = LLMService.get_structured_llm(
                FeatureEngineeringOutput
            )

            self.chain = feature_engineering_prompt | self.llm

        self.feature_engineering_tool = FeatureEngineeringTool()

    @staticmethod
    def _detect_identifier_columns(state: PipelineState) -> list[str]:
        """
        Heuristically detects identifier-like or free-text columns
        (e.g. IDs, names, ticket numbers) that should be dropped
        before modeling when no LLM is available to make the call.

        A column is flagged when it is (close to) unique per row,
        i.e. it carries no generalizable signal for the model.
        """

        df = state.dataset.dataframe

        target_column = state.validation.target_column

        if df is None or len(df) == 0:

            return []

        candidates = []

        for column in df.columns:

            if column == target_column:

                continue

            if pd.api.types.is_numeric_dtype(df[column]):

                # Numeric columns that look like row identifiers
                if column.lower() in ("id", "index", "unnamed: 0") or (
                    column.lower().endswith("id")
                    and df[column].nunique() == len(df)
                ):

                    candidates.append(column)

                continue

            # Non-numeric (object / string / category) columns
            uniqueness_ratio = df[column].nunique() / len(df)

            if uniqueness_ratio > 0.9:

                candidates.append(column)

        return candidates

    def _plan_without_llm(
        self,
        state: PipelineState,
    ) -> FeatureEngineeringOutput:
        """
        Fallback feature engineering plan when no LLM is available.
        """

        missing_values = state.validation.missing_values or {}

        has_missing = any(
            count > 0 for count in missing_values.values()
        )

        drop_columns = self._detect_identifier_columns(state)

        return FeatureEngineeringOutput(

            drop_columns=drop_columns,

            handle_missing_values=has_missing,

            numeric_missing_strategy="mean",

            categorical_missing_strategy="mode",

            handle_outliers=False,

            outlier_method="none",

            encode_categorical=True,

            encoding_method="onehot",

            scale_numerical=True,

            scaling_method="standard",

            reasoning=(
                "Using default feature engineering steps because "
                "no LLM is available."
            ),

            confidence=0.5,

            needs_clarification=False,

            clarification_question=None,
        )

    def run(
        self,
        state: PipelineState,
    ) -> PipelineState:

        try:

            state.current_agent = "FeatureEngineeringAgent"

            if state.dataset.dataframe is None:

                raise ValueError(
                    "No dataset available for feature engineering."
                )

            if self.chain is None:

                logger.info(
                    "No LLM available. Using default feature engineering plan."
                )

                fe_output = self._plan_without_llm(state)

            else:

                logger.info(
                    "Invoking LLM to determine feature engineering steps..."
                )

                fe_output = self.chain.invoke(

                    {
                        "user_prompt": state.user_prompt,

                        "metadata": state.dataset.metadata,

                        "target_column": state.validation.target_column,

                        "problem_type": state.validation.problem_type,

                        "missing_values": state.validation.missing_values,

                        "duplicate_rows": state.validation.duplicate_rows,

                        "data_types": state.validation.data_types,
                    }

                )

            if fe_output.needs_clarification:

                state.status = "waiting_for_user"

                logger.warning(
                    "Feature engineering requires clarification: %s",
                    fe_output.clarification_question,
                )

                state.logs.append(
                    fe_output.clarification_question
                )

                return state

            state = self.feature_engineering_tool.execute(

                state,

                fe_output,

            )

            state.completed_steps.append(
                "Feature Engineering"
            )

            logger.info(
                "Feature engineering completed. Final shape: %s",
                state.feature_engineering.final_shape,
            )

            state.logs.append(
                "Feature engineering completed successfully."
            )

            return state

        except Exception as e:

            state.status = "failed"

            state.error = str(e)

            logger.error("Feature engineering failed: %s", e, exc_info=True)

            state.logs.append(
                f"Feature engineering failed: {e}"
            )

            return state
