"""
Dataset Resolver Agent — identifies the dataset source from the user prompt.
Never loads datasets.
"""

import os
import re

from agents.base_agent import BaseAgent
from prompts.dataset_resolver_prompt import dataset_resolver_prompt
from schemas.dataset_resolver_schema import DatasetResolverOutput
from services.llm_service import LLMService
from state.pipeline_state import PipelineState
from utils.logger import logger
from server.core.constants import PipelineStatus

_EXT_TO_SOURCE_TYPE = {"csv": "csv", "xlsx": "excel", "xls": "excel", "json": "json", "zip": "zip"}
_BUILTIN_NAMES = ["iris", "wine", "breast cancer"]


class DatasetResolverAgent(BaseAgent):

    def __init__(self):
        self.chain = None
        if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
            self.llm = LLMService.get_structured_llm(DatasetResolverOutput)
            self.chain = dataset_resolver_prompt | self.llm

    def _resolve_without_llm(self, user_prompt: str) -> DatasetResolverOutput:
        lowered = user_prompt.lower()

        # Try to match a file path / name with a known extension
        file_match = re.search(
            r"([\w./\\-]+\.(csv|xlsx|xls|json|zip))", user_prompt, flags=re.IGNORECASE
        )
        if file_match:
            source = file_match.group(1)
            ext = file_match.group(2).lower()
            name = re.sub(r"\.(csv|xlsx|xls|json|zip)$", "", source.split("/")[-1].split("\\")[-1], flags=re.IGNORECASE)
            return DatasetResolverOutput(
                source_type=_EXT_TO_SOURCE_TYPE[ext],
                source=source,
                dataset_name=name,
                reasoning=f"Detected a {ext.upper()} dataset.",
                confidence=0.90,
                needs_clarification=False,
                clarification_question=None,
            )

        # Try built-in sklearn datasets
        matched = next((n for n in _BUILTIN_NAMES if n in lowered), None)
        if matched:
            return DatasetResolverOutput(
                source_type="builtin",
                source=matched,
                dataset_name=matched,
                reasoning="Detected a built-in scikit-learn dataset.",
                confidence=0.80,
                needs_clarification=False,
                clarification_question=None,
            )

        return DatasetResolverOutput(
            source_type="csv",
            source="",
            dataset_name=None,
            reasoning="Unable to determine the dataset source.",
            confidence=0.10,
            needs_clarification=True,
            clarification_question="Please provide a dataset path, URL, or built-in dataset name.",
        )

    def run(
        self,
        state: PipelineState,
    ) -> PipelineState:
        """
        adding this to handle uploaded datasets
        if finds a dataset which is uploaded resolver
        is skipped
        """
        if state.dataset.dataset_path:

            logger.info(
                "Using uploaded dataset. Skipping dataset resolution."
            )

            state.completed_steps.append(
                "Dataset Resolver"
            )

            state.logs.append(
                "Uploaded dataset detected."
            )

            return state
    
    def run(self, state: PipelineState) -> PipelineState:
        try:
            state.current_agent = "DatasetResolverAgent"
            logger.info("Resolving dataset source...")

            if self.chain is None:
                resolver_output = self._resolve_without_llm(state.user_prompt)
                logger.info("Dataset source resolved without LLM.")
            else:
                try:
                    resolver_output = self.chain.invoke({"user_prompt": state.user_prompt})
                except Exception as llm_err:
                    err_str = str(llm_err)
                    if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
                        logger.warning(
                            "LLM quota exhausted (%s...). Falling back to rule-based resolver.",
                            err_str[:120],
                        )
                        resolver_output = self._resolve_without_llm(state.user_prompt)
                    else:
                        raise

            if resolver_output.needs_clarification:

                state.status = PipelineStatus.WAITING_FOR_USER

                state.logs.append(
                    resolver_output.clarification_question
                )

                state.logs.append(resolver_output.clarification_question)
                return state

            state.dataset.source_type = resolver_output.source_type
            state.dataset.dataset_name = resolver_output.dataset_name

            """
            for uploaded datasets we already have 
            state.dataset.dataset_path
            so checking before overwriting
            """
            if not state.dataset.dataset_path:
                state.dataset.dataset_path = resolver_output.source

            state.completed_steps.append(
                "Dataset Resolver"
            )

            state.logs.append(
                "Dataset resolved successfully."
            )

            state.dataset.dataset_path = resolver_output.source
            state.completed_steps.append("Dataset Resolver")
            state.logs.append("Dataset resolved successfully.")
            return state

        except Exception as e:

            state.status = PipelineStatus.FAILED
            state.error = str(e)
            state.logs.append(f"Dataset Resolver failed: {e}")
            return state