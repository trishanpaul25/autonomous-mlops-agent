"""
Dataset Resolver Agent.

Responsible for identifying the dataset source
and updating the DatasetState.

This agent NEVER loads datasets.
"""

import os
import re
from agents.base_agent import BaseAgent
from prompts.dataset_resolver_prompt import dataset_resolver_prompt
from services.llm_service import LLMService
from schemas.dataset_resolver_schema import DatasetResolverOutput
from state.pipeline_state import PipelineState
from utils.logger import logger

class DatasetResolverAgent(BaseAgent):

    def __init__(self):

        self.chain = None

        if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):

            self.llm = LLMService.get_structured_llm(
                DatasetResolverOutput
            )

            self.chain = dataset_resolver_prompt | self.llm

    def _resolve_without_llm(
        self,
        user_prompt: str,
    ) -> DatasetResolverOutput:

        lowered_prompt = user_prompt.lower()

        file_match = re.search(
            r"([\w./\\-]+\.(csv|xlsx|xls|json|zip))",
            user_prompt,
            flags=re.IGNORECASE,
        )

        if file_match:

            source = file_match.group(1)

            extension = file_match.group(2).lower()

            source_type = {
                "csv": "csv",
                "xlsx": "excel",
                "xls": "excel",
                "json": "json",
                "zip": "zip",
            }[extension]

            return DatasetResolverOutput(

                source_type=source_type,

                source=source,

                dataset_name=re.sub(
                    r"\.(csv|xlsx|xls|json|zip)$",
                    "",
                    source.split("/")[-1].split("\\")[-1],
                    flags=re.IGNORECASE,
                ),

                reasoning=f"Detected a {extension.upper()} dataset.",

                confidence=0.90,

                needs_clarification=False,

                clarification_question=None,
            )

        if any(
            name in lowered_prompt
            for name in [
                "iris",
                "wine",
                "breast cancer",
            ]
        ):

            dataset_name = (
                "iris"
                if "iris" in lowered_prompt
                else (
                    "wine"
                    if "wine" in lowered_prompt
                    else "breast cancer"
                )
            )

            return DatasetResolverOutput(

                source_type="builtin",

                source=dataset_name,

                dataset_name=dataset_name,

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

        try:
            state.current_agent = "DatasetResolverAgent"
            logger.info("Resolving dataset source...")
            if self.chain is None:
                resolver_output = self._resolve_without_llm(
                    state.user_prompt
                )
                logger.info("Dataset source resolved without LLM.")
            else:
                try:
                    resolver_output = self.chain.invoke(
                        {
                            "user_prompt": state.user_prompt
                        }
                    )
                except Exception as llm_err:
                    err_str = str(llm_err)
                    if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
                        logger.warning(
                            f"LLM quota exhausted ({err_str[:120]}...). "
                            "Falling back to rule-based resolver."
                        )
                        resolver_output = self._resolve_without_llm(
                            state.user_prompt
                        )
                    else:
                        raise

            if resolver_output.needs_clarification:

                state.status = "waiting_for_user"

                state.logs.append(
                    resolver_output.clarification_question
                )

                return state
            state.dataset.source_type = resolver_output.source_type

            state.dataset.dataset_name = resolver_output.dataset_name

            state.dataset.dataset_path = resolver_output.source

            state.completed_steps.append(
                "Dataset Resolver"
            )

            state.logs.append(
                "Dataset resolved successfully."
            )

            return state

        except Exception as e:

            state.status = "failed"

            state.error = str(e)

            state.logs.append(
                f"Dataset Resolver failed: {e}"
            )
            return state