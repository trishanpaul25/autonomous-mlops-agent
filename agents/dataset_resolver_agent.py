import os
import re
from agents.base_agent import BaseAgent
from prompts.dataset_resolver_prompt import dataset_resolver_prompt
from services.llm_service import LLMService
from schemas.dataset_resolver_schema import DatasetResolverOutput
from tools.ingestion.ingestion_tool import IngestionTool
from state.pipeline_state import PipelineState
class DatasetResolverAgent(BaseAgent):

    def __init__(self):

        self.chain = None

        if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
            self.llm = LLMService.get_structured_llm(
                DatasetResolverOutput
            )

            self.chain = dataset_resolver_prompt | self.llm

        self.ingestion_tool = IngestionTool()

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
                reasoning=f"Detected a {extension.upper()} path in the user request.",
                confidence=0.9,
                needs_clarification=False,
                clarification_question=None,
            )

        if any(name in lowered_prompt for name in ["iris", "wine", "breast cancer"]):
            dataset_name = "iris" if "iris" in lowered_prompt else (
                "wine" if "wine" in lowered_prompt else "breast cancer"
            )

            return DatasetResolverOutput(
                source_type="builtin",
                source=dataset_name,
                dataset_name=dataset_name,
                reasoning="Detected a builtin scikit-learn dataset request.",
                confidence=0.8,
                needs_clarification=False,
                clarification_question=None,
            )

        return DatasetResolverOutput(
            source_type="csv",
            source="",
            dataset_name=None,
            reasoning="Could not determine the dataset source without model access.",
            confidence=0.1,
            needs_clarification=True,
            clarification_question="Please provide a dataset file path, URL, or builtin dataset name.",
        )

    def run(self, state: PipelineState) -> PipelineState:

        try:

            state.current_agent = "DatasetResolverAgent"

            if self.chain is None:
                resolver_output = self._resolve_without_llm(state.user_prompt)
            else:
                resolver_output = self.chain.invoke(
                    {
                        "user_prompt": state.user_prompt
                    }
                )

            if resolver_output.needs_clarification:

                state.status = "waiting_for_user"

                state.logs.append(
                    resolver_output.clarification_question
                )

                return state

            state = self.ingestion_tool.execute(
                state,
                resolver_output
            )

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

            return state