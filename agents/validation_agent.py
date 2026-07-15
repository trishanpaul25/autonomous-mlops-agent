"""
Validation Agent — decides which dataset validation checks to run via LLM or fallback.
"""

import os

from agents.base_agent import BaseAgent
from prompts.validation_prompt import validation_prompt
from schemas.validation_schema import ValidationOutput
from services.llm_service import LLMService
from state.pipeline_state import PipelineState
from tools.validation.validation_tool import ValidationTool
from utils.logger import logger

from server.core.constants import PipelineStatus

class ValidationAgent(BaseAgent):

    def __init__(self):
        self.chain = None
        if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
            self.llm = LLMService.get_structured_llm(ValidationOutput)
            self.chain = validation_prompt | self.llm
        self.validation_tool = ValidationTool()

    def _validate_without_llm(self) -> ValidationOutput:
        """Fallback validation plan when no LLM is available."""
        return ValidationOutput(
            check_missing_values=True,
            check_duplicates=True,
            check_data_types=True,
            detect_target_column=True,
            infer_problem_type=True,
            problem_type="unknown",
            reasoning="Using default validation checks because no LLM is available.",
            confidence=0.5,
            needs_clarification=False,
            clarification_question=None,
        )

    def run(self, state: PipelineState) -> PipelineState:
        try:
            state.current_agent = "ValidationAgent"

            if self.chain is None:
                logger.info("No LLM available. Using default validation plan.")
                validation_output = self._validate_without_llm()
            else:
                logger.info("Invoking LLM to determine validation checks...")
                validation_output = self.chain.invoke({
                    "user_prompt": state.user_prompt,
                    "metadata": state.dataset.metadata,
                })

            if validation_output.needs_clarification:

                state.status = PipelineStatus.WAITING_FOR_USER

                logger.warning(
                    "Validation requires clarification: %s",
                    validation_output.clarification_question,
                )
                state.logs.append(validation_output.clarification_question)
                return state

            state = self.validation_tool.execute(state, validation_output)
            state.completed_steps.append("Validation")
            logger.info(
                "Validation completed. Problem type: %s, Target column: %s",
                state.validation.problem_type,
                state.validation.target_column,
            )
            state.logs.append("Dataset validation completed successfully.")
            return state

        except Exception as e:

            state.status = PipelineStatus.FAILED

            state.error = str(e)
            logger.error("Validation failed: %s", e, exc_info=True)
            state.logs.append(f"Validation failed: {e}")
            return state