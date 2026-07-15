from graph import workflow
from state import PipelineState


class OrchestrationService:
    """
    Service responsible for executing the LangGraph workflow.
    """

    def run(self, state: PipelineState) -> PipelineState:
        result = workflow.invoke(state)

        # Convert dict -> PipelineState
        return PipelineState.model_validate(result)