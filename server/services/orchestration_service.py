import time
from datetime import datetime

from graph import workflow
from state import PipelineState
from server.core.constants import PipelineStatus

class OrchestrationService:
    """
    Service responsible for executing the LangGraph workflow.
    """

    def run(self, state: PipelineState) -> PipelineState:
        #pipeline started
        state.status = PipelineStatus.RUNNING
        start = time.perf_counter()
        try:
            result = workflow.invoke(state)

            # Convert dict -> PipelineState
            result = PipelineState.model_validate(result)
            result.status = PipelineStatus.SUCCESS
        except Exception:
            state.status = PipelineStatus.FAILED
            raise
        finally:
            finish = datetime.now()
        
        result.completed_at = finish
        result.execution_time = round(
            time.perf_counter() - start, 2
        )

        return result