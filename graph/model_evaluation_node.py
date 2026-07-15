"""
LangGraph node for the Model Evaluation Agent.

Follows the same thin-wrapper pattern as all other agent nodes:
a single module-level agent instance is created at import time and
the node function delegates directly to agent.run().

This node is available for future use if the workflow is refactored
to invoke evaluation as a standalone LangGraph node rather than through
the Master Orchestrator. Currently, the Master Orchestrator calls
ModelEvaluationAgent.run() directly in its execution loop.
"""

from agents.model_evaluation_agent import ModelEvaluationAgent
from state.pipeline_state import PipelineState


# Single instance reused across all graph invocations
agent = ModelEvaluationAgent()


def model_evaluation_node(state: PipelineState) -> PipelineState:
    """
    Executes the Model Evaluation Agent within the LangGraph workflow.

    Parameters
    ----------
    state : PipelineState
        Shared pipeline state passed in by LangGraph.
        Must have model_training.is_completed == True.

    Returns
    -------
    PipelineState
        Updated state after all available models have been evaluated
        on the held-out test set.
    """
    return agent.run(state)
