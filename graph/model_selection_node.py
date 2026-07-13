"""
LangGraph node for the Model Selection Agent.

Follows the same thin-wrapper pattern as all other agent nodes:
a single module-level agent instance is created at import time and
the node function delegates directly to agent.run().
"""

from agents.model_selection_agent import ModelSelectionAgent

from state.pipeline_state import PipelineState


# Single instance reused across all graph invocations
agent = ModelSelectionAgent()


def model_selection_node(
    state: PipelineState,
) -> PipelineState:
    """
    Executes the Model Selection Agent within the LangGraph workflow.

    Parameters
    ----------
    state : PipelineState
        Shared pipeline state passed in by LangGraph.

    Returns
    -------
    PipelineState
        Updated state after model selection completes.
    """
    return agent.run(state)
