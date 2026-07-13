"""
LangGraph node for the Model Training Agent.

Follows the same thin-wrapper pattern as all other agent nodes:
a single module-level agent instance is created at import time and
the node function delegates directly to agent.run().
"""

from agents.model_training_agent import ModelTrainingAgent

from state.pipeline_state import PipelineState


# Single instance reused across all graph invocations
agent = ModelTrainingAgent()


def model_training_node(
    state: PipelineState,
) -> PipelineState:
    """
    Executes the Model Training Agent within the LangGraph workflow.

    Parameters
    ----------
    state : PipelineState
        Shared pipeline state passed in by LangGraph.

    Returns
    -------
    PipelineState
        Updated state after all candidate models have been trained.
    """
    return agent.run(state)
