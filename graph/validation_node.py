"""
LangGraph node for the Validation Agent.
"""

from agents.validation_agent import ValidationAgent

from state.pipeline_state import PipelineState


# Create a single instance of the Validation Agent
agent = ValidationAgent()


def validation_node(
    state: PipelineState,
) -> PipelineState:
    """
    Executes the Validation Agent.
    """

    return agent.run(state)