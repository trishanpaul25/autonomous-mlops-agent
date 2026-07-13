"""
LangGraph node for the Feature Engineering Agent.
"""

from agents.feature_engineering_agent import FeatureEngineeringAgent

from state.pipeline_state import PipelineState


# Create a single instance of the Feature Engineering Agent
agent = FeatureEngineeringAgent()


def feature_engineering_node(
    state: PipelineState,
) -> PipelineState:
    """
    Executes the Feature Engineering Agent.
    """

    return agent.run(state)
