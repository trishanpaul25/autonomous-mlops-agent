"""
LangGraph node for the Master Orchestrator Agent.
"""

from agents.master_orchestrator_agent import MasterOrchestratorAgent

from state.pipeline_state import PipelineState


# Create a single instance of the orchestrator
agent = MasterOrchestratorAgent()


def master_orchestrator_node(
    state: PipelineState,
) -> PipelineState:
    """
    Executes the Master Orchestrator Agent.
    """

    return agent.run(state)