"""
LangGraph node for the Data Ingestion Agent.
"""

from agents.data_ingestion_agent import DataIngestionAgent

from state.pipeline_state import PipelineState


agent = DataIngestionAgent()


def data_ingestion_node(
    state: PipelineState,
) -> PipelineState:
    """
    Executes the Data Ingestion Agent.
    """

    return agent.run(state)