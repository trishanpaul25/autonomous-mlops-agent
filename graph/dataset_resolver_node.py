from agents.dataset_resolver_agent import DatasetResolverAgent
from state.pipeline_state import PipelineState

_agent: DatasetResolverAgent | None = None


def _get_agent() -> DatasetResolverAgent:
    global _agent

    if _agent is None:
        _agent = DatasetResolverAgent()

    return _agent


def dataset_resolver_node(state: PipelineState) -> PipelineState:
    """
    LangGraph node for the Dataset Resolver Agent.
    """

    return _get_agent().run(state)