from langgraph.graph import StateGraph, START, END

from state.pipeline_state import PipelineState
from graph.dataset_resolver_node import dataset_resolver_node


builder = StateGraph(PipelineState)

builder.add_node(
    "dataset_resolver",
    dataset_resolver_node
)

builder.add_edge(
    START,
    "dataset_resolver"
)

builder.add_edge(
    "dataset_resolver",
    END
)

workflow = builder.compile()