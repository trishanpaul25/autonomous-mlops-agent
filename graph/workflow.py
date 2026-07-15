"""
LangGraph workflow.

Routes execution through the Master Orchestrator, which coordinates
all sub-agents in the MLOps pipeline.
"""

from langgraph.graph import StateGraph, START, END
from state.pipeline_state import PipelineState
from graph.nodes import master_orchestrator_node

builder = StateGraph(PipelineState)
builder.add_node("master_orchestrator", master_orchestrator_node)
builder.add_edge(START, "master_orchestrator")
builder.add_edge("master_orchestrator", END)

workflow = builder.compile()