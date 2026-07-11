"""
LangGraph workflow.

This workflow starts execution from the Master Orchestrator.
The Master Orchestrator is responsible for coordinating
all sub-agents in the MLOps pipeline.
"""

from langgraph.graph import StateGraph, START, END

from state.pipeline_state import PipelineState

from graph.master_orchestrator_node import (
    master_orchestrator_node,
)
# Create the graph
builder = StateGraph(PipelineState)

# Add nodes
builder.add_node(
    "master_orchestrator",
    master_orchestrator_node,
)

# Define graph flow
builder.add_edge(
    START,
    "master_orchestrator",
)

builder.add_edge(
    "master_orchestrator",
    END,
)

# Compile graph
workflow = builder.compile()