"""
LangGraph node for the Hyperparameter Optimization Agent.

Follows the same thin-wrapper pattern as all other agent nodes:
a single module-level agent instance is created at import time and
the node function delegates directly to agent.run().

This node is available for future use if the workflow is refactored
to invoke HPO as a standalone LangGraph node rather than through the
Master Orchestrator. Currently, the Master Orchestrator calls
HyperparameterOptimizationAgent.run() directly in its execution loop.
"""

from agents.hyperparameter_optimization_agent import (
    HyperparameterOptimizationAgent,
)

from state.pipeline_state import PipelineState


# Single instance reused across all graph invocations
agent = HyperparameterOptimizationAgent()


def hyperparameter_optimization_node(
    state: PipelineState,
) -> PipelineState:
    """
    Executes the Hyperparameter Optimization Agent within the LangGraph
    workflow.

    Parameters
    ----------
    state : PipelineState
        Shared pipeline state passed in by LangGraph.
        Must have model_training.is_completed == True.

    Returns
    -------
    PipelineState
        Updated state after hyperparameter optimization has been
        attempted for all trained candidate models.
    """
    return agent.run(state)
