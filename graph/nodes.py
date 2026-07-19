"""
LangGraph node functions — one module for all agent nodes.

Each node is a thin wrapper: a single module-level agent instance
is created at import time and the node function delegates to agent.run().
"""

from agents import (
    DatasetResolverAgent,
    DataIngestionAgent,
    ValidationAgent,
    FeatureEngineeringAgent,
    ModelSelectionAgent,
    ModelTrainingAgent,
    HyperparameterOptimizationAgent,
    ModelEvaluationAgent,
    MasterOrchestratorAgent,
    ExplainabilityAgent,
)
from state.pipeline_state import PipelineState


def _make_node(agent_cls):
    """Factory: creates a singleton agent and returns a LangGraph node function."""
    agent = agent_cls()
    def node(state: PipelineState) -> PipelineState:
        return agent.run(state)
    node.__name__ = agent_cls.__name__.replace("Agent", "Node")
    return node


dataset_resolver_node            = _make_node(DatasetResolverAgent)
data_ingestion_node              = _make_node(DataIngestionAgent)
validation_node                  = _make_node(ValidationAgent)
feature_engineering_node         = _make_node(FeatureEngineeringAgent)
model_selection_node             = _make_node(ModelSelectionAgent)
model_training_node              = _make_node(ModelTrainingAgent)
hyperparameter_optimization_node = _make_node(HyperparameterOptimizationAgent)
model_evaluation_node            = _make_node(ModelEvaluationAgent)
master_orchestrator_node         = _make_node(MasterOrchestratorAgent)
explainability_node              = _make_node(ExplainabilityAgent)
