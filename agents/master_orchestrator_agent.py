"""
Master Orchestrator Agent — coordinates all pipeline agents in sequence.
"""

from agents.base_agent import BaseAgent
from agents.dataset_resolver_agent import DatasetResolverAgent
from agents.data_ingestion_agent import DataIngestionAgent
from agents.validation_agent import ValidationAgent
from agents.feature_engineering_agent import FeatureEngineeringAgent
from agents.model_selection_agent import ModelSelectionAgent
from agents.model_training_agent import ModelTrainingAgent
from agents.hyperparameter_optimization_agent import HyperparameterOptimizationAgent
from agents.model_evaluation_agent import ModelEvaluationAgent
from agents.explainability_agent import ExplainabilityAgent
from server.core.constants import PipelineStatus

from state.pipeline_state import PipelineState
from utils.logger import logger


class MasterOrchestratorAgent(BaseAgent):

    def __init__(self):
        self.agents = {
            "dataset_resolver":            DatasetResolverAgent(),
            "data_ingestion":              DataIngestionAgent(),
            "validation":                  ValidationAgent(),
            "feature_engineering":         FeatureEngineeringAgent(),
            "model_selection":             ModelSelectionAgent(),
            "model_training":              ModelTrainingAgent(),
            "hyperparameter_optimization": HyperparameterOptimizationAgent(),
            "model_evaluation":            ModelEvaluationAgent(),
            "explainability":              ExplainabilityAgent(),
        }
        self.execution_order = [
            "dataset_resolver", "data_ingestion", "validation",
            "feature_engineering", "model_selection", "model_training",
            "hyperparameter_optimization", "model_evaluation",
            "explainability",
        ]

    def run(self, state: PipelineState) -> PipelineState:
        state.current_agent = "MasterOrchestrator"

        state.status = PipelineStatus.RUNNING

        logger.info("Pipeline started. Execution order: %s", self.execution_order)

        for agent_name in self.execution_order:
            agent = self.agents[agent_name]
            logger.info("Dispatching agent: %s", agent_name)
            state = agent.run(state)

            if state.status == PipelineStatus.FAILED:

                logger.error(
                    "Pipeline stopped at agent '%s'. Error: %s",
                    agent_name, state.error,
                )
                state.logs.append(f"Pipeline stopped at {agent_name}.")
                return state

            if state.status == PipelineStatus.WAITING_FOR_USER:

                logger.warning(
                    "Pipeline paused at agent '%s' — waiting for user input.",
                    agent_name,
                )
                return state

            # ExplainabilityAgent never touches state.status (a failed or
            # skipped explanation shouldn't halt the pipeline the way a
            # broken train/test split would), so it can't trip the
            # PipelineStatus.FAILED check above. Surface it as a visible,
            # non-blocking warning instead of letting it fail silently.
            if agent_name == "explainability":
                ex_status = getattr(state.explainability, "explainability_status", None)
                if ex_status == "failed":
                    logger.warning(
                        "Explainability step failed (pipeline continuing): %s",
                        getattr(state.explainability, "errors", None),
                    )
                elif ex_status == "skipped":
                    logger.info(
                        "Explainability step skipped: %s",
                        getattr(state.explainability, "warnings", None),
                    )

        state.status = PipelineStatus.SUCCESS

        logger.info("Pipeline completed successfully. Steps: %s", state.completed_steps)
        state.logs.append("Pipeline executed successfully.")
        return state