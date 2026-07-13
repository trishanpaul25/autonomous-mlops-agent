"""
Master Orchestrator Agent.

Coordinates the execution of all pipeline agents.

Responsibilities
----------------
1. Execute agents in the correct order.
2. Stop the pipeline if an agent fails.
3. Handle user clarification requests.
4. Return the updated PipelineState.
"""

from agents.base_agent import BaseAgent

from agents.dataset_resolver_agent import DatasetResolverAgent
from agents.data_ingestion_agent import DataIngestionAgent
from agents.validation_agent import ValidationAgent
from agents.feature_engineering_agent import FeatureEngineeringAgent
from agents.model_selection_agent import ModelSelectionAgent

from state.pipeline_state import PipelineState

from utils.logger import logger


class MasterOrchestratorAgent(BaseAgent):

    def __init__(self):

        # Register every agent here
        self.agents = {

            "dataset_resolver": DatasetResolverAgent(),

            "data_ingestion": DataIngestionAgent(),

            "validation": ValidationAgent(),

            "feature_engineering": FeatureEngineeringAgent(),

            "model_selection": ModelSelectionAgent(),
            # "training": TrainingAgent(),
            # "evaluation": EvaluationAgent(),
            # "registry": RegistryAgent(),
            # "deployment": DeploymentAgent(),
            # "monitoring": MonitoringAgent(),
        }

        # Current execution order
        self.execution_order = [

            "dataset_resolver",

            "data_ingestion",

            "validation",

            "feature_engineering",

            "model_selection",

        ]

    def run(
        self,
        state: PipelineState
    ) -> PipelineState:

        state.current_agent = "MasterOrchestrator"

        state.status = "running"

        logger.info("Pipeline started. Execution order: %s", self.execution_order)

        for agent_name in self.execution_order:

            agent = self.agents[agent_name]

            logger.info("Dispatching agent: %s", agent_name)

            state = agent.run(state)

            if state.status == "failed":

                logger.error(
                    "Pipeline stopped at agent '%s'. Error: %s",
                    agent_name,
                    state.error,
                )

                state.logs.append(
                    f"Pipeline stopped at {agent_name}."
                )

                return state

            if state.status == "waiting_for_user":

                logger.warning(
                    "Pipeline paused at agent '%s' — waiting for user input.",
                    agent_name,
                )

                return state

        state.status = "completed"

        logger.info("Pipeline completed successfully. Steps: %s", state.completed_steps)

        state.logs.append(
            "Pipeline executed successfully."
        )

        return state