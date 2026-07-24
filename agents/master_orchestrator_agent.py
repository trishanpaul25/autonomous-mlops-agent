"""
Master Orchestrator Agent — coordinates all pipeline agents in sequence.
"""

import os

from agents.base_agent import BaseAgent
from agents.dataset_resolver_agent import DatasetResolverAgent
from agents.data_ingestion_agent import DataIngestionAgent
from agents.validation_agent import ValidationAgent
from agents.feature_engineering_agent import FeatureEngineeringAgent
from agents.model_selection_agent import ModelSelectionAgent
from agents.model_training_agent import ModelTrainingAgent
from agents.hyperparameter_optimization_agent import HyperparameterOptimizationAgent
from agents.model_evaluation_agent import ModelEvaluationAgent
from agents.explainability_agent import ExplainabilityAgent, LangChainLLMAdapter
from agents.model_registry_agent import ModelRegistryAgent
from agents.deployment_agent import DeploymentAgent
from server.core.constants import PipelineStatus

from state.pipeline_state import PipelineState
from utils.logger import logger

from server.services.progress_service import ProgressService
from server.services.progress_messages import PROGRESS_MESSAGES
from server.services.progress_types import ProgressEventType


class MasterOrchestratorAgent(BaseAgent):

    def __init__(self):
        # ExplainabilityAgent takes its LLM client injected (rather than
        # resolving it internally like the other agents), so we build it
        # here using the same GOOGLE_API_KEY/GEMINI_API_KEY auto-detection
        # convention the rest of the pipeline follows. LLMService.get_llm()
        # returns a plain (non-structured-output) chat model, which is what
        # LangChainLLMAdapter's .generate(prompt) -> str contract needs —
        # get_structured_llm() would be wrong here since narration is free
        # text, not a Pydantic schema.
        explainability_llm_client = None
        if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
            from services.llm_service import LLMService
            explainability_llm_client = LangChainLLMAdapter(LLMService.get_llm())

        self.agents = {
            "dataset_resolver":            DatasetResolverAgent(),
            "data_ingestion":              DataIngestionAgent(),
            "validation":                  ValidationAgent(),
            "feature_engineering":         FeatureEngineeringAgent(),
            "model_selection":             ModelSelectionAgent(),
            "model_training":              ModelTrainingAgent(),
            "hyperparameter_optimization": HyperparameterOptimizationAgent(),
            "model_evaluation":            ModelEvaluationAgent(),
            "explainability":              ExplainabilityAgent(llm_client=explainability_llm_client),
            "model_registry":              ModelRegistryAgent(),
            "deployment":                  DeploymentAgent(),
        }
        self.execution_order = [
            "dataset_resolver", "data_ingestion", "validation",
            "feature_engineering", "model_selection", "model_training",
            "hyperparameter_optimization", "model_evaluation",
            "explainability", "model_registry", "deployment",
        ]

    def run(self, state: PipelineState) -> PipelineState:
        state.current_agent = "MasterOrchestrator"

        state.status = PipelineStatus.RUNNING

        logger.info("Pipeline started. Execution order: %s", self.execution_order)

        for agent_name in self.execution_order:
            agent = self.agents[agent_name]
            logger.info("Dispatching agent: %s", agent_name)

            ProgressService.emit(
                state.run_id,
                PROGRESS_MESSAGES[agent_name],
                ProgressEventType.STEP,
            )
            
            state = agent.run(state)

            if state.status == PipelineStatus.FAILED:

                logger.error(
                    "Pipeline stopped at agent '%s'. Error: %s",
                    agent_name, state.error,
                )
                state.logs.append(f"Pipeline stopped at {agent_name}.")

                ProgressService.emit(
                    state.run_id,
                    f"❌ Pipeline failed in {agent_name}",
                    ProgressEventType.ERROR,
                )

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

            # ModelRegistryAgent also never touches state.status for the
            # same reason (a registration failure shouldn't block a
            # pipeline that otherwise produced a perfectly good model).
            if agent_name == "model_registry":
                reg_status = getattr(state.model_registry, "registry_status", None)
                if reg_status == "failed":
                    logger.warning(
                        "Model registration failed (pipeline continuing): %s",
                        getattr(state.model_registry, "errors", None),
                    )
                elif reg_status == "skipped":
                    logger.info(
                        "Model registration skipped: %s",
                        getattr(state.model_registry, "warnings", None),
                    )
                elif reg_status == "completed":
                    logger.info(
                        "Model registered: %s v%s (%s)",
                        state.model_registry.registered_model_name,
                        state.model_registry.model_version,
                        state.model_registry.mlflow_model_uri,
                    )

            # DeploymentAgent also never touches state.status for the same
            # reason (a deployment failure shouldn't block a pipeline that
            # otherwise produced and registered a perfectly good model).
            if agent_name == "deployment":
                dep_status = getattr(state.deployment, "deployment_status", None)
                if dep_status == "failed":
                    logger.warning(
                        "Deployment failed (pipeline continuing): %s",
                        getattr(state.deployment, "errors", None),
                    )
                elif dep_status == "skipped":
                    logger.info(
                        "Deployment skipped: %s",
                        getattr(state.deployment, "warnings", None),
                    )
                elif dep_status == "completed":
                    logger.info(
                        "Model deployed: %s (%s)",
                        state.deployment.model_uri,
                        state.deployment.endpoint,
                    )

        ProgressService.emit(
            state.run_id,
            "🎉 Pipeline completed successfully!",
            ProgressEventType.COMPLETE,
        )

        state.status = PipelineStatus.SUCCESS

        logger.info("Pipeline completed successfully. Steps: %s", state.completed_steps)
        state.logs.append("Pipeline executed successfully.")
        return state