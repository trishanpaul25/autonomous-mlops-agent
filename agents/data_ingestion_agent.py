"""
Agent responsible for loading the dataset into memory.

The Data Ingestion Agent reads the dataset information from the
PipelineState and uses the IngestionTool to load the dataset.
"""

from agents.base_agent import BaseAgent

from state.pipeline_state import PipelineState

from schemas.dataset_resolver_schema import DatasetResolverOutput

from tools.ingestion.ingestion_tool import IngestionTool

from utils.logger import logger


class DataIngestionAgent(BaseAgent):
    """
    Loads the dataset and updates the DatasetState.
    """

    def __init__(self):

        self.ingestion_tool = IngestionTool()

    def run(
        self,
        state: PipelineState
    ) -> PipelineState:

        try:

            state.current_agent = "DataIngestionAgent"

            logger.info(
                "Starting data ingestion for dataset: %s",
                state.dataset.dataset_name or state.dataset.dataset_path,
            )

            # Create a resolver output object from the state
            resolver_output = DatasetResolverOutput(
                source_type=state.dataset.source_type,
                source=state.dataset.dataset_path,
                dataset_name=state.dataset.dataset_name,
                reasoning="Resolved by Dataset Resolver Agent.",
                confidence=1.0,
                needs_clarification=False,
                clarification_question=None,
            )

            state = self.ingestion_tool.execute(
                state,
                resolver_output,
            )

            state.completed_steps.append(
                "Data Ingestion"
            )

            logger.info(
                "Data ingestion completed. Rows: %s, Columns: %s",
                state.dataset.dataframe.shape[0] if state.dataset.dataframe is not None else "N/A",
                state.dataset.dataframe.shape[1] if state.dataset.dataframe is not None else "N/A",
            )

            state.logs.append(
                "Data ingestion completed successfully."
            )

            return state

        except Exception as e:

            state.status = "failed"

            state.error = str(e)

            logger.error("Data ingestion failed: %s", e, exc_info=True)

            state.logs.append(
                f"Data ingestion failed: {e}"
            )

            return state