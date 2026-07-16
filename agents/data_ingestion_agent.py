"""
Data Ingestion Agent — loads the dataset into memory and updates DatasetState.
"""

from agents.base_agent import BaseAgent
from schemas.dataset_resolver_schema import DatasetResolverOutput
from state.pipeline_state import PipelineState
from tools.ingestion.ingestion_tool import IngestionTool
from utils.logger import logger

from server.core.constants import PipelineStatus

class DataIngestionAgent(BaseAgent):
    """Loads the dataset and updates the DatasetState."""

    def __init__(self):
        self.ingestion_tool = IngestionTool()

    def run(self, state: PipelineState) -> PipelineState:
        try:
            state.current_agent = "DataIngestionAgent"
            logger.info(
                "Starting data ingestion for dataset: %s",
                state.dataset.dataset_name or state.dataset.dataset_path,
            )

            resolver_output = DatasetResolverOutput(
                source_type=state.dataset.source_type,
                source=state.dataset.dataset_path,
                dataset_name=state.dataset.dataset_name,
                reasoning="Resolved by Dataset Resolver Agent.",
                confidence=1.0,
                needs_clarification=False,
                clarification_question=None,
            )

            state = self.ingestion_tool.execute(state, resolver_output)
            state.completed_steps.append("Data Ingestion")

            df = state.dataset.dataframe
            rows = df.shape[0] if df is not None else "N/A"
            cols = df.shape[1] if df is not None else "N/A"
            logger.info("Data ingestion completed. Rows: %s, Columns: %s", rows, cols)
            state.logs.append("Data ingestion completed successfully.")
            return state

        except Exception as e:

            state.status = PipelineStatus.FAILED

            state.error = str(e)
            logger.error("Data ingestion failed: %s", e, exc_info=True)
            state.logs.append(f"Data ingestion failed: {e}")
            return state