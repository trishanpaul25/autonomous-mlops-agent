"""
tools/ingestion/ingestion_tool.py

Tool responsible for loading the dataset, generating metadata,
and updating the PipelineState.

This tool performs execution only.
It does not contain any LLM reasoning.
"""

from state.pipeline_state import PipelineState
from schemas.dataset_resolver_schema import DatasetResolverOutput

from tools.base_tool import BaseTool

from .dataset_loader import DatasetLoader
from .metadata_generator import MetadataGenerator


class IngestionTool(BaseTool):
    """
    Executes dataset ingestion and updates the PipelineState.
    """

    def __init__(self):
        self.dataset_loader = DatasetLoader()

    def execute(
        self,
        pipeline_state: PipelineState,
        resolver_output: DatasetResolverOutput,
    ) -> PipelineState:
        """
        Load the dataset, generate metadata, and update the PipelineState.
        """

        try:

            # ------------------------------------
            # Load Dataset
            # ------------------------------------

            df = self.dataset_loader.load(
                resolver_output.source_type,
                resolver_output.source,
            )

            # ------------------------------------
            # Generate Metadata
            # ------------------------------------

            metadata = MetadataGenerator.generate(
                df,
                resolver_output.source,
            )

            # ------------------------------------
            # Update Dataset State
            # ------------------------------------

            pipeline_state.dataset.source_type = resolver_output.source_type

            pipeline_state.dataset.dataset_name = (
                resolver_output.dataset_name
            )

            pipeline_state.dataset.dataset_path = (
                resolver_output.source
            )

            pipeline_state.dataset.current_dataset_path = (
                resolver_output.source
            )

            pipeline_state.dataset.dataframe = df

            pipeline_state.dataset.metadata = metadata

            pipeline_state.dataset.num_rows = len(df)

            pipeline_state.dataset.num_columns = len(df.columns)

            pipeline_state.dataset.loaded = True

            # Initial dataset version
            pipeline_state.dataset.dataset_version = "v1"

            # ------------------------------------
            # Update Pipeline State
            # ------------------------------------

            pipeline_state.logs.append(
                "Dataset loaded successfully."
            )

            pipeline_state.completed_steps.append(
                "Data Ingestion"
            )

            pipeline_state.status = "running"

            return pipeline_state

        except Exception as e:

            pipeline_state.status = "failed"

            pipeline_state.error = str(e)

            pipeline_state.logs.append(
                f"Data ingestion failed: {e}"
            )

            return pipeline_state