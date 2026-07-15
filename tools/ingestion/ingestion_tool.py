"""
Ingestion Tool — loads dataset, generates metadata, and updates PipelineState.
No LLM reasoning; execution only.
"""

from schemas.dataset_resolver_schema import DatasetResolverOutput
from state.pipeline_state import PipelineState
from tools.base_tool import BaseTool
from .dataset_loader import DatasetLoader
from .metadata_generator import MetadataGenerator


class IngestionTool(BaseTool):
    """Executes dataset ingestion and updates the PipelineState."""

    def __init__(self):
        self.dataset_loader = DatasetLoader()

    def execute(self, pipeline_state: PipelineState, resolver_output: DatasetResolverOutput) -> PipelineState:
        """Load the dataset, generate metadata, and update the PipelineState."""
        try:
            df = self.dataset_loader.load(resolver_output.source_type, resolver_output.source)
            metadata = MetadataGenerator.generate(df, resolver_output.source)

            ds = pipeline_state.dataset
            ds.source_type = resolver_output.source_type
            ds.dataset_name = resolver_output.dataset_name
            ds.dataset_path = resolver_output.source
            ds.current_dataset_path = resolver_output.source
            ds.dataframe = df
            ds.metadata = metadata
            ds.num_rows = len(df)
            ds.num_columns = len(df.columns)
            ds.loaded = True
            ds.dataset_version = "v1"

            pipeline_state.logs.append("Dataset loaded successfully.")
            pipeline_state.status = "running"
            return pipeline_state

        except Exception as e:
            pipeline_state.status = "failed"
            pipeline_state.error = str(e)
            pipeline_state.logs.append(f"Data ingestion failed: {e}")
            return pipeline_state