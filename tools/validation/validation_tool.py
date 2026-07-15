"""
Validation Tool — executes dataset validation checks and updates ValidationState.
"""

from schemas.validation_schema import ValidationOutput
from state.pipeline_state import PipelineState

_POSSIBLE_TARGETS = {"target", "label", "class", "output", "survived", "price", "sales", "quality", "species"}


class ValidationTool:
    """Performs dataset validation using pandas."""

    def execute(self, pipeline_state: PipelineState, validation_output: ValidationOutput) -> PipelineState:
        df = pipeline_state.dataset.dataframe
        vs = pipeline_state.validation

        if validation_output.check_missing_values:
            vs.missing_values = df.isnull().sum().to_dict()

        if validation_output.check_duplicates:
            vs.duplicate_rows = int(df.duplicated().sum())

        if validation_output.check_data_types:
            vs.data_types = df.dtypes.astype(str).to_dict()

        if validation_output.detect_target_column:
            for col in df.columns:
                if col.lower() in _POSSIBLE_TARGETS:
                    vs.target_column = col
                    break

        if validation_output.infer_problem_type:
            vs.problem_type = validation_output.problem_type

        vs.is_valid = True
        vs.summary = "Dataset validation completed successfully."
        return pipeline_state