"""
Validation Tool.

Executes dataset validation and updates the ValidationState.
"""

from state.pipeline_state import PipelineState

from schemas.validation_schema import ValidationOutput


class ValidationTool:
    """
    Performs dataset validation using pandas.
    """

    def execute(
        self,
        pipeline_state: PipelineState,
        validation_output: ValidationOutput
    ) -> PipelineState:

        df = pipeline_state.dataset.dataframe

        validation_state = pipeline_state.validation
        if validation_output.check_missing_values:

            validation_state.missing_values = (
                df.isnull()
                .sum()
                .to_dict()
            )
        if validation_output.check_duplicates:

            validation_state.duplicate_rows = int(
                df.duplicated().sum()
            )
        if validation_output.check_data_types:

            validation_state.data_types = (
                df.dtypes.astype(str).to_dict()
            )
        if validation_output.detect_target_column:

            possible_targets = [

                "target",
                "label",
                "class",
                "output",
                "survived",
                "price",
                "sales",
                "quality",
                "species"

            ]

            for column in df.columns:

                if column.lower() in possible_targets:

                    validation_state.target_column = column

                    break
        if validation_output.infer_problem_type:

            validation_state.problem_type = (
                validation_output.problem_type
            )
        validation_state.is_valid = True

        validation_state.summary = (
            "Dataset validation completed successfully."
        )

        return pipeline_state