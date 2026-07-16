"""
Validation Tool — executes dataset validation checks and updates ValidationState.
"""
from schemas.validation_schema import ValidationOutput
from state.pipeline_state import PipelineState
_COMMON_TARGET_KEYWORDS = {
    "target", "label", "class", "output", "y", "survived",
    "price", "sales", "quality", "species", "diagnosis",
    "churn", "outcome", "result", "attrition", "default"
}


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
            # Layer 1: Check what the LLM detected from user prompt & metadata
            detected_target = None
            if hasattr(validation_output, "target_column") and validation_output.target_column and validation_output.target_column in df.columns:
                detected_target = validation_output.target_column

            # Layer 2: Check if user prompt explicitly mentions any column name verbatim
            if not detected_target and pipeline_state.user_prompt:
                prompt_lower = pipeline_state.user_prompt.lower()
                for col in df.columns:
                    if f"predict {col.lower()}" in prompt_lower or f"classifying {col.lower()}" in prompt_lower or f"target is {col.lower()}" in prompt_lower:
                        detected_target = col
                        break

            # Layer 3: Check heuristics & keywords (exact match or substring)
            if not detected_target:
                for col in df.columns:
                    col_clean = col.lower().strip()
                    if col_clean in _COMMON_TARGET_KEYWORDS or any(kw in col_clean for kw in ["target", "label", "class", "y_"]):
                        detected_target = col
                        break

            # Layer 4: Fallback to the last column if classification/regression problem type is known
            if not detected_target and validation_output.problem_type in ["classification", "regression"] and len(df.columns) > 1:
                detected_target = df.columns[-1]

            vs.target_column = detected_target

        if validation_output.infer_problem_type:
            vs.problem_type = validation_output.problem_type

        vs.is_valid = True
        vs.summary = f"Dataset validation completed successfully. Target column detected: {vs.target_column}"
        return pipeline_state