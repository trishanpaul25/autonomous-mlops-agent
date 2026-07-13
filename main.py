from dotenv import load_dotenv

load_dotenv()  # Load .env before anything else so API keys are available

from graph.workflow import workflow

from state.pipeline_state import PipelineState

from utils.logger import logger


def main():
    logger.info("Initialising pipeline...")

    state = PipelineState(

        user_prompt="Build a Titanic survival prediction model using data/titanic.csv"

    )

    logger.info("User prompt: %s", state.user_prompt)

    result = PipelineState.model_validate(workflow.invoke(state))

    logger.info("========== PIPELINE STATUS ==========")
    logger.info("Status: %s", result.status)

    logger.info("========== COMPLETED STEPS ==========")
    logger.info("Steps: %s", result.completed_steps)

    logger.info("========== LOGS ==========")
    for log in result.logs:
        logger.info(log)

    logger.info("========== DATASET ==========")
    logger.info("Metadata: %s", result.dataset.metadata)

    logger.info("========== VALIDATION ==========")
    logger.info("Validation: %s", result.validation.model_dump())


    logger.info("========== FEATURE ENGINEERING ==========")
    logger.info("Feature Engineering: %s", result.feature_engineering.model_dump())

    logger.info("========== MODEL SELECTION ==========")
    logger.info("Task type: %s", result.model_selection.task_type)
    logger.info("Primary model: %s", result.model_selection.primary_model_name)
    logger.info("Primary model class: %s", result.model_selection.primary_model_class_path)
    logger.info("Ranking: %s", result.model_selection.ranking)
    logger.info("Confidence: %.2f", result.model_selection.confidence)
    logger.info("Reasoning: %s", result.model_selection.reasoning)
    if result.model_selection.assumptions:
        logger.info("Assumptions: %s", result.model_selection.assumptions)
    if result.model_selection.warnings:
        logger.info("Warnings: %s", result.model_selection.warnings)

if __name__ == "__main__":

    main()
