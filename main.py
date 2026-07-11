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


if __name__ == "__main__":

    main()
