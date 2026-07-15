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

    logger.info("========== MODEL TRAINING ==========")
    logger.info("Training status: %s", result.model_training.training_status)
    logger.info("Train samples: %d | Test samples: %d | Stratified: %s",
                result.model_training.train_samples,
                result.model_training.test_samples,
                result.model_training.stratified)
    logger.info("Total training time: %.2fs", result.model_training.total_execution_time_seconds)
    logger.info("Trained models (%d):", len(result.model_training.trained_models))
    for record in result.model_training.trained_models:
        logger.info(
            "  [SUCCESS] %s — %.3fs — id: %s",
            record.get("model_name"),
            record.get("training_time_seconds", 0),
            record.get("model_identifier"),
        )
    if result.model_training.failed_models:
        logger.info("Failed models (%d):", len(result.model_training.failed_models))
        for record in result.model_training.failed_models:
            logger.info(
                "  [FAILED]  %s — %s",
                record.get("model_name"),
                record.get("notes"),
            )
    logger.info("Summary: %s", result.model_training.summary)

    logger.info("========== HYPERPARAMETER OPTIMIZATION ==========")
    hpo = result.hyperparameter_optimization
    logger.info("HPO status: %s", hpo.optimization_status)
    logger.info("Scoring metric: %s", hpo.scoring_metric)
    logger.info("Total HPO time: %.2fs", hpo.total_execution_time_seconds)
    if hpo.best_overall_model_name:
        logger.info(
            "Best model: %s (score=%.4f)",
            hpo.best_overall_model_name,
            hpo.best_overall_score,
        )
    logger.info("Optimized models (%d):", len(hpo.optimized_models))
    for record in hpo.optimized_models:
        logger.info(
            "  [OPTIMIZED] %s — %.3fs — score: %.4f — params: %s",
            record.get("model_name"),
            record.get("optimization_time_seconds", 0),
            record.get("best_score", 0),
            record.get("best_parameters", {}),
        )
    if hpo.failed_models:
        logger.info("Failed/Skipped models (%d):", len(hpo.failed_models))
        for record in hpo.failed_models:
            logger.info(
                "  [%s] %s — %s",
                record.get("optimization_status", "failed").upper(),
                record.get("model_name"),
                record.get("notes"),
            )
    logger.info("Summary: %s", hpo.summary)

    logger.info("========== MODEL EVALUATION ==========")
    ev = result.model_evaluation
    logger.info("Evaluation status: %s", ev.evaluation_status)
    logger.info("Task type: %s", ev.task_type)
    logger.info("Primary metric: %s", ev.primary_metric)
    logger.info("Total evaluation time: %.2fs", ev.total_execution_time_seconds)
    if ev.best_model_name:
        primary_val = ev.best_model_metrics.get(ev.primary_metric or "", 0.0)
        logger.info(
            "Best model: %s (%s=%.4f)",
            ev.best_model_name,
            ev.primary_metric,
            primary_val,
        )
    logger.info("Evaluated models (%d):", len(ev.evaluated_models))
    for record in ev.evaluated_models:
        metrics = record.get("metrics", {})
        metrics_str = " | ".join(
            f"{k}={v:.4f}" for k, v in sorted(metrics.items())
        )
        logger.info(
            "  [Rank %d] %s — %s (%.3fs)",
            record.get("rank", 0),
            record.get("model_name"),
            metrics_str,
            record.get("prediction_time_seconds", 0),
        )
    if ev.failed_models:
        logger.info("Failed/Skipped models (%d):", len(ev.failed_models))
        for record in ev.failed_models:
            logger.info(
                "  [%s] %s — %s",
                record.get("evaluation_status", "failed").upper(),
                record.get("model_name"),
                record.get("notes"),
            )
    if ev.comparison_table:
        logger.info("Comparison table:")
        for row in ev.comparison_table:
            row_str = " | ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in row.items()
                if k not in ("model_name", "prediction_time_seconds")
            )
            logger.info("  %s: %s", row.get("model_name"), row_str)
    if ev.visualization_data:
        logger.info(
            "Visualization data available for: %s",
            list(ev.visualization_data.keys()),
        )
    if ev.narrative:
        logger.info("Narrative: %s", ev.narrative)
    logger.info("Summary: %s", ev.summary)

if __name__ == "__main__":

    main()
