from dotenv import load_dotenv

load_dotenv()  # Load .env before anything else so API keys are available

from graph.workflow import workflow
from state.pipeline_state import PipelineState
from utils.logger import logger


def _section(title: str) -> None:
    logger.info("========== %s ==========", title)


def main():
    logger.info("Initialising pipeline...")

    state = PipelineState(

        user_prompt="Build a Titanic survival prediction model using data/titanic.csv"
    )
    logger.info("User prompt: %s", state.user_prompt)

    result = PipelineState.model_validate(workflow.invoke(state))

    _section("PIPELINE STATUS")
    logger.info("Status: %s", result.status)

    _section("COMPLETED STEPS")
    logger.info("Steps: %s", result.completed_steps)

    _section("LOGS")
    for log in result.logs:
        logger.info(log)

    _section("DATASET")
    logger.info("Metadata: %s", result.dataset.metadata)

    _section("VALIDATION")
    logger.info("Validation: %s", result.validation.model_dump())

    _section("FEATURE ENGINEERING")
    logger.info("Feature Engineering: %s", result.feature_engineering.model_dump())

    _section("MODEL SELECTION")
    ms = result.model_selection
    logger.info("Task type: %s", ms.task_type)
    logger.info("Primary model: %s", ms.primary_model_name)
    logger.info("Primary model class: %s", ms.primary_model_class_path)
    logger.info("Ranking: %s", ms.ranking)
    logger.info("Confidence: %.2f", ms.confidence)
    logger.info("Reasoning: %s", ms.reasoning)
    if ms.assumptions:
        logger.info("Assumptions: %s", ms.assumptions)
    if ms.warnings:
        logger.info("Warnings: %s", ms.warnings)

    _section("MODEL TRAINING")
    mt = result.model_training
    logger.info("Training status: %s", mt.training_status)
    logger.info(
        "Train samples: %d | Test samples: %d | Stratified: %s",
        mt.train_samples, mt.test_samples, mt.stratified,
    )
    logger.info("Total training time: %.2fs", mt.total_execution_time_seconds)
    logger.info("Trained models (%d):", len(mt.trained_models))
    for r in mt.trained_models:
        logger.info("  [SUCCESS] %s — %.3fs — id: %s",
                    r.get("model_name"), r.get("training_time_seconds", 0), r.get("model_identifier"))
    if mt.failed_models:
        logger.info("Failed models (%d):", len(mt.failed_models))
        for r in mt.failed_models:
            logger.info("  [FAILED]  %s — %s", r.get("model_name"), r.get("notes"))
    logger.info("Summary: %s", mt.summary)

    _section("HYPERPARAMETER OPTIMIZATION")
    hpo = result.hyperparameter_optimization
    logger.info("HPO status: %s", hpo.optimization_status)
    logger.info("Scoring metric: %s", hpo.scoring_metric)
    logger.info("Total HPO time: %.2fs", hpo.total_execution_time_seconds)
    if hpo.best_overall_model_name:
        logger.info("Best model: %s (score=%.4f)", hpo.best_overall_model_name, hpo.best_overall_score)
    logger.info("Optimized models (%d):", len(hpo.optimized_models))
    for r in hpo.optimized_models:
        logger.info("  [OPTIMIZED] %s — %.3fs — score: %.4f — params: %s",
                    r.get("model_name"), r.get("optimization_time_seconds", 0),
                    r.get("best_score", 0), r.get("best_parameters", {}))
    if hpo.failed_models:
        logger.info("Failed/Skipped models (%d):", len(hpo.failed_models))
        for r in hpo.failed_models:
            logger.info("  [%s] %s — %s",
                        r.get("optimization_status", "failed").upper(), r.get("model_name"), r.get("notes"))
    logger.info("Summary: %s", hpo.summary)

    _section("MODEL EVALUATION")
    ev = result.model_evaluation
    logger.info("Evaluation status: %s", ev.evaluation_status)
    logger.info("Task type: %s", ev.task_type)
    logger.info("Primary metric: %s", ev.primary_metric)
    logger.info("Total evaluation time: %.2fs", ev.total_execution_time_seconds)
    if ev.best_model_name:
        primary_val = ev.best_model_metrics.get(ev.primary_metric or "", 0.0)
        logger.info("Best model: %s (%s=%.4f)", ev.best_model_name, ev.primary_metric, primary_val)
    logger.info("Evaluated models (%d):", len(ev.evaluated_models))
    for r in ev.evaluated_models:
        metrics_str = " | ".join(f"{k}={v:.4f}" for k, v in sorted(r.get("metrics", {}).items()))
        logger.info("  [Rank %d] %s — %s (%.3fs)",
                    r.get("rank", 0), r.get("model_name"), metrics_str, r.get("prediction_time_seconds", 0))
    if ev.failed_models:
        logger.info("Failed/Skipped models (%d):", len(ev.failed_models))
        for r in ev.failed_models:
            logger.info("  [%s] %s — %s",
                        r.get("evaluation_status", "failed").upper(), r.get("model_name"), r.get("notes"))
    if ev.comparison_table:
        logger.info("Comparison table:")
        for row in ev.comparison_table:
            row_str = " | ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in row.items() if k not in ("model_name", "prediction_time_seconds")
            )
            logger.info("  %s: %s", row.get("model_name"), row_str)
    if ev.visualization_data:
        logger.info("Visualization data available for: %s", list(ev.visualization_data.keys()))
    if ev.narrative:
        logger.info("Narrative: %s", ev.narrative)
    logger.info("Summary: %s", ev.summary)

    _section("EXPLAINABILITY")
    ex = result.explainability
    logger.info("Explainability status: %s", ex.explainability_status)
    logger.info("Task type: %s", ex.task_type)
    logger.info(
        "Techniques computed — SHAP: %s (%s) | Permutation: %s | Native: %s | "
        "Coefficient: %s | Partial dependence: %s",
        ex.shap_computed, ex.shap_explainer_type,
        ex.permutation_computed, ex.native_importance_computed,
        ex.coefficient_computed, ex.partial_dependence_computed,
    )
    if ex.skipped_methods:
        logger.info("Skipped/fallback methods:")
        for method, reason in ex.skipped_methods.items():
            logger.info("  [SKIPPED] %s — %s", method, reason)
    if ex.feature_ranking:
        logger.info("Top features (unified ranking):")
        for row in ex.feature_ranking[:10]:
            logger.info(
                "  #%d %s — overall_score=%.4f",
                row.get("overall_rank", 0),
                row.get("feature_name"),
                row.get("overall_score", 0.0),
            )
    if ex.global_explanation:
        logger.info("Global explanation: %s", ex.global_explanation.get("summary"))
    if ex.local_explanations:
        logger.info("Local explanations (%d):", len(ex.local_explanations))
        for le in ex.local_explanations:
            logger.info(
                "  Sample %s — predicted=%s — %s",
                le.get("sample_index"), le.get("predicted_value"),
                le.get("prediction_explanation"),
            )
    if ex.technical_explanation:
        logger.info("Technical explanation: %s", ex.technical_explanation)
    if ex.business_explanation:
        logger.info("Business explanation: %s", ex.business_explanation)
    if ex.non_technical_explanation:
        logger.info("Non-technical explanation: %s", ex.non_technical_explanation)
    if ex.warnings:
        logger.info("Warnings: %s", ex.warnings)
    if ex.errors:
        logger.info("Errors: %s", ex.errors)
    logger.info("Total explainability time: %.2fs", ex.total_execution_time_seconds)
    logger.info("Summary: %s", ex.summary)


if __name__ == "__main__":
    main()