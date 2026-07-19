"""
explainability_prompt.py

Prompt construction for the LLM-narration step of the Explainability
Agent.

Hard rule enforced throughout this module: the LLM is given ONLY
already-computed numeric results (feature ranking, global explanation,
sample metrics) and asked to phrase them in natural language. Every
prompt explicitly instructs the model not to invent or recompute numbers,
and to use only the figures supplied in the prompt itself.
"""

from __future__ import annotations

from typing import List

from schemas.explainability_schema import GlobalExplanation, TaskType, UnifiedFeatureRanking

_SHARED_GUARDRAIL = (
    "You are narrating machine learning explainability results that have "
    "already been computed by deterministic statistical methods (SHAP, "
    "permutation importance, model coefficients, etc.). "
    "Do NOT invent, estimate, or recompute any numbers. "
    "Use ONLY the feature names, scores, and directions provided below. "
    "If a number is not given, do not state a number for it."
)


def _format_ranking(ranking: List[UnifiedFeatureRanking], top_n: int = 10) -> str:
    lines = []
    for r in ranking[:top_n]:
        lines.append(
            f"- {r.feature_name}: overall_score={r.overall_score:.4f}, "
            f"rank={r.overall_rank}"
        )
    return "\n".join(lines) if lines else "(no ranked features available)"


def build_technical_explanation_prompt(
    ranking: List[UnifiedFeatureRanking],
    global_explanation: GlobalExplanation,
    task_type: TaskType,
) -> str:
    """Prompt for an audience of ML practitioners: precise, references
    method names and scores."""
    return (
        f"{_SHARED_GUARDRAIL}\n\n"
        f"Task type: {task_type.value}\n\n"
        f"Unified feature ranking (top features, higher score = more important):\n"
        f"{_format_ranking(ranking)}\n\n"
        f"Most important features: {', '.join(global_explanation.most_important_features)}\n"
        f"Least important features: {', '.join(global_explanation.least_important_features)}\n"
        f"Positively influential features: "
        f"{', '.join(global_explanation.positively_influential_features) or 'not determined'}\n"
        f"Negatively influential features: "
        f"{', '.join(global_explanation.negatively_influential_features) or 'not determined'}\n\n"
        "Write a technical explanation (3-5 sentences) for a data scientist "
        "audience. Reference the ranking and importance methodology in "
        "general terms (e.g. 'combined SHAP, permutation, and native "
        "importance signal'). Do not state specific decimal scores beyond "
        "what is given above."
    )


def build_business_explanation_prompt(
    ranking: List[UnifiedFeatureRanking],
    global_explanation: GlobalExplanation,
    task_type: TaskType,
) -> str:
    """Prompt for an audience of business stakeholders: outcome-focused,
    minimal jargon, emphasizes actionability."""
    return (
        f"{_SHARED_GUARDRAIL}\n\n"
        f"Task type: {task_type.value}\n\n"
        f"Most important features driving predictions: "
        f"{', '.join(global_explanation.most_important_features)}\n"
        f"Least important features: {', '.join(global_explanation.least_important_features)}\n"
        f"Features that increase the predicted outcome: "
        f"{', '.join(global_explanation.positively_influential_features) or 'not determined'}\n"
        f"Features that decrease the predicted outcome: "
        f"{', '.join(global_explanation.negatively_influential_features) or 'not determined'}\n\n"
        "Write a business explanation (2-4 sentences) for a non-technical "
        "stakeholder (e.g. a product manager or executive). Avoid statistical "
        "jargon (no 'SHAP', 'permutation', 'coefficient'). Focus on what "
        "drives the outcome and what that means for the business, in the "
        "spirit of: 'The model primarily relies on X and Y to make "
        "predictions. Higher X increases the likelihood of the outcome, "
        "while lower Y decreases it.'"
    )


def build_non_technical_explanation_prompt(
    global_explanation: GlobalExplanation,
    task_type: TaskType,
) -> str:
    """Prompt for a general/non-technical audience: plain language, one or
    two short sentences."""
    return (
        f"{_SHARED_GUARDRAIL}\n\n"
        f"Task type: {task_type.value}\n\n"
        f"Most important factors: {', '.join(global_explanation.most_important_features)}\n"
        f"Factors that increase the result: "
        f"{', '.join(global_explanation.positively_influential_features) or 'not determined'}\n"
        f"Factors that decrease the result: "
        f"{', '.join(global_explanation.negatively_influential_features) or 'not determined'}\n\n"
        "In 1-2 simple sentences, explain what mainly drives the model's "
        "predictions, in plain everyday language suitable for someone with "
        "no data science background."
    )