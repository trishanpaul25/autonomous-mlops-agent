"""
Scoring Strategy Selector.

Provides a single source of truth for which scoring metric to use
during hyperparameter cross-validation search, based on the ML task type.

Design Principles
-----------------
* Pure function: no state, no side effects, no I/O.
* Single Responsibility: this module only maps task_type → scorer string.
* Open/Closed: adding a new task type requires only a new dict entry.

The scoring strings are compatible with scikit-learn's `scoring` parameter
in RandomizedSearchCV / GridSearchCV.

Reference
---------
https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
"""

from __future__ import annotations

from utils.logger import logger


# ---------------------------------------------------------------------------
# Scoring metric mapping
# ---------------------------------------------------------------------------
# Keys: task_type values stored in ModelSelectionState.task_type
# Values: scikit-learn scoring strings, or None if HPO is not supported
#
# None means the HPOptimizer should skip all models for that task type.
# The agent will mark the entire step as "skipped" rather than "failed".
# ---------------------------------------------------------------------------

_TASK_TYPE_TO_SCORING: dict[str, str | None] = {
    "binary_classification":      "roc_auc",
    "multiclass_classification":  "f1_weighted",
    "regression":                 "r2",
    "clustering":                 None,       # No label-based scorer available
    "time_series":                None,       # Requires specialised CV; skip
}


class ScoringStrategySelector:
    """
    Maps ML task types to scikit-learn cross-validation scoring metrics.

    This class encapsulates the mapping so:
    - The HPOptimizer does not contain any task-type branching logic.
    - New task types can be supported by updating _TASK_TYPE_TO_SCORING only.

    Methods
    -------
    get_scoring_metric(task_type) -> str | None
        Returns the scorer string, or None if the task type is not
        supported for standard cross-validation HPO.
    is_supported(task_type) -> bool
        Returns True if HPO is applicable to the given task type.
    """

    def get_scoring_metric(self, task_type: str) -> str | None:
        """
        Return the scikit-learn scoring metric string for a task type.

        Parameters
        ----------
        task_type : str
            Task type from ModelSelectionState.task_type.
            Expected values: "binary_classification",
            "multiclass_classification", "regression", "clustering",
            "time_series".

        Returns
        -------
        str | None
            A scikit-learn scoring string (e.g. "roc_auc", "f1_weighted",
            "r2"), or None if HPO is not applicable.
        """
        normalized = (task_type or "").lower().strip()
        scoring = _TASK_TYPE_TO_SCORING.get(normalized)

        if scoring is None and normalized not in _TASK_TYPE_TO_SCORING:
            logger.warning(
                "ScoringStrategySelector: unknown task type '%s'. "
                "Defaulting to None (HPO will be skipped). "
                "Register the task type in _TASK_TYPE_TO_SCORING to enable HPO.",
                task_type,
            )

        return scoring

    def is_supported(self, task_type: str) -> bool:
        """
        Return True if hyperparameter optimization is applicable to
        the given task type.

        Parameters
        ----------
        task_type : str
            Task type from ModelSelectionState.task_type.

        Returns
        -------
        bool
            True if HPO is supported (scoring metric is not None).
        """
        return self.get_scoring_metric(task_type) is not None
