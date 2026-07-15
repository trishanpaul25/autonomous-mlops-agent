"""
Hyperparameter Optimizer.

Core engine that executes CV search for each candidate model.

Design Principles
-----------------
* Error isolation: each model search is wrapped in its own try/except
  block. One model failure never prevents remaining models from being
  optimized.
* Timing: wall-clock optimization duration is recorded per model using
  time.perf_counter() for sub-second precision.
* Unique identifiers: each optimized model receives a unique
  `model_identifier` so downstream agents can retrieve the exact
  refitted estimator from HyperparameterOptimizationState.
* Strategy dispatch: the optimizer reads HPSearchConfig.strategy and
  selects the appropriate sklearn search class (RandomizedSearchCV or
  GridSearchCV). Optuna is supported as an optional strategy with
  graceful fallback.
* Fresh estimators: the optimizer imports a fresh unfitted estimator
  via importlib for each model. The already-fitted training object is
  NOT reused — CV search requires unfitted estimators.

Error Handling Matrix
---------------------
┌─────────────────────────────────────────┬──────────────────────────────────┐
│ Scenario                                │ Behaviour                        │
├─────────────────────────────────────────┼──────────────────────────────────┤
│ class_path not in registry              │ Skip, log warning, continue      │
│ Model import/instantiation fails        │ Record "failed", continue        │
│ CV search raises ConvergenceWarning     │ Suppress, record, continue       │
│ CV search raises any other exception    │ Record "failed", continue        │
│ Optuna not installed (strategy=optuna)  │ Fallback to RandomizedSearchCV   │
│ All models fail                         │ Return list of failed records    │
│ Task type not supported (clustering)    │ Return empty list immediately    │
└─────────────────────────────────────────┴──────────────────────────────────┘
"""

from __future__ import annotations

import importlib
import re
import time
import uuid
import warnings
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from tools.hyperparameter_optimization.hp_search_space_registry import (
    HPSearchConfig,
    HPSearchSpaceRegistry,
)
from tools.hyperparameter_optimization.scoring_strategy_selector import (
    ScoringStrategySelector,
)
from utils.logger import logger

if TYPE_CHECKING:
    from state.pipeline_state import PipelineState


# ---------------------------------------------------------------------------
# Internal result dataclass
# ---------------------------------------------------------------------------

@dataclass
class HPOptimizationResult:
    """
    Internal result produced for each optimization attempt.

    This intermediate structure is used within HPOptimizer.
    HyperparameterOptimizationTool converts these to
    OptimizedModelRecord objects and stores them in
    HyperparameterOptimizationState.
    """

    model_name: str
    class_path: str
    status: str                      # "optimized" | "failed" | "skipped"
    best_parameters: dict[str, Any]
    best_score: float = 0.0
    optimization_time_seconds: float = 0.0
    model_identifier: str = ""
    fitted_model: Any = None         # Refitted estimator with best params
    strategy_used: str = ""
    scoring_metric: str = ""
    notes: str = ""
    error: str = ""


# ---------------------------------------------------------------------------
# HPOptimizer
# ---------------------------------------------------------------------------

class HPOptimizer:
    """
    Runs hyperparameter search for all candidate models from
    ModelTrainingState.

    Constructor wires up:
      - HPSearchSpaceRegistry   for declarative search space lookup
      - ScoringStrategySelector for task-type → scoring metric mapping

    Methods
    -------
    optimize_all(state) -> list[HPOptimizationResult]
        Iterates over all trained models and optimizes each one.
    """

    def __init__(
        self,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        """
        Parameters
        ----------
        random_state : int
            Random seed for reproducibility in CV searches. Default 42.
        n_jobs : int
            Number of parallel jobs for CV search. -1 uses all CPUs.
        """
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.registry = HPSearchSpaceRegistry()
        self.scorer_selector = ScoringStrategySelector()

    def optimize_all(
        self,
        state: "PipelineState",
    ) -> list[HPOptimizationResult]:
        """
        Execute hyperparameter optimization for every trained model.

        Execution flow per model
        ------------------------
        1. Look up search config in HPSearchSpaceRegistry.
        2. If not found, skip model (log warning).
        3. Instantiate a fresh unfitted estimator via importlib.
        4. Run CV search (RandomizedSearchCV or GridSearchCV).
        5. Record result with best params, best score, timing.
        6. Store refitted estimator in the result.
        7. Continue to next model regardless of outcome.

        Parameters
        ----------
        state : PipelineState
            Shared pipeline state after Model Training. Must have:
            - state.model_training.trained_models (list of dicts)
            - state.model_training.X_train / y_train
            - state.model_selection.task_type

        Returns
        -------
        list[HPOptimizationResult]
            One result per trained model, in the same order as
            state.model_training.trained_models.
        """
        task_type: str = (state.model_selection.task_type or "").lower()
        scoring_metric: str | None = self.scorer_selector.get_scoring_metric(
            task_type
        )

        mt = state.model_training
        X_train = np.array(mt.X_train)
        y_train = np.array(mt.y_train)

        results: list[HPOptimizationResult] = []

        for record in mt.trained_models:
            model_name: str = record.get("model_name", "Unknown")
            class_path: str = record.get("class_path", "")

            logger.info(
                "[HPO] %s: optimization started.",
                model_name,
            )
            state.logs.append(f"[HPO] {model_name}: optimization started.")

            result = self._optimize_single(
                model_name=model_name,
                class_path=class_path,
                X_train=X_train,
                y_train=y_train,
                scoring_metric=scoring_metric,
                task_type=task_type,
            )

            results.append(result)

            if result.status == "optimized":
                logger.info(
                    "[HPO] %s: optimization completed in %.2fs. "
                    "Best score (%.4f) with params: %s",
                    model_name,
                    result.optimization_time_seconds,
                    result.best_score,
                    result.best_parameters,
                )
                state.logs.append(
                    f"[HPO] {model_name}: optimization completed "
                    f"({result.optimization_time_seconds:.2f}s). "
                    f"Best score: {result.best_score:.4f}."
                )
            elif result.status == "skipped":
                logger.warning(
                    "[HPO] %s: skipped — %s",
                    model_name,
                    result.notes,
                )
                state.logs.append(
                    f"[HPO] {model_name}: skipped — {result.notes}"
                )
            else:
                logger.warning(
                    "[HPO] %s: optimization failed — %s",
                    model_name,
                    result.error,
                )
                state.logs.append(
                    f"[HPO] {model_name}: optimization failed — {result.error}"
                )

        return results

    def _optimize_single(
        self,
        model_name: str,
        class_path: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        scoring_metric: str | None,
        task_type: str,
    ) -> HPOptimizationResult:
        """
        Attempt hyperparameter optimization for a single model.

        All exceptions are caught and encoded into the HPOptimizationResult
        so the caller's loop is never interrupted.

        Parameters
        ----------
        model_name : str
            Human-readable model name for logging.
        class_path : str
            Fully-qualified class path of the estimator.
        X_train : np.ndarray
            Training feature matrix.
        y_train : np.ndarray
            Training target vector.
        scoring_metric : str | None
            scikit-learn scoring string. None means task type is unsupported.
        task_type : str
            The ML task type (for context in notes).

        Returns
        -------
        HPOptimizationResult
            Outcome of the optimization attempt.
        """
        # ------------------------------------------------------------------
        # Guard: task type not supported for standard CV scoring
        # ------------------------------------------------------------------
        if scoring_metric is None:
            return HPOptimizationResult(
                model_name=model_name,
                class_path=class_path,
                status="skipped",
                best_parameters={},
                notes=(
                    f"Task type '{task_type}' does not support standard "
                    "cross-validation scoring. HPO skipped."
                ),
            )

        # ------------------------------------------------------------------
        # Guard: empty class_path
        # ------------------------------------------------------------------
        if not class_path:
            return HPOptimizationResult(
                model_name=model_name,
                class_path=class_path,
                status="skipped",
                best_parameters={},
                notes="No class_path available — model was skipped.",
                error="class_path is empty",
            )

        # ------------------------------------------------------------------
        # Guard: no search space registered for this model
        # ------------------------------------------------------------------
        search_config: HPSearchConfig | None = self.registry.get(class_path)

        if search_config is None:
            return HPOptimizationResult(
                model_name=model_name,
                class_path=class_path,
                status="skipped",
                best_parameters={},
                notes=(
                    f"No search space registered for '{class_path}'. "
                    "Register a HPSearchConfig to enable HPO for this model."
                ),
            )

        start_time = time.perf_counter()

        # ------------------------------------------------------------------
        # Instantiate a fresh unfitted estimator
        # ------------------------------------------------------------------
        try:
            estimator = self._import_estimator(class_path)
        except (ImportError, AttributeError, TypeError) as exc:
            elapsed = time.perf_counter() - start_time
            error_msg = str(exc)
            return HPOptimizationResult(
                model_name=model_name,
                class_path=class_path,
                status="failed",
                best_parameters={},
                optimization_time_seconds=round(elapsed, 4),
                notes=f"Estimator import/instantiation failed: {error_msg}",
                error=error_msg,
            )

        # ------------------------------------------------------------------
        # Build and run the CV search
        # ------------------------------------------------------------------
        try:
            search = self._build_search(
                estimator=estimator,
                search_config=search_config,
                scoring=scoring_metric,
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                search.fit(X_train, y_train)

        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            error_msg = str(exc)
            return HPOptimizationResult(
                model_name=model_name,
                class_path=class_path,
                status="failed",
                best_parameters={},
                optimization_time_seconds=round(elapsed, 4),
                notes=f"CV search failed: {error_msg}",
                strategy_used=search_config.strategy,
                scoring_metric=scoring_metric,
                error=error_msg,
            )

        elapsed = time.perf_counter() - start_time
        best_params: dict[str, Any] = dict(search.best_params_)
        best_score: float = float(search.best_score_)
        strategy_name = (
            "RandomizedSearchCV"
            if search_config.strategy == "randomized"
            else "GridSearchCV"
        )
        model_identifier = self._make_identifier(model_name)

        return HPOptimizationResult(
            model_name=model_name,
            class_path=class_path,
            status="optimized",
            best_parameters=best_params,
            best_score=round(best_score, 6),
            optimization_time_seconds=round(elapsed, 4),
            model_identifier=model_identifier,
            fitted_model=search.best_estimator_,
            strategy_used=strategy_name,
            scoring_metric=scoring_metric,
            notes=(
                f"Optimized with {strategy_name} in {elapsed:.2f}s. "
                f"Best {scoring_metric}: {best_score:.4f}. "
                f"Identifier: {model_identifier}"
            ),
        )

    def _build_search(
        self,
        estimator: Any,
        search_config: HPSearchConfig,
        scoring: str,
    ) -> RandomizedSearchCV | GridSearchCV:
        """
        Construct the appropriate sklearn CV search object.

        Selects RandomizedSearchCV or GridSearchCV based on
        search_config.strategy. Optuna strategy falls back to
        RandomizedSearchCV with a warning if the optuna package
        is not installed.

        Parameters
        ----------
        estimator : Any
            An unfitted sklearn-compatible estimator.
        search_config : HPSearchConfig
            Configuration containing strategy, search_space, n_iter, cv.
        scoring : str
            scikit-learn scoring string.

        Returns
        -------
        RandomizedSearchCV | GridSearchCV
            Configured but unfitted CV search object.
        """
        if search_config.strategy == "grid":
            return GridSearchCV(
                estimator=estimator,
                param_grid=search_config.search_space,
                scoring=scoring,
                cv=search_config.cv,
                n_jobs=self.n_jobs,
                refit=True,
                error_score="raise",
            )

        # strategy == "randomized" (default) or "optuna" with fallback
        if search_config.strategy == "optuna":
            try:
                import optuna  # noqa: F401 — only import to check availability
                logger.info(
                    "HPOptimizer: Optuna is installed but Optuna integration "
                    "is not yet wired. Falling back to RandomizedSearchCV."
                )
            except ImportError:
                logger.warning(
                    "HPOptimizer: strategy 'optuna' requested but Optuna is "
                    "not installed. Falling back to RandomizedSearchCV."
                )

        return RandomizedSearchCV(
            estimator=estimator,
            param_distributions=search_config.search_space,
            n_iter=search_config.n_iter,
            scoring=scoring,
            cv=search_config.cv,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            refit=True,
            error_score="raise",
        )

    @staticmethod
    def _import_estimator(class_path: str) -> Any:
        """
        Dynamically import and instantiate a fresh unfitted estimator.

        Uses importlib so no if/else is required for different libraries.
        The estimator is instantiated with no extra kwargs — the CV search
        will control all hyperparameters through its param grid.

        Parameters
        ----------
        class_path : str
            Fully-qualified Python class path.

        Returns
        -------
        Any
            An unfitted estimator instance.

        Raises
        ------
        ImportError
            If the module cannot be imported.
        AttributeError
            If the class is not found in the module.
        TypeError
            If the constructor rejects the default kwargs.
        """
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)

        # Inject silent defaults for known verbose libraries
        silent_kwargs: dict[str, Any] = {}
        if "xgboost" in class_path.lower():
            silent_kwargs["verbosity"] = 0
        elif "lightgbm" in class_path.lower():
            silent_kwargs["verbosity"] = -1
        elif "catboost" in class_path.lower():
            silent_kwargs["verbose"] = 0

        return cls(**silent_kwargs)

    @staticmethod
    def _make_identifier(model_name: str) -> str:
        """
        Produce a unique, filesystem-safe identifier for an optimized model.

        Format: hpo_{sanitized_model_name}_{8-char hex suffix}

        Example: "hpo_random_forest_classifier_a3f1b2c0"
        """
        sanitized = re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")
        suffix = uuid.uuid4().hex[:8]
        return f"hpo_{sanitized}_{suffix}"
