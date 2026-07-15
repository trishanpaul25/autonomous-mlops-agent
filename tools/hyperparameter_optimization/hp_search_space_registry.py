"""
Hyperparameter Search Space Registry.

Declarative catalogue of hyperparameter search spaces for every
supported ML algorithm.

Design Principles
-----------------
* Open/Closed: New algorithms are added by calling
  `registry.register(class_path, config)` or by appending to the
  _BUILTIN_SEARCH_CONFIGS dict. No existing code changes required.

* Single Responsibility: This module only knows ABOUT search spaces;
  it never instantiates models or runs CV search.

* Dependency Inversion: The HPOptimizer depends on the
  HPSearchSpaceRegistry abstraction, not on any concrete algorithm.

* No if/else: The registry is a pure dict lookup. Any class_path
  not in the registry is handled gracefully (returned as None so the
  optimizer can skip or warn instead of crash).

Usage
-----
    registry = HPSearchSpaceRegistry()
    config = registry.get(class_path)
    if config is not None:
        # config.search_space, config.strategy, config.n_iter
        pass
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


# ---------------------------------------------------------------------------
# HPSearchConfig — describes one algorithm's search configuration
# ---------------------------------------------------------------------------

@dataclass
class HPSearchConfig:
    """
    Declarative hyperparameter search configuration for a single algorithm.

    Attributes
    ----------
    search_space : dict[str, Any]
        Parameter distributions or grids.
        - For RandomizedSearchCV: use scipy.stats distributions or lists.
        - For GridSearchCV: use lists of discrete values only.
    strategy : Literal["randomized", "grid"]
        Which CV search strategy to use for this algorithm.
        "randomized" is preferred for large search spaces.
        "grid" is used only for small, fully discrete spaces.
    n_iter : int
        Number of parameter settings sampled in RandomizedSearchCV.
        Ignored when strategy == "grid".
    cv : int
        Number of cross-validation folds.
    """

    search_space: dict[str, Any]
    strategy: Literal["randomized", "grid"] = "randomized"
    n_iter: int = 20
    cv: int = 5


# ---------------------------------------------------------------------------
# Built-in search configurations
# ---------------------------------------------------------------------------
# Keys must exactly match the class_path strings stored in
# ModelSelectionState.candidate_models[*]["class_path"].
#
# Extending: add a new key → HPSearchConfig entry below.
# The rest of the pipeline picks it up automatically.
# ---------------------------------------------------------------------------

_BUILTIN_SEARCH_CONFIGS: dict[str, HPSearchConfig] = {

    # -----------------------------------------------------------------------
    # sklearn — Classification
    # -----------------------------------------------------------------------

    "sklearn.linear_model.LogisticRegression": HPSearchConfig(
        strategy="randomized",
        n_iter=20,
        cv=5,
        search_space={
            "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            "penalty": ["l2", "l1"],
            "solver": ["liblinear", "saga"],
            "max_iter": [500, 1000, 2000],
        },
    ),

    "sklearn.linear_model.RidgeClassifier": HPSearchConfig(
        strategy="randomized",
        n_iter=15,
        cv=5,
        search_space={
            "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
            "fit_intercept": [True, False],
        },
    ),

    "sklearn.tree.DecisionTreeClassifier": HPSearchConfig(
        strategy="randomized",
        n_iter=30,
        cv=5,
        search_space={
            "max_depth": [None, 3, 5, 8, 10, 15, 20],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 8],
            "criterion": ["gini", "entropy"],
            "max_features": [None, "sqrt", "log2"],
        },
    ),

    "sklearn.ensemble.RandomForestClassifier": HPSearchConfig(
        strategy="randomized",
        n_iter=30,
        cv=5,
        search_space={
            "n_estimators": [50, 100, 200, 300, 500],
            "max_depth": [None, 5, 10, 15, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
            "bootstrap": [True, False],
        },
    ),

    "sklearn.ensemble.ExtraTreesClassifier": HPSearchConfig(
        strategy="randomized",
        n_iter=25,
        cv=5,
        search_space={
            "n_estimators": [50, 100, 200, 300],
            "max_depth": [None, 5, 10, 15, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
        },
    ),

    "sklearn.ensemble.GradientBoostingClassifier": HPSearchConfig(
        strategy="randomized",
        n_iter=30,
        cv=5,
        search_space={
            "n_estimators": [50, 100, 200, 300],
            "learning_rate": [0.001, 0.01, 0.05, 0.1, 0.2, 0.3],
            "max_depth": [3, 4, 5, 6, 8],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "max_features": ["sqrt", "log2", None],
        },
    ),

    "sklearn.svm.SVC": HPSearchConfig(
        strategy="randomized",
        n_iter=25,
        cv=5,
        search_space={
            "C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "kernel": ["rbf", "linear", "poly", "sigmoid"],
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
            "degree": [2, 3, 4],          # only used by poly kernel
        },
    ),

    "sklearn.neighbors.KNeighborsClassifier": HPSearchConfig(
        strategy="randomized",
        n_iter=20,
        cv=5,
        search_space={
            "n_neighbors": [3, 5, 7, 9, 11, 15, 21],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan", "minkowski"],
            "leaf_size": [10, 20, 30, 40],
        },
    ),

    # -----------------------------------------------------------------------
    # sklearn — Regression
    # -----------------------------------------------------------------------

    "sklearn.linear_model.Ridge": HPSearchConfig(
        strategy="randomized",
        n_iter=20,
        cv=5,
        search_space={
            "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
            "fit_intercept": [True, False],
            "solver": ["auto", "svd", "cholesky", "lsqr", "sag"],
        },
    ),

    "sklearn.linear_model.Lasso": HPSearchConfig(
        strategy="randomized",
        n_iter=20,
        cv=5,
        search_space={
            "alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
            "fit_intercept": [True, False],
            "selection": ["cyclic", "random"],
        },
    ),

    "sklearn.linear_model.ElasticNet": HPSearchConfig(
        strategy="randomized",
        n_iter=25,
        cv=5,
        search_space={
            "alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
            "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
            "fit_intercept": [True, False],
            "selection": ["cyclic", "random"],
        },
    ),

    "sklearn.tree.DecisionTreeRegressor": HPSearchConfig(
        strategy="randomized",
        n_iter=30,
        cv=5,
        search_space={
            "max_depth": [None, 3, 5, 8, 10, 15, 20],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 8],
            "criterion": ["squared_error", "friedman_mse", "absolute_error"],
            "max_features": [None, "sqrt", "log2"],
        },
    ),

    "sklearn.ensemble.RandomForestRegressor": HPSearchConfig(
        strategy="randomized",
        n_iter=30,
        cv=5,
        search_space={
            "n_estimators": [50, 100, 200, 300, 500],
            "max_depth": [None, 5, 10, 15, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
            "bootstrap": [True, False],
        },
    ),

    "sklearn.ensemble.GradientBoostingRegressor": HPSearchConfig(
        strategy="randomized",
        n_iter=30,
        cv=5,
        search_space={
            "n_estimators": [50, 100, 200, 300],
            "learning_rate": [0.001, 0.01, 0.05, 0.1, 0.2, 0.3],
            "max_depth": [3, 4, 5, 6, 8],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        },
    ),

    "sklearn.svm.SVR": HPSearchConfig(
        strategy="randomized",
        n_iter=25,
        cv=5,
        search_space={
            "C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "kernel": ["rbf", "linear", "poly"],
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
            "epsilon": [0.01, 0.05, 0.1, 0.2, 0.5],
        },
    ),

    "sklearn.linear_model.LinearRegression": HPSearchConfig(
        strategy="grid",
        n_iter=4,
        cv=5,
        search_space={
            "fit_intercept": [True, False],
            "positive": [True, False],
        },
    ),

    # -----------------------------------------------------------------------
    # XGBoost
    # -----------------------------------------------------------------------

    "xgboost.XGBClassifier": HPSearchConfig(
        strategy="randomized",
        n_iter=30,
        cv=5,
        search_space={
            "n_estimators": [50, 100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
            "max_depth": [3, 4, 5, 6, 8, 10],
            "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "min_child_weight": [1, 3, 5, 7],
            "gamma": [0, 0.1, 0.2, 0.3, 0.5],
            "reg_alpha": [0, 0.01, 0.1, 1.0],
            "reg_lambda": [0.5, 1.0, 2.0, 5.0],
        },
    ),

    "xgboost.XGBRegressor": HPSearchConfig(
        strategy="randomized",
        n_iter=30,
        cv=5,
        search_space={
            "n_estimators": [50, 100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
            "max_depth": [3, 4, 5, 6, 8, 10],
            "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "min_child_weight": [1, 3, 5, 7],
            "gamma": [0, 0.1, 0.2, 0.3, 0.5],
            "reg_alpha": [0, 0.01, 0.1, 1.0],
            "reg_lambda": [0.5, 1.0, 2.0, 5.0],
        },
    ),

    # -----------------------------------------------------------------------
    # LightGBM
    # -----------------------------------------------------------------------

    "lightgbm.LGBMClassifier": HPSearchConfig(
        strategy="randomized",
        n_iter=30,
        cv=5,
        search_space={
            "n_estimators": [50, 100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
            "max_depth": [-1, 5, 8, 10, 15],
            "num_leaves": [20, 31, 50, 70, 100],
            "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "min_child_samples": [5, 10, 20, 50],
            "reg_alpha": [0, 0.01, 0.1, 1.0],
            "reg_lambda": [0, 0.01, 0.1, 1.0],
        },
    ),

    "lightgbm.LGBMRegressor": HPSearchConfig(
        strategy="randomized",
        n_iter=30,
        cv=5,
        search_space={
            "n_estimators": [50, 100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
            "max_depth": [-1, 5, 8, 10, 15],
            "num_leaves": [20, 31, 50, 70, 100],
            "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "min_child_samples": [5, 10, 20, 50],
            "reg_alpha": [0, 0.01, 0.1, 1.0],
            "reg_lambda": [0, 0.01, 0.1, 1.0],
        },
    ),

    # -----------------------------------------------------------------------
    # CatBoost
    # -----------------------------------------------------------------------

    "catboost.CatBoostClassifier": HPSearchConfig(
        strategy="randomized",
        n_iter=20,
        cv=5,
        search_space={
            "iterations": [50, 100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
            "depth": [4, 5, 6, 7, 8, 10],
            "l2_leaf_reg": [1, 3, 5, 7, 9],
            "border_count": [32, 64, 128, 254],
        },
    ),

    "catboost.CatBoostRegressor": HPSearchConfig(
        strategy="randomized",
        n_iter=20,
        cv=5,
        search_space={
            "iterations": [50, 100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
            "depth": [4, 5, 6, 7, 8, 10],
            "l2_leaf_reg": [1, 3, 5, 7, 9],
            "border_count": [32, 64, 128, 254],
        },
    ),
}


class HPSearchSpaceRegistry:
    """
    Registry that maps ML algorithm class paths to their hyperparameter
    search configurations.

    Open/Closed principle: new algorithms are registered by calling
    `registry.register()` or by appending to _BUILTIN_SEARCH_CONFIGS.
    No existing code needs to change.

    Methods
    -------
    get(class_path) -> HPSearchConfig | None
        Returns the search config for the given class path.
        Returns None if the algorithm is not registered (caller skips HPO).
    register(class_path, config)
        Register or override a search config at runtime.
    list_supported() -> list[str]
        Returns all registered class paths.
    """

    def __init__(self) -> None:
        # Start from built-in configs; callers can extend at runtime
        self._registry: dict[str, HPSearchConfig] = dict(_BUILTIN_SEARCH_CONFIGS)

    def get(self, class_path: str) -> HPSearchConfig | None:
        """
        Return the search configuration for a class path, or None if not found.

        A return value of None means the optimizer should skip this model
        (not crash). This is the correct behaviour for unknown algorithms.

        Parameters
        ----------
        class_path : str
            Fully-qualified class path of the estimator.

        Returns
        -------
        HPSearchConfig | None
            The registered config, or None if not registered.
        """
        return self._registry.get(class_path)

    def register(
        self,
        class_path: str,
        config: HPSearchConfig,
    ) -> None:
        """
        Register or override a search configuration for a class path.

        Allows external code to extend the registry at runtime without
        modifying this module.

        Parameters
        ----------
        class_path : str
            Fully-qualified class path of the estimator.
        config : HPSearchConfig
            Search configuration to associate with the class path.
        """
        self._registry[class_path] = config

    def list_supported(self) -> list[str]:
        """
        Return all class paths currently registered in the registry.

        Returns
        -------
        list[str]
            All registered class path strings.
        """
        return list(self._registry.keys())
