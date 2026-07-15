"""
Model Instantiator for the Model Training Agent.

Dynamically imports and constructs any ML estimator from its
fully-qualified class path string using Python's importlib.

Design Principles
-----------------
* Zero if/else for model type: any class_path string from the
  ModelRegistry is automatically importable without code changes.
* Open/Closed: new algorithms are supported the moment they are
  added to the ModelRegistry and installed in the environment.
* DEFAULT_KWARGS provides sensible default constructor arguments
  (random_state, n_jobs, verbosity) without hard-coding per-model
  branches. Adding a new default is a single dict entry.

Usage
-----
    instantiator = ModelInstantiator()
    model = instantiator.instantiate("sklearn.ensemble.RandomForestClassifier")
    # Returns a RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
"""

from __future__ import annotations
import importlib
from typing import Any
from utils.logger import logger
_DEFAULT_KWARGS: dict[str, dict[str, Any]] = {
    "sklearn.linear_model.LogisticRegression": {
        "random_state": 42,
        "max_iter": 1000,
        "n_jobs": -1,
    },
    "sklearn.linear_model.RidgeClassifier": {
        "random_state": 42,
    },
    "sklearn.tree.DecisionTreeClassifier": {
        "random_state": 42,
    },
    "sklearn.ensemble.RandomForestClassifier": {
        "n_estimators": 100,
        "random_state": 42,
        "n_jobs": -1,
    },
    "sklearn.ensemble.GradientBoostingClassifier": {
        "n_estimators": 100,
        "random_state": 42,
    },
    "sklearn.ensemble.ExtraTreesClassifier": {
        "n_estimators": 100,
        "random_state": 42,
        "n_jobs": -1,
    },
    "sklearn.svm.SVC": {
        "random_state": 42,
        "probability": True,  # enables predict_proba for downstream metrics
    },
    "sklearn.neighbors.KNeighborsClassifier": {
        "n_jobs": -1,
    },
    "sklearn.linear_model.LinearRegression": {
        "n_jobs": -1,
    },
    "sklearn.linear_model.Ridge": {
        "random_state": 42,
    },
    "sklearn.linear_model.Lasso": {
        "random_state": 42,
        "max_iter": 5000,
    },
    "sklearn.linear_model.ElasticNet": {
        "random_state": 42,
        "max_iter": 5000,
    },
    "sklearn.tree.DecisionTreeRegressor": {
        "random_state": 42,
    },
    "sklearn.ensemble.RandomForestRegressor": {
        "n_estimators": 100,
        "random_state": 42,
        "n_jobs": -1,
    },
    "sklearn.ensemble.GradientBoostingRegressor": {
        "n_estimators": 100,
        "random_state": 42,
    },
    "sklearn.svm.SVR": {},
    "sklearn.cluster.KMeans": {
        "n_clusters": 5,
        "random_state": 42,
        "n_init": "auto",
    },
    "sklearn.cluster.DBSCAN": {},
    "sklearn.cluster.AgglomerativeClustering": {
        "n_clusters": 5,
    },
    "sklearn.mixture.GaussianMixture": {
        "n_components": 5,
        "random_state": 42,
    },
    "xgboost.XGBClassifier": {
        "n_estimators": 100,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
        "eval_metric": "logloss",
    },
    "xgboost.XGBRegressor": {
        "n_estimators": 100,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    },
    "lightgbm.LGBMClassifier": {
        "n_estimators": 100,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    },
    "lightgbm.LGBMRegressor": {
        "n_estimators": 100,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    },
    "catboost.CatBoostClassifier": {
        "iterations": 100,
        "random_state": 42,
        "verbose": 0,
    },
    "catboost.CatBoostRegressor": {
        "iterations": 100,
        "random_state": 42,
        "verbose": 0,
    },
}
class ModelInstantiator:
    """
    Dynamically imports and constructs ML estimator objects from
    fully-qualified class path strings.

    No if/else or match/case logic for model types. Any class that
    is importable from `class_path` and callable with keyword
    arguments from _DEFAULT_KWARGS (or no arguments at all) is
    automatically supported.
    """

    def instantiate(
        self,
        class_path: str,
        extra_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """
        Import the class at `class_path` and return a new instance.

        Default kwargs are looked up from _DEFAULT_KWARGS. Any
        `extra_kwargs` are merged on top, allowing callers to override
        individual parameters without replacing the defaults entirely.

        Parameters
        ----------
        class_path : str
            Fully-qualified Python class path.
            Example: "sklearn.ensemble.RandomForestClassifier"
        extra_kwargs : dict | None
            Optional overrides merged on top of DEFAULT_KWARGS.

        Returns
        -------
        Any
            An unfitted estimator instance.

        Raises
        ------
        ImportError
            If the module cannot be imported (library not installed).
        AttributeError
            If the class is not found in the module.
        TypeError
            If the constructor rejects the supplied kwargs.
        """
        module_path, class_name = class_path.rsplit(".", 1)

        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            raise ImportError(
                f"ModelInstantiator: cannot import '{module_path}'. "
                f"Ensure the library is installed. Original error: {e}"
            ) from e

        try:
            cls = getattr(module, class_name)
        except AttributeError as e:
            raise AttributeError(
                f"ModelInstantiator: class '{class_name}' not found in "
                f"module '{module_path}'. Original error: {e}"
            ) from e

        # Merge default kwargs with any caller-supplied overrides
        kwargs: dict[str, Any] = dict(_DEFAULT_KWARGS.get(class_path, {}))
        if extra_kwargs:
            kwargs.update(extra_kwargs)

        try:
            instance = cls(**kwargs)
        except TypeError as e:
            raise TypeError(
                f"ModelInstantiator: failed to instantiate '{class_path}' "
                f"with kwargs {kwargs}. Original error: {e}"
            ) from e

        logger.debug(
            "ModelInstantiator: instantiated '%s' with kwargs %s",
            class_path,
            kwargs,
        )

        return instance

    def register_defaults(
        self,
        class_path: str,
        kwargs: dict[str, Any],
    ) -> None:
        """
        Register or override default constructor kwargs for a class path.

        Allows external code to configure defaults without modifying
        this module.

        Parameters
        ----------
        class_path : str
            Fully-qualified class path to register defaults for.
        kwargs : dict
            Default constructor keyword arguments.
        """
        _DEFAULT_KWARGS[class_path] = kwargs
        logger.debug(
            "ModelInstantiator: registered defaults for '%s': %s",
            class_path,
            kwargs,
        )
