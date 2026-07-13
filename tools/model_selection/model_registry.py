"""
Model Registry — declarative catalogue of ML algorithms.

Design Principles
-----------------
* Open/Closed: New algorithms are added by calling `registry.register(descriptor)`
  or by appending to the _BUILTIN_MODELS list. No existing code changes required.
* Single Responsibility: This module only knows ABOUT algorithms; it never
  instantiates, trains, or evaluates them.
* Dependency Inversion: The ModelSelectionAgent depends on the ModelRegistry
  abstraction, not on any concrete algorithm implementation.

Usage
-----
    from tools.model_selection.model_registry import ModelRegistry, TaskType

    registry = ModelRegistry()
    candidates = registry.get_candidates(TaskType.BINARY_CLASSIFICATION, profile)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tools.model_selection.dataset_profiler import DatasetProfile
class TaskType(str, Enum):
    """
    Supported ML task types.

    Using a str-based Enum keeps values serialisable (JSON-safe) and
    compatible with the Literal type used in the Pydantic schema.
    """

    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    TIME_SERIES = "time_series"

    @classmethod
    def from_validation_type(cls, problem_type: str | None) -> "TaskType":
        """
        Converts the coarse problem_type stored by the Validation Agent
        ('classification', 'regression', 'clustering') into a TaskType.

        The model selection agent will refine binary vs. multi-class
        classification using the actual class distribution from the dataset.
        """
        if problem_type is None:
            return cls.BINARY_CLASSIFICATION

        mapping = {
            "classification": cls.BINARY_CLASSIFICATION,
            "regression": cls.REGRESSION,
            "clustering": cls.CLUSTERING,
            "time_series": cls.TIME_SERIES,
        }
        return mapping.get(problem_type.lower(), cls.BINARY_CLASSIFICATION)
class DatasetSizeCategory(str, Enum):
    TINY = "tiny"          # < 500 rows
    SMALL = "small"        # 500 – 5,000 rows
    MEDIUM = "medium"      # 5,000 – 100,000 rows
    LARGE = "large"        # > 100,000 rows


def categorise_dataset_size(num_rows: int) -> DatasetSizeCategory:
    """Maps row count to a size category used for filtering recommendations."""
    if num_rows < 500:
        return DatasetSizeCategory.TINY
    if num_rows < 5_000:
        return DatasetSizeCategory.SMALL
    if num_rows < 100_000:
        return DatasetSizeCategory.MEDIUM
    return DatasetSizeCategory.LARGE
@dataclass
class ModelDescriptor:
    """
    Declarative description of a single ML algorithm.

    Every attribute is metadata ABOUT the algorithm — no code is ever
    executed from this class.
    """

    # Unique identifier; used as a key in the registry
    name: str

    # Python package (e.g. "sklearn", "xgboost", "lightgbm")
    library: str

    # Fully-qualified importable class path
    class_path: str

    # Task types this model can handle
    task_types: list[TaskType]

    # Ordinal score 1 (low) – 5 (high) for interpretability
    interpretability: int  # 1 = black box, 5 = fully transparent

    # Ordinal score 1 (low) – 5 (high) for computational scalability
    scalability: int

    # Minimum recommended rows for this model to perform reliably
    min_rows: int = 50

    # Maximum rows before performance degrades significantly (0 = unlimited)
    max_rows: int = 0

    # Whether this model natively handles categorical features
    handles_categorical: bool = False

    # Whether this model natively handles missing values
    handles_missing: bool = False

    # Whether this model supports sample_weight or class_weight for imbalance
    handles_imbalance: bool = False

    # Whether this model scales well to high-dimensional feature spaces
    handles_high_dimensions: bool = False

    # Descriptive tags for filtering and display
    tags: list[str] = field(default_factory=list)

    # Short human-readable description
    description: str = ""

    # Base suitability score before dataset-specific adjustments (0.0 – 1.0)
    base_score: float = 0.7

    def supports_task(self, task_type: TaskType) -> bool:
        """Returns True if this model supports the given task type."""
        return task_type in self.task_types

    def is_feasible_for(self, num_rows: int) -> bool:
        """Returns True if row count is within the model's operating range."""
        if num_rows < self.min_rows:
            return False
        if self.max_rows > 0 and num_rows > self.max_rows:
            return False
        return True

_BUILTIN_MODELS: list[ModelDescriptor] = [
    ModelDescriptor(
        name="Logistic Regression",
        library="sklearn",
        class_path="sklearn.linear_model.LogisticRegression",
        task_types=[TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION],
        interpretability=5,
        scalability=4,
        min_rows=50,
        handles_imbalance=True,
        handles_high_dimensions=True,
        tags=["linear", "parametric", "interpretable", "baseline"],
        description=(
            "Fast, interpretable linear model. Excellent baseline for "
            "classification tasks. Requires feature scaling."
        ),
        base_score=0.72,
    ),

    ModelDescriptor(
        name="Ridge Classifier",
        library="sklearn",
        class_path="sklearn.linear_model.RidgeClassifier",
        task_types=[TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION],
        interpretability=5,
        scalability=5,
        min_rows=50,
        handles_high_dimensions=True,
        tags=["linear", "regularized", "parametric", "baseline"],
        description=(
            "Regularized linear classifier. Very fast on large high-dimensional "
            "datasets. Good when multicollinearity is present."
        ),
        base_score=0.68,
    ),

    ModelDescriptor(
        name="Decision Tree Classifier",
        library="sklearn",
        class_path="sklearn.tree.DecisionTreeClassifier",
        task_types=[TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION],
        interpretability=5,
        scalability=3,
        min_rows=50,
        handles_categorical=False,
        handles_imbalance=True,
        tags=["tree-based", "interpretable", "non-parametric"],
        description=(
            "Fully interpretable tree model. Prone to overfitting without "
            "pruning. Useful as a baseline or for rule extraction."
        ),
        base_score=0.60,
    ),

    ModelDescriptor(
        name="Random Forest Classifier",
        library="sklearn",
        class_path="sklearn.ensemble.RandomForestClassifier",
        task_types=[TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION],
        interpretability=3,
        scalability=3,
        min_rows=200,
        handles_imbalance=True,
        tags=["ensemble", "tree-based", "bagging", "non-parametric", "robust"],
        description=(
            "Robust ensemble of decision trees. Handles non-linearity, "
            "feature interactions, and moderate class imbalance well."
        ),
        base_score=0.85,
    ),

    ModelDescriptor(
        name="Gradient Boosting Classifier",
        library="sklearn",
        class_path="sklearn.ensemble.GradientBoostingClassifier",
        task_types=[TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION],
        interpretability=2,
        scalability=2,
        min_rows=300,
        max_rows=50_000,
        handles_imbalance=True,
        tags=["ensemble", "tree-based", "boosting", "non-parametric"],
        description=(
            "Sequential boosting ensemble. High accuracy on tabular data "
            "but slower to train than LightGBM on large datasets."
        ),
        base_score=0.82,
    ),

    ModelDescriptor(
        name="XGBoost Classifier",
        library="xgboost",
        class_path="xgboost.XGBClassifier",
        task_types=[TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION],
        interpretability=2,
        scalability=4,
        min_rows=300,
        handles_missing=True,
        handles_imbalance=True,
        tags=["ensemble", "tree-based", "boosting", "scalable", "gpu-ready"],
        description=(
            "High-performance gradient boosting. Natively handles missing "
            "values and scales to large datasets with GPU support."
        ),
        base_score=0.88,
    ),

    ModelDescriptor(
        name="LightGBM Classifier",
        library="lightgbm",
        class_path="lightgbm.LGBMClassifier",
        task_types=[TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION],
        interpretability=2,
        scalability=5,
        min_rows=300,
        handles_categorical=True,
        handles_missing=True,
        handles_imbalance=True,
        handles_high_dimensions=True,
        tags=["ensemble", "tree-based", "boosting", "scalable", "categorical-native"],
        description=(
            "Extremely fast gradient boosting. Best choice for large datasets "
            "with many categorical features."
        ),
        base_score=0.90,
    ),

    ModelDescriptor(
        name="CatBoost Classifier",
        library="catboost",
        class_path="catboost.CatBoostClassifier",
        task_types=[TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION],
        interpretability=2,
        scalability=4,
        min_rows=300,
        handles_categorical=True,
        handles_missing=True,
        handles_imbalance=True,
        tags=["ensemble", "tree-based", "boosting", "categorical-native"],
        description=(
            "Gradient boosting with native categorical encoding. "
            "Minimal preprocessing required."
        ),
        base_score=0.87,
    ),

    ModelDescriptor(
        name="Support Vector Classifier",
        library="sklearn",
        class_path="sklearn.svm.SVC",
        task_types=[TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION],
        interpretability=2,
        scalability=1,
        min_rows=50,
        max_rows=10_000,
        handles_imbalance=True,
        tags=["kernel-based", "non-parametric", "margin-based"],
        description=(
            "Powerful for small-to-medium non-linear datasets. Computationally "
            "expensive (O(n²) memory); not recommended above 10k rows."
        ),
        base_score=0.78,
    ),

    ModelDescriptor(
        name="K-Nearest Neighbors Classifier",
        library="sklearn",
        class_path="sklearn.neighbors.KNeighborsClassifier",
        task_types=[TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION],
        interpretability=4,
        scalability=1,
        min_rows=50,
        max_rows=20_000,
        tags=["instance-based", "lazy-learner", "non-parametric"],
        description=(
            "Simple instance-based learner. Sensitive to feature scaling "
            "and suffers from the curse of dimensionality."
        ),
        base_score=0.60,
    ),

    ModelDescriptor(
        name="Extra Trees Classifier",
        library="sklearn",
        class_path="sklearn.ensemble.ExtraTreesClassifier",
        task_types=[TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION],
        interpretability=3,
        scalability=4,
        min_rows=200,
        handles_imbalance=True,
        tags=["ensemble", "tree-based", "bagging", "fast"],
        description=(
            "Extremely randomized trees. Faster than Random Forest with "
            "comparable accuracy. Slightly higher bias, lower variance."
        ),
        base_score=0.82,
    ),
    ModelDescriptor(
        name="Linear Regression",
        library="sklearn",
        class_path="sklearn.linear_model.LinearRegression",
        task_types=[TaskType.REGRESSION],
        interpretability=5,
        scalability=5,
        min_rows=20,
        handles_high_dimensions=True,
        tags=["linear", "parametric", "interpretable", "baseline"],
        description=(
            "OLS baseline regression. Interpretable and fast. "
            "Sensitive to outliers and multicollinearity."
        ),
        base_score=0.65,
    ),

    ModelDescriptor(
        name="Ridge Regression",
        library="sklearn",
        class_path="sklearn.linear_model.Ridge",
        task_types=[TaskType.REGRESSION],
        interpretability=5,
        scalability=5,
        min_rows=20,
        handles_high_dimensions=True,
        tags=["linear", "regularized", "parametric", "L2"],
        description=(
            "L2-regularized linear regression. Recommended over plain "
            "OLS when multicollinearity is present."
        ),
        base_score=0.70,
    ),

    ModelDescriptor(
        name="Lasso Regression",
        library="sklearn",
        class_path="sklearn.linear_model.Lasso",
        task_types=[TaskType.REGRESSION],
        interpretability=5,
        scalability=5,
        min_rows=20,
        handles_high_dimensions=True,
        tags=["linear", "regularized", "parametric", "L1", "feature-selection"],
        description=(
            "L1-regularized regression with automatic feature selection. "
            "Useful when many features are expected to be irrelevant."
        ),
        base_score=0.72,
    ),

    ModelDescriptor(
        name="ElasticNet",
        library="sklearn",
        class_path="sklearn.linear_model.ElasticNet",
        task_types=[TaskType.REGRESSION],
        interpretability=4,
        scalability=5,
        min_rows=20,
        handles_high_dimensions=True,
        tags=["linear", "regularized", "parametric", "L1", "L2"],
        description=(
            "Combined L1+L2 regularization. Balances feature selection "
            "(Lasso) with coefficient stability (Ridge)."
        ),
        base_score=0.73,
    ),

    ModelDescriptor(
        name="Random Forest Regressor",
        library="sklearn",
        class_path="sklearn.ensemble.RandomForestRegressor",
        task_types=[TaskType.REGRESSION],
        interpretability=3,
        scalability=3,
        min_rows=200,
        tags=["ensemble", "tree-based", "bagging", "non-parametric", "robust"],
        description=(
            "Ensemble of decision trees for regression. Robust to outliers "
            "and non-linear relationships. Good general-purpose choice."
        ),
        base_score=0.85,
    ),

    ModelDescriptor(
        name="Gradient Boosting Regressor",
        library="sklearn",
        class_path="sklearn.ensemble.GradientBoostingRegressor",
        task_types=[TaskType.REGRESSION],
        interpretability=2,
        scalability=2,
        min_rows=300,
        max_rows=50_000,
        tags=["ensemble", "tree-based", "boosting", "non-parametric"],
        description=(
            "Boosting ensemble for regression. High accuracy on tabular "
            "data but slower than LightGBM on large datasets."
        ),
        base_score=0.82,
    ),

    ModelDescriptor(
        name="XGBoost Regressor",
        library="xgboost",
        class_path="xgboost.XGBRegressor",
        task_types=[TaskType.REGRESSION],
        interpretability=2,
        scalability=4,
        min_rows=300,
        handles_missing=True,
        tags=["ensemble", "tree-based", "boosting", "scalable", "gpu-ready"],
        description=(
            "High-performance gradient boosting for regression. "
            "Natively handles missing values; GPU-accelerated training available."
        ),
        base_score=0.88,
    ),

    ModelDescriptor(
        name="LightGBM Regressor",
        library="lightgbm",
        class_path="lightgbm.LGBMRegressor",
        task_types=[TaskType.REGRESSION],
        interpretability=2,
        scalability=5,
        min_rows=300,
        handles_categorical=True,
        handles_missing=True,
        handles_high_dimensions=True,
        tags=["ensemble", "tree-based", "boosting", "scalable", "categorical-native"],
        description=(
            "Fastest gradient boosting framework. Recommended for large "
            "regression datasets with categorical features."
        ),
        base_score=0.90,
    ),

    ModelDescriptor(
        name="Support Vector Regressor",
        library="sklearn",
        class_path="sklearn.svm.SVR",
        task_types=[TaskType.REGRESSION],
        interpretability=2,
        scalability=1,
        min_rows=50,
        max_rows=10_000,
        tags=["kernel-based", "non-parametric", "margin-based"],
        description=(
            "Kernel-based regression. High accuracy on small non-linear "
            "datasets. Does not scale beyond ~10k rows."
        ),
        base_score=0.75,
    ),

    ModelDescriptor(
        name="Decision Tree Regressor",
        library="sklearn",
        class_path="sklearn.tree.DecisionTreeRegressor",
        task_types=[TaskType.REGRESSION],
        interpretability=5,
        scalability=3,
        min_rows=50,
        tags=["tree-based", "interpretable", "non-parametric", "baseline"],
        description=(
            "Interpretable baseline regression tree. Prone to overfitting "
            "without depth constraints."
        ),
        base_score=0.58,
    ),
    ModelDescriptor(
        name="K-Means",
        library="sklearn",
        class_path="sklearn.cluster.KMeans",
        task_types=[TaskType.CLUSTERING],
        interpretability=4,
        scalability=4,
        min_rows=50,
        tags=["clustering", "centroid-based", "unsupervised"],
        description=(
            "Classic centroid-based clustering. Fast and scalable. "
            "Requires specifying k in advance; assumes spherical clusters."
        ),
        base_score=0.80,
    ),

    ModelDescriptor(
        name="DBSCAN",
        library="sklearn",
        class_path="sklearn.cluster.DBSCAN",
        task_types=[TaskType.CLUSTERING],
        interpretability=3,
        scalability=3,
        min_rows=100,
        max_rows=100_000,
        tags=["clustering", "density-based", "unsupervised", "noise-robust"],
        description=(
            "Density-based clustering that detects arbitrary cluster shapes "
            "and marks noise points. No need to specify k in advance."
        ),
        base_score=0.75,
    ),

    ModelDescriptor(
        name="Agglomerative Clustering",
        library="sklearn",
        class_path="sklearn.cluster.AgglomerativeClustering",
        task_types=[TaskType.CLUSTERING],
        interpretability=4,
        scalability=2,
        min_rows=50,
        max_rows=20_000,
        tags=["clustering", "hierarchical", "unsupervised"],
        description=(
            "Hierarchical bottom-up clustering. Produces a dendrogram; "
            "computationally expensive above 20k rows."
        ),
        base_score=0.70,
    ),

    ModelDescriptor(
        name="Gaussian Mixture Model",
        library="sklearn",
        class_path="sklearn.mixture.GaussianMixture",
        task_types=[TaskType.CLUSTERING],
        interpretability=3,
        scalability=3,
        min_rows=100,
        tags=["clustering", "probabilistic", "unsupervised", "soft-assignment"],
        description=(
            "Probabilistic model that assigns soft cluster memberships. "
            "Supports elliptical cluster shapes unlike K-Means."
        ),
        base_score=0.72,
    ),
    ModelDescriptor(
        name="ARIMA",
        library="statsmodels",
        class_path="statsmodels.tsa.arima.model.ARIMA",
        task_types=[TaskType.TIME_SERIES],
        interpretability=4,
        scalability=2,
        min_rows=50,
        tags=["time-series", "statistical", "univariate"],
        description=(
            "Classic ARIMA model for univariate time series forecasting. "
            "Requires stationarity and manual order selection."
        ),
        base_score=0.70,
    ),

    ModelDescriptor(
        name="Prophet",
        library="prophet",
        class_path="prophet.Prophet",
        task_types=[TaskType.TIME_SERIES],
        interpretability=4,
        scalability=3,
        min_rows=100,
        tags=["time-series", "additive", "seasonal", "trend"],
        description=(
            "Facebook Prophet. Handles seasonality, holidays, and trend "
            "changes automatically. Designed for business forecasting."
        ),
        base_score=0.78,
    ),

    ModelDescriptor(
        name="LightGBM Time Series",
        library="lightgbm",
        class_path="lightgbm.LGBMRegressor",
        task_types=[TaskType.TIME_SERIES],
        interpretability=2,
        scalability=5,
        min_rows=300,
        handles_categorical=True,
        handles_missing=True,
        tags=["time-series", "ml-based", "feature-engineering-required", "scalable"],
        description=(
            "LightGBM used as a multi-step forecaster with lag features. "
            "Highly effective when engineered lag/rolling features are available."
        ),
        base_score=0.82,
    ),
]
def _compute_score(
    descriptor: ModelDescriptor,
    task_type: TaskType,
    profile: "DatasetProfile",
) -> float:
    """
    Adjusts a model's base_score based on the concrete dataset profile.

    Scoring is entirely additive/subtractive on top of the base score.
    Each rule is independent and documented so maintainers can reason
    about the outcome.
    """
    score: float = descriptor.base_score

    # --- Size feasibility (hard filter should have removed infeasible ones,
    #     but apply a soft penalty here just in case) ---
    size_cat = categorise_dataset_size(profile.num_rows)

    if size_cat == DatasetSizeCategory.TINY:
        # Penalise complex ensembles on tiny datasets (overfitting risk)
        if descriptor.scalability <= 2 and descriptor.interpretability <= 3:
            score -= 0.12
        # Reward interpretable models on tiny datasets
        if descriptor.interpretability >= 4:
            score += 0.05

    elif size_cat in (DatasetSizeCategory.MEDIUM, DatasetSizeCategory.LARGE):
        # Reward scalable models on large datasets
        score += (descriptor.scalability - 3) * 0.04

    # --- Missing value handling ---
    if profile.has_missing_after_engineering and descriptor.handles_missing:
        score += 0.04

    # --- Categorical features ---
    cat_ratio = (
        len(profile.categorical_features) / profile.num_feature_cols
        if profile.num_feature_cols > 0 else 0.0
    )
    if cat_ratio > 0.4 and descriptor.handles_categorical:
        score += 0.05
    if cat_ratio > 0.4 and not descriptor.handles_categorical:
        score -= 0.04

    # --- Class imbalance (classification only) ---
    if (
        task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION)
        and profile.class_imbalance_ratio is not None
        and profile.class_imbalance_ratio > 3.0
        and descriptor.handles_imbalance
    ):
        score += 0.05

    # --- High-dimensional feature space ---
    if profile.num_feature_cols > 50 and descriptor.handles_high_dimensions:
        score += 0.04
    if profile.num_feature_cols > 50 and not descriptor.handles_high_dimensions:
        score -= 0.05

    # Clamp to [0.0, 1.0]
    return round(min(max(score, 0.0), 1.0), 4)
class ModelRegistry:
    """
    Central registry of all known ML algorithm descriptors.

    The registry is the single source of truth for algorithm metadata.
    To add a new algorithm:

        registry = ModelRegistry()
        registry.register(ModelDescriptor(...))

    New descriptors registered at runtime are included in all subsequent
    calls to get_candidates() within the same process.
    """

    def __init__(self) -> None:
        self._models: dict[str, ModelDescriptor] = {
            m.name: m for m in _BUILTIN_MODELS
        }
    def register(self, descriptor: ModelDescriptor) -> None:
        """
        Add or replace a ModelDescriptor in the registry.

        If a descriptor with the same name already exists it will be
        overwritten, allowing callers to override built-in defaults.
        """
        self._models[descriptor.name] = descriptor

    def get_all(self) -> list[ModelDescriptor]:
        """Return all registered descriptors."""
        return list(self._models.values())

    def get_by_name(self, name: str) -> ModelDescriptor | None:
        """Return a descriptor by its unique name, or None if not found."""
        return self._models.get(name)

    def get_candidates(
        self,
        task_type: TaskType,
        profile: "DatasetProfile",
        max_candidates: int = 7,
    ) -> list[ModelDescriptor]:
        """
        Return a ranked list of model descriptors suitable for the given
        task type and dataset profile.

        Filtering:
          1. Must support the task type.
          2. Must be feasible given the row count (min_rows / max_rows).

        Ranking:
          Scored by _compute_score(), then sorted descending.
          Top `max_candidates` are returned.

        Parameters
        ----------
        task_type : TaskType
            The ML task to solve.
        profile : DatasetProfile
            Quantitative characteristics of the dataset.
        max_candidates : int
            Maximum number of candidates to return (default 7).

        Returns
        -------
        list[ModelDescriptor]
            Descriptors sorted from best (index 0) to least suitable.
        """
        feasible = [
            m for m in self._models.values()
            if m.supports_task(task_type) and m.is_feasible_for(profile.num_rows)
        ]

        scored = sorted(
            feasible,
            key=lambda m: _compute_score(m, task_type, profile),
            reverse=True,
        )

        return scored[:max_candidates]

    def get_score(
        self,
        descriptor: ModelDescriptor,
        task_type: TaskType,
        profile: "DatasetProfile",
    ) -> float:
        """
        Expose the scoring function publicly so agents can display or log
        individual model scores.
        """
        return _compute_score(descriptor, task_type, profile)
