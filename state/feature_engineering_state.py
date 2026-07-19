"""
State used by the Feature Engineering Agent.
"""

from typing import Any

from pydantic import Field

from .base_state import BaseState


class FeatureEngineeringState(BaseState):
    """
    Stores the results of feature engineering.
    """

    # Whether feature engineering completed successfully
    is_completed: bool = False

    # New columns created (e.g. extracted title, family size), in order applied
    derived_features: list[str] = Field(
        default_factory=list
    )

    # Columns dropped (ids, constant columns, user/LLM specified, etc.)
    dropped_columns: list[str] = Field(
        default_factory=list
    )

    # Columns whose missing values were imputed, mapped to the strategy used
    imputed_columns: dict[str, str] = Field(
        default_factory=dict
    )

    # Categorical columns that were encoded, mapped to the method used
    encoded_columns: dict[str, str] = Field(
        default_factory=dict
    )

    # Numerical columns that were scaled
    scaled_columns: list[str] = Field(
        default_factory=list
    )

    # Numerical columns where outliers were capped/removed
    outlier_treated_columns: list[str] = Field(
        default_factory=list
    )

    # Method used for outlier treatment, if any
    outlier_method: str | None = None

    # Number of rows removed as a result of feature engineering (e.g. dropna)
    rows_removed: int = 0

    # Final list of feature columns after transformation (excludes target)
    final_feature_columns: list[str] = Field(
        default_factory=list
    )

    # Shape of the dataset after feature engineering
    final_shape: dict[str, int] = Field(
        default_factory=dict
    )

    # High level list of transformations applied, in order
    transformations_applied: list[str] = Field(
        default_factory=list
    )

    # Any warnings generated during feature engineering
    warnings: list[str] = Field(
        default_factory=list
    )

    # Free-form extra info bag
    extra: dict[str, Any] = Field(
        default_factory=dict
    )

    # Summary of feature engineering
    summary: str | None = None

    # ------------------------------------------------------------
    # NEW: reproducibility fields, added for the Model Registry Agent.
    #
    # Everything above this point describes WHAT happened. The fields
    # below capture the actual fitted artifacts needed to REPLAY the
    # same pipeline on brand-new raw data at inference time — without
    # them, a registered model can only ever be applied to data that's
    # already been through an identical hand-run of this tool, which
    # isn't useful for real deployment.
    #
    # All fitted objects (scaler, encoder) are stored as the actual
    # sklearn objects, not a serialized summary — BaseState already
    # sets arbitrary_types_allowed=True for exactly this reason (see
    # HyperparameterOptimizationState.optimized_model_objects for the
    # same pattern with fitted estimators).
    # ------------------------------------------------------------

    # The original FeatureEngineeringOutput config (derived_features specs,
    # drop_columns, missing/outlier/encoding/scaling strategy choices),
    # stored as a plain dict via .model_dump(). Needed to replay the
    # row-wise steps (derive, drop) that don't have "fitted state" of
    # their own — they're just deterministic functions of the config.
    config: dict[str, Any] | None = None

    # {column_name: fill_value} — computed from the TRAIN split only.
    fitted_imputation_values: dict[str, Any] = Field(default_factory=dict)

    # {column_name: [lower_bound, upper_bound]} — computed from TRAIN only.
    fitted_outlier_bounds: dict[str, list[float]] = Field(default_factory=dict)

    # The fitted sklearn OneHotEncoder instance (encoding_method == "onehot"),
    # or None if label encoding or no encoding was used.
    fitted_onehot_encoder: Any | None = None

    # Ordered list of columns the OneHotEncoder was fit on (needed to feed
    # columns to encoder.transform() in the exact fitted order).
    onehot_encoded_columns: list[str] = Field(default_factory=list)

    # {column_name: {category_value: code, ..., "__unseen__": fallback_code}}
    # — used when encoding_method == "label" instead of a sklearn object,
    # since the tool implements label encoding by hand (see
    # feature_engineering_tool.py's _encode_categorical).
    fitted_label_encoding_maps: dict[str, dict[str, int]] = Field(default_factory=dict)

    # The fitted sklearn scaler instance (StandardScaler / MinMaxScaler /
    # RobustScaler), or None if scaling was not applied.
    fitted_scaler: Any | None = None