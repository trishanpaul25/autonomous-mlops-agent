"""
Feature Transform Replay.

Reconstructs the exact fitted feature-engineering pipeline (derived
features -> drop columns -> impute -> clip outliers -> encode -> scale)
from a FeatureEngineeringState, so it can be applied to brand-new raw
data at inference time — not just to data that was already processed
by an in-memory run of FeatureEngineeringTool.

Why this exists
----------------
FeatureEngineeringTool.execute() fits every statistic (imputation
values, outlier bounds, encoder categories, scaler mean/std) on the
train split and stores the fitted objects on FeatureEngineeringState.
That's sufficient for training. But a registered/deployed model needs
to accept genuinely new raw rows (e.g. a new passenger record) and
produce a prediction end-to-end — which means replaying the same
transform steps, in the same order, using those same fitted values,
without re-fitting anything.

This class is the single place that replay logic lives, so both the
Model Registry Agent's bundled MLflow pyfunc wrapper and (later) the
Deployment Agent's real-time inference path use identical logic
instead of two hand-written copies that could drift apart.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from schemas.feature_engineering_schema import DerivedFeatureSpec
from tools.feature_engineering.feature_engineering_tool import FeatureEngineeringTool

if TYPE_CHECKING:
    from state.feature_engineering_state import FeatureEngineeringState


class FeatureTransformReplay:
    """
    Wraps a fitted FeatureEngineeringState and exposes a single
    .transform(raw_df) -> transformed_df method that reproduces the
    exact fit-time pipeline on new data.
    """

    def __init__(
        self,
        fe_state: "FeatureEngineeringState",
        target_column: str | None = None,
    ) -> None:
        self.fe_state = fe_state
        self.target_column = target_column

        cfg: dict[str, Any] = fe_state.config or {}
        self.derived_feature_specs: list[DerivedFeatureSpec] = [
            DerivedFeatureSpec(**spec) for spec in cfg.get("derived_features", []) or []
        ]
        self.drop_columns: list[str] = cfg.get("drop_columns", []) or []
        self.encoding_method: str | None = cfg.get("encoding_method")
        self.scaling_method: str | None = cfg.get("scaling_method")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replay the full fitted pipeline on new raw data.

        Parameters
        ----------
        df : pd.DataFrame
            Raw input, with the same original column names/shape the
            pipeline originally saw (before any feature engineering).

        Returns
        -------
        pd.DataFrame
            Transformed features, column-aligned to
            fe_state.final_feature_columns (missing engineered columns
            — e.g. a one-hot category never seen in this batch — are
            filled with 0 rather than raising).
        """
        df = df.copy()

        # 1. Derived features — row-wise, reuses FeatureEngineeringTool's
        #    own derivation logic so there's exactly one implementation.
        for spec in self.derived_feature_specs:
            try:
                new_col = FeatureEngineeringTool._apply_derivation(df, spec)
            except Exception:
                continue
            if new_col is not None:
                df[spec.new_column] = new_col

        # 2. Drop columns
        drop_cols = [
            c for c in self.drop_columns
            if c in df.columns and c != self.target_column
        ]
        if drop_cols:
            df = df.drop(columns=drop_cols)

        # 3. Impute (transform-only — using fit-time fill values)
        for col, fill_value in self.fe_state.fitted_imputation_values.items():
            if col in df.columns:
                df[col] = df[col].fillna(fill_value)

        # 4. Outlier clipping (transform-only — using fit-time bounds)
        for col, bounds in self.fe_state.fitted_outlier_bounds.items():
            if col in df.columns and len(bounds) == 2:
                lower, upper = bounds
                df[col] = df[col].clip(lower=lower, upper=upper)

        # 5. Encode (transform-only)
        if self.encoding_method == "onehot" and self.fe_state.fitted_onehot_encoder is not None:
            cols = self.fe_state.onehot_encoded_columns
            for c in cols:
                if c not in df.columns:
                    # Column absent from this batch entirely — treat as
                    # missing so handle_unknown="ignore" produces an
                    # all-zero one-hot row instead of crashing.
                    df[c] = None
            encoded = self.fe_state.fitted_onehot_encoder.transform(
                df[cols].astype(str)
            )
            encoded_cols = self.fe_state.fitted_onehot_encoder.get_feature_names_out(cols)
            df = df.drop(columns=[c for c in cols if c in df.columns])
            encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)
            df = pd.concat([df, encoded_df], axis=1)

        elif self.encoding_method == "label" and self.fe_state.fitted_label_encoding_maps:
            for col, mapping in self.fe_state.fitted_label_encoding_maps.items():
                if col not in df.columns:
                    continue
                unseen_code = mapping.get("__unseen__", 0)
                lookup = {k: v for k, v in mapping.items() if k != "__unseen__"}
                df[col] = (
                    df[col].astype(str).map(lookup).fillna(unseen_code).astype(int)
                )

        # 6. Scale (transform-only)
        if self.fe_state.fitted_scaler is not None:
            numeric_cols = [c for c in self.fe_state.scaled_columns if c in df.columns]
            if numeric_cols:
                df[numeric_cols] = self.fe_state.fitted_scaler.transform(df[numeric_cols])

        # 7. Align to the exact training-time feature column set/order.
        #    Any column the fitted pipeline expects but this batch never
        #    produced (e.g. a rare one-hot category) is filled with 0.
        final_cols = self.fe_state.final_feature_columns
        df = df.reindex(columns=final_cols, fill_value=0)

        return df