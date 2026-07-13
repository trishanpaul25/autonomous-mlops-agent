"""
Feature Engineering Tool.

Executes dataset transformations (missing value imputation, outlier
treatment, categorical encoding, numerical scaling) decided by the
Feature Engineering Agent, and updates the FeatureEngineeringState.

This tool performs execution only. It does not contain any LLM
reasoning.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

from schemas.feature_engineering_schema import FeatureEngineeringOutput
from state.pipeline_state import PipelineState
from tools.base_tool import BaseTool


class FeatureEngineeringTool(BaseTool):
    """
    Performs feature engineering using pandas and scikit-learn.
    """

    def execute(
        self,
        state: PipelineState,
        config: FeatureEngineeringOutput,
    ) -> PipelineState:

        df = state.dataset.dataframe.copy()

        fe_state = state.feature_engineering

        target_column = state.validation.target_column

        rows_before = len(df)

        # ------------------------------------
        # 1. Drop columns
        # ------------------------------------

        drop_cols = [
            col
            for col in config.drop_columns
            if col in df.columns and col != target_column
        ]

        if drop_cols:

            df = df.drop(columns=drop_cols)

            fe_state.dropped_columns = drop_cols

            fe_state.transformations_applied.append(
                f"Dropped columns: {drop_cols}"
            )

        # ------------------------------------
        # 2. Handle missing values
        # ------------------------------------

        if config.handle_missing_values:

            df = self._handle_missing_values(
                df,
                target_column,
                config,
                fe_state,
            )

        # ------------------------------------
        # 3. Handle outliers
        # ------------------------------------

        if config.handle_outliers and config.outlier_method != "none":

            df = self._handle_outliers(
                df,
                target_column,
                config,
                fe_state,
            )

        # ------------------------------------
        # 4. Encode categorical columns
        # ------------------------------------

        if config.encode_categorical:

            df = self._encode_categorical(
                df,
                target_column,
                config,
                fe_state,
            )

        # ------------------------------------
        # 5. Scale numerical columns
        # ------------------------------------

        if config.scale_numerical and config.scaling_method != "none":

            df = self._scale_numerical(
                df,
                target_column,
                config,
                fe_state,
            )

        # ------------------------------------
        # Update state
        # ------------------------------------

        fe_state.rows_removed = rows_before - len(df)

        feature_columns = [
            col for col in df.columns if col != target_column
        ]

        fe_state.final_feature_columns = feature_columns

        fe_state.final_shape = {
            "rows": df.shape[0],
            "columns": df.shape[1],
        }

        fe_state.is_completed = True

        fe_state.summary = (
            "Feature engineering completed successfully. "
            f"Final shape: {df.shape[0]} rows x {df.shape[1]} columns."
        )

        state.dataset.dataframe = df

        state.dataset.feature_columns = feature_columns

        state.dataset.target_column = target_column

        state.dataset.num_rows = df.shape[0]

        state.dataset.num_columns = df.shape[1]

        return state

    # ==========================================================
    # Internal helpers
    # ==========================================================

    @staticmethod
    def _feature_columns(df: pd.DataFrame, target_column: str | None) -> pd.DataFrame:
        """
        Returns the dataframe restricted to feature columns
        (i.e. every column except the target).
        """

        if target_column and target_column in df.columns:

            return df.drop(columns=[target_column])

        return df

    def _handle_missing_values(
        self,
        df: pd.DataFrame,
        target_column: str | None,
        config: FeatureEngineeringOutput,
        fe_state,
    ) -> pd.DataFrame:

        features_df = self._feature_columns(df, target_column)

        numeric_cols = features_df.select_dtypes(
            include=[np.number]
        ).columns.tolist()

        categorical_cols = features_df.select_dtypes(
            include=["object", "category", "string"]
        ).columns.tolist()

        drop_subset: list[str] = []

        for col in numeric_cols:

            if df[col].isnull().sum() == 0:

                continue

            strategy = config.numeric_missing_strategy

            if strategy == "mean":

                df[col] = df[col].fillna(df[col].mean())

            elif strategy == "median":

                df[col] = df[col].fillna(df[col].median())

            elif strategy == "constant":

                df[col] = df[col].fillna(0)

            elif strategy == "drop":

                drop_subset.append(col)

                continue

            fe_state.imputed_columns[col] = strategy

        for col in categorical_cols:

            if df[col].isnull().sum() == 0:

                continue

            strategy = config.categorical_missing_strategy

            if strategy == "mode":

                mode_values = df[col].mode(dropna=True)

                fill_value = mode_values.iloc[0] if not mode_values.empty else "unknown"

                df[col] = df[col].fillna(fill_value)

            elif strategy == "constant":

                df[col] = df[col].fillna("missing")

            elif strategy == "drop":

                drop_subset.append(col)

                continue

            fe_state.imputed_columns[col] = strategy

        if drop_subset:

            df = df.dropna(subset=drop_subset)

            fe_state.imputed_columns.update(
                {col: "drop" for col in drop_subset}
            )

        fe_state.transformations_applied.append(
            "Handled missing values."
        )

        return df

    def _handle_outliers(
        self,
        df: pd.DataFrame,
        target_column: str | None,
        config: FeatureEngineeringOutput,
        fe_state,
    ) -> pd.DataFrame:

        features_df = self._feature_columns(df, target_column)

        numeric_cols = features_df.select_dtypes(
            include=[np.number]
        ).columns.tolist()

        treated_cols = []

        for col in numeric_cols:

            if config.outlier_method == "iqr":

                q1 = df[col].quantile(0.25)

                q3 = df[col].quantile(0.75)

                iqr = q3 - q1

                lower_bound = q1 - 1.5 * iqr

                upper_bound = q3 + 1.5 * iqr

            elif config.outlier_method == "zscore":

                mean = df[col].mean()

                std = df[col].std()

                if std == 0 or pd.isna(std):

                    continue

                lower_bound = mean - 3 * std

                upper_bound = mean + 3 * std

            else:

                continue

            outlier_count = int(
                ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            )

            if outlier_count == 0:

                continue

            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

            treated_cols.append(col)

        if treated_cols:

            fe_state.outlier_treated_columns = treated_cols

            fe_state.outlier_method = config.outlier_method

            fe_state.transformations_applied.append(
                f"Capped outliers in: {treated_cols} using {config.outlier_method}."
            )

        return df

    def _encode_categorical(
        self,
        df: pd.DataFrame,
        target_column: str | None,
        config: FeatureEngineeringOutput,
        fe_state,
    ) -> pd.DataFrame:

        features_df = self._feature_columns(df, target_column)

        categorical_cols = features_df.select_dtypes(
            include=["object", "category", "string"]
        ).columns.tolist()

        if not categorical_cols:

            return df

        if config.encoding_method == "onehot":

            df = pd.get_dummies(
                df,
                columns=categorical_cols,
                drop_first=True,
            )

            for col in categorical_cols:

                fe_state.encoded_columns[col] = "onehot"

        elif config.encoding_method == "label":

            for col in categorical_cols:

                encoder = LabelEncoder()

                df[col] = encoder.fit_transform(
                    df[col].astype(str)
                )

                fe_state.encoded_columns[col] = "label"

        fe_state.transformations_applied.append(
            f"Encoded categorical columns: {categorical_cols} using {config.encoding_method}."
        )

        return df

    def _scale_numerical(
        self,
        df: pd.DataFrame,
        target_column: str | None,
        config: FeatureEngineeringOutput,
        fe_state,
    ) -> pd.DataFrame:

        features_df = self._feature_columns(df, target_column)

        numeric_cols = features_df.select_dtypes(
            include=[np.number]
        ).columns.tolist()

        if not numeric_cols:

            return df

        if config.scaling_method == "standard":

            scaler = StandardScaler()

        elif config.scaling_method == "minmax":

            scaler = MinMaxScaler()

        elif config.scaling_method == "robust":

            scaler = RobustScaler()

        else:

            return df

        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        fe_state.scaled_columns = numeric_cols

        fe_state.transformations_applied.append(
            f"Scaled numerical columns: {numeric_cols} using {config.scaling_method} scaling."
        )

        return df
