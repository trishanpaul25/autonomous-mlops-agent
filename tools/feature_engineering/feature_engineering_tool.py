"""
Feature Engineering Tool (leakage-fixed).

Executes dataset transformations (missing value imputation, outlier
treatment, categorical encoding, numerical scaling) decided by the
Feature Engineering Agent, and updates the FeatureEngineeringState.

WHY THIS VERSION IS DIFFERENT
------------------------------
The previous version fit every statistic-based transform (imputation
means/modes, outlier IQR/z-score bounds, one-hot categories, scaler
mean/std) on the FULL dataset, and only split into train/test afterwards
(in TrainTestSplitter, which runs later during Model Training). That
means test-set values leaked into every one of those statistics before
the model ever saw the "unseen" data — invalidating any accuracy number
downstream.

This version fixes that by:
  1. Doing everything that is safely row-wise (derived features, column
     drops) BEFORE the split — these don't use cross-row statistics, so
     there's no leakage risk.
  2. Splitting into train/test immediately after that.
  3. Fitting every statistic (means, modes, IQR/z-score bounds, one-hot
     categories, scaler parameters) on the TRAIN split only, then
     applying (transform-only) those fitted statistics to the TEST split.
  4. Writing the resulting train/test matrices directly into
     PipelineState.model_training (X_train/X_test/y_train/y_test/etc.),
     the exact same fields TrainTestSplitter populates.

INTEGRATION NOTE
-----------------
TrainTestSplitter.split() already has an idempotency guard:

    if mt.X_train is not None:
        logger.info("... split already exists ... Skipping.")
        return state

Because this tool now populates `mt.X_train` etc. itself, that guard
means TrainTestSplitter naturally no-ops when it runs later during
Model Training — no orchestration/graph changes are required. You can
leave train_test_splitter.py exactly as-is.

No new PipelineState fields are required either: every field this tool
writes into `state.model_training` already exists (populated previously
by TrainTestSplitter), and no fitted transformer objects need to be
persisted in state, since fit and transform both happen inside this
single execute() call.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)

from schemas.feature_engineering_schema import (
    DerivedFeatureSpec,
    FeatureEngineeringOutput,
)
from state.pipeline_state import PipelineState
from tools.base_tool import BaseTool

# Task types that should use stratified splitting (mirrors TrainTestSplitter)
_CLASSIFICATION_TASK_TYPES = {
    "binary_classification",
    "multiclass_classification",
    "classification",
}

# Minimum samples per class required to enable stratification.
_MIN_SAMPLES_PER_CLASS_FOR_STRATIFY = 2


class FeatureEngineeringTool(BaseTool):
    """
    Performs feature engineering using pandas and scikit-learn, with all
    statistic-based transforms fit on the training split only.
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42) -> None:
        """
        Parameters
        ----------
        test_size : float
            Fraction of data reserved for testing. Default 0.2 (20%),
            matching TrainTestSplitter's previous default so overall
            pipeline behaviour is unchanged apart from removing leakage.
        random_state : int
            Random seed for reproducibility. Default 42.
        """
        self.test_size = test_size
        self.random_state = random_state

    def execute(
        self,
        state: PipelineState,
        config: FeatureEngineeringOutput,
    ) -> PipelineState:

        df = state.dataset.dataframe.copy()

        fe_state = state.feature_engineering

        target_column = state.validation.target_column

        fe_state.config = config.model_dump()

        rows_before = len(df)

        # ------------------------------------------------------------
        # 1. Create derived features — row-wise only (title extraction,
        #    family_size, missing flags, etc.), so this is safe to run
        #    on the full dataset before splitting.
        # ------------------------------------------------------------

        if config.derived_features:
            df = self._create_derived_features(df, config, fe_state)

        # ------------------------------------------------------------
        # 2. Drop columns — column-wise, safe before splitting.
        # ------------------------------------------------------------

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

        # ------------------------------------------------------------
        # 3. Split BEFORE any statistic-based transform. Everything
        #    after this point fits on train_df only.
        # ------------------------------------------------------------

        train_df, test_df, stratified = self._split(df, target_column, state)

        # ------------------------------------------------------------
        # 4. Handle missing values (fit on train, apply to both)
        # ------------------------------------------------------------

        if config.handle_missing_values:
            train_df, test_df = self._handle_missing_values(
                train_df, test_df, target_column, config, fe_state,
            )

        # ------------------------------------------------------------
        # 5. Handle outliers (bounds fit on train, applied to both)
        # ------------------------------------------------------------

        if config.handle_outliers and config.outlier_method != "none":
            train_df, test_df = self._handle_outliers(
                train_df, test_df, target_column, config, fe_state,
            )

        # ------------------------------------------------------------
        # 6. Encode categorical columns (fit on train, applied to both)
        # ------------------------------------------------------------

        if config.encode_categorical:
            train_df, test_df = self._encode_categorical(
                train_df, test_df, target_column, config, fe_state,
            )

        # ------------------------------------------------------------
        # 7. Scale numerical columns (fit on train, applied to both)
        # ------------------------------------------------------------

        if config.scale_numerical and config.scaling_method != "none":
            train_df, test_df = self._scale_numerical(
                train_df, test_df, target_column, config, fe_state,
            )

        # ------------------------------------------------------------
        # Column safety net: guarantee identical feature columns on
        # both sides even if one-hot ever produced a mismatch.
        # ------------------------------------------------------------

        feature_columns = [c for c in train_df.columns if c != target_column]
        test_df = test_df.reindex(columns=train_df.columns, fill_value=0)

        # ------------------------------------------------------------
        # Update FeatureEngineeringState
        # ------------------------------------------------------------

        total_rows_after = len(train_df) + len(test_df)
        fe_state.rows_removed = rows_before - total_rows_after

        fe_state.final_feature_columns = feature_columns

        fe_state.final_shape = {
            "rows": total_rows_after,
            "columns": train_df.shape[1],
        }

        fe_state.is_completed = True

        fe_state.summary = (
            "Feature engineering completed successfully (leak-free — all "
            "statistics fit on the training split only). "
            f"Train: {train_df.shape[0]} rows | Test: {test_df.shape[0]} rows | "
            f"Columns: {train_df.shape[1]}."
        )

        # ------------------------------------------------------------
        # Write the split + transformed matrices directly into
        # ModelTrainingState. TrainTestSplitter's idempotency guard
        # will see mt.X_train is already populated and no-op later.
        # ------------------------------------------------------------

        mt = state.model_training

        mt.X_train = train_df[feature_columns].values.tolist()
        mt.X_test = test_df[feature_columns].values.tolist()

        if target_column and target_column in train_df.columns:
            mt.y_train = train_df[target_column].values.tolist()
            mt.y_test = test_df[target_column].values.tolist()
        else:
            mt.y_train = [0] * len(train_df)
            mt.y_test = [0] * len(test_df)

        mt.feature_columns = feature_columns
        mt.target_column = target_column
        mt.test_size = self.test_size
        mt.random_state = self.random_state
        mt.stratified = stratified
        mt.train_samples = len(train_df)
        mt.test_samples = len(test_df)

        state.model_training = mt

        # ------------------------------------------------------------
        # state.dataset.dataframe now reflects the TRAIN split only.
        # Anything downstream (Model Selection's DatasetProfiler, etc.)
        # that reads state.dataset.dataframe will only ever see
        # training data — the held-out test set lives exclusively in
        # state.model_training.X_test / y_test and must not be reused
        # for any fitting/profiling/selection step.
        # ------------------------------------------------------------

        state.dataset.dataframe = train_df
        state.dataset.feature_columns = feature_columns
        state.dataset.target_column = target_column
        state.dataset.num_rows = train_df.shape[0]
        state.dataset.num_columns = train_df.shape[1]

        return state

    # ==========================================================
    # Splitting
    # ==========================================================

    def _split(
        self,
        df: pd.DataFrame,
        target_column: str | None,
        state: PipelineState,
    ) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
        """
        Row-level train/test split performed BEFORE any statistic-based
        transform is fit, to prevent test-set leakage.

        Stratifies automatically for classification problem types when
        the minority class has enough samples, mirroring
        TrainTestSplitter's original stratification logic. Uses
        state.validation.problem_type since Model Selection (which sets
        the more granular task_type) hasn't run yet at this point in
        the pipeline.
        """
        problem_type = (state.validation.problem_type or "").lower()

        stratify_values = None
        stratified = False

        if (
            target_column
            and target_column in df.columns
            and problem_type in _CLASSIFICATION_TASK_TYPES
        ):
            y = df[target_column]
            counts = y.value_counts(dropna=False)
            min_count = int(counts.min()) if len(counts) > 0 else 0

            if min_count >= _MIN_SAMPLES_PER_CLASS_FOR_STRATIFY:
                stratify_values = y
                stratified = True

        train_df, test_df = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_values,
        )

        return (
            train_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
            stratified,
        )

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

    def _create_derived_features(
        self,
        df: pd.DataFrame,
        config: FeatureEngineeringOutput,
        fe_state,
    ) -> pd.DataFrame:

        created: list[str] = []

        for spec in config.derived_features:

            try:
                new_col = self._apply_derivation(df, spec)
            except Exception as exc:
                fe_state.warnings.append(
                    f"Could not create derived feature '{spec.new_column}': {exc}"
                )
                continue

            if new_col is None:
                continue

            df[spec.new_column] = new_col
            created.append(spec.new_column)

        if created:
            fe_state.derived_features = created
            fe_state.transformations_applied.append(
                f"Created derived features: {created}"
            )

        return df

    @staticmethod
    def _apply_derivation(
        df: pd.DataFrame,
        spec: DerivedFeatureSpec,
    ) -> pd.Series | None:
        """
        Computes a single derived column. Returns None (and skips the
        feature) when a required source column is missing, rather than
        failing the whole pipeline over one bad spec.

        All operations below are strictly row-wise (they never use a
        cross-row statistic like mean/std/mode), which is why derived
        features are safe to compute before the train/test split.
        """

        missing_sources = [c for c in spec.source_columns if c not in df.columns]

        if missing_sources or not spec.source_columns:
            return None

        if spec.operation == "regex_extract":
            if not spec.pattern:
                return None
            col = spec.source_columns[0]
            result = df[col].astype(str).str.extract(spec.pattern, expand=False)
            result = result.str.strip()
            if spec.fillna is not None:
                result = result.fillna(spec.fillna)
            return result

        if spec.operation == "first_char":
            col = spec.source_columns[0]
            result = df[col].astype(str).str[0]
            result = result.where(df[col].notna(), spec.fillna)
            return result

        if spec.operation == "sum_columns":
            if len(spec.source_columns) < 2:
                return None
            return df[spec.source_columns].sum(axis=1) + (spec.constant or 0)

        if spec.operation == "ratio_columns":
            if len(spec.source_columns) != 2:
                return None
            numerator, denominator = spec.source_columns
            safe_denominator = df[denominator].replace(0, np.nan)
            return (df[numerator] / safe_denominator).fillna(0)

        if spec.operation == "log1p":
            col = spec.source_columns[0]
            return np.log1p(df[col].clip(lower=0))

        if spec.operation == "missing_flag":
            col = spec.source_columns[0]
            return df[col].isnull().astype(int)

        if spec.operation == "equals_flag":
            if spec.constant is None:
                return None
            col = spec.source_columns[0]
            return (df[col] == spec.constant).astype(int)

        return None

    def _handle_missing_values(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_column: str | None,
        config: FeatureEngineeringOutput,
        fe_state,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:

        features_df = self._feature_columns(train_df, target_column)

        numeric_cols = features_df.select_dtypes(
            include=[np.number]
        ).columns.tolist()

        categorical_cols = features_df.select_dtypes(
            include=["object", "category", "string"]
        ).columns.tolist()

        train_drop_subset: list[str] = []
        test_drop_subset: list[str] = []

        for col in numeric_cols:

            if train_df[col].isnull().sum() == 0 and test_df[col].isnull().sum() == 0:
                continue

            strategy = config.numeric_missing_strategy

            if strategy == "mean":
                fill_value = train_df[col].mean()
            elif strategy == "median":
                fill_value = train_df[col].median()
            elif strategy == "constant":
                fill_value = 0
            elif strategy == "drop":
                train_drop_subset.append(col)
                test_drop_subset.append(col)
                continue
            else:
                continue

            # Fit value comes from train only; applied identically to both.
            train_df[col] = train_df[col].fillna(fill_value)
            test_df[col] = test_df[col].fillna(fill_value)

            fe_state.imputed_columns[col] = strategy
            fe_state.fitted_imputation_values[col] = fill_value

        for col in categorical_cols:

            if train_df[col].isnull().sum() == 0 and test_df[col].isnull().sum() == 0:
                continue

            strategy = config.categorical_missing_strategy

            if strategy == "mode":
                mode_values = train_df[col].mode(dropna=True)
                fill_value = mode_values.iloc[0] if not mode_values.empty else "unknown"
            elif strategy == "constant":
                fill_value = "missing"
            elif strategy == "drop":
                train_drop_subset.append(col)
                test_drop_subset.append(col)
                continue
            else:
                continue

            train_df[col] = train_df[col].fillna(fill_value)
            test_df[col] = test_df[col].fillna(fill_value)

            fe_state.imputed_columns[col] = strategy
            fe_state.fitted_imputation_values[col] = fill_value

        if train_drop_subset:
            train_df = train_df.dropna(subset=train_drop_subset)
        if test_drop_subset:
            test_df = test_df.dropna(subset=test_drop_subset)
        if train_drop_subset or test_drop_subset:
            fe_state.imputed_columns.update(
                {col: "drop" for col in set(train_drop_subset) | set(test_drop_subset)}
            )

        fe_state.transformations_applied.append(
            "Handled missing values (statistics fit on train split only)."
        )

        return train_df, test_df

    def _handle_outliers(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_column: str | None,
        config: FeatureEngineeringOutput,
        fe_state,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:

        features_df = self._feature_columns(train_df, target_column)

        numeric_cols = features_df.select_dtypes(
            include=[np.number]
        ).columns.tolist()

        treated_cols = []

        for col in numeric_cols:

            if config.outlier_method == "iqr":
                q1 = train_df[col].quantile(0.25)
                q3 = train_df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

            elif config.outlier_method == "zscore":
                mean = train_df[col].mean()
                std = train_df[col].std()
                if std == 0 or pd.isna(std):
                    continue
                lower_bound = mean - 3 * std
                upper_bound = mean + 3 * std

            else:
                continue

            outlier_count = int(
                ((train_df[col] < lower_bound) | (train_df[col] > upper_bound)).sum()
            )

            if outlier_count == 0:
                continue

            # Bounds are computed from train only, then applied to both.
            train_df[col] = train_df[col].clip(lower=lower_bound, upper=upper_bound)
            test_df[col] = test_df[col].clip(lower=lower_bound, upper=upper_bound)

            treated_cols.append(col)
            fe_state.fitted_outlier_bounds[col] = [float(lower_bound), float(upper_bound)]

        if treated_cols:
            fe_state.outlier_treated_columns = treated_cols
            fe_state.outlier_method = config.outlier_method
            fe_state.transformations_applied.append(
                f"Capped outliers (bounds fit on train split only) in: "
                f"{treated_cols} using {config.outlier_method}."
            )

        return train_df, test_df

    def _encode_categorical(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_column: str | None,
        config: FeatureEngineeringOutput,
        fe_state,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:

        features_df = self._feature_columns(train_df, target_column)

        categorical_cols = features_df.select_dtypes(
            include=["object", "category", "string"]
        ).columns.tolist()

        if not categorical_cols:
            return train_df, test_df

        if config.encoding_method == "onehot":

            # handle_unknown="ignore" means any category seen only in
            # test (never in train) becomes an all-zero row instead of
            # crashing — the correct behaviour for genuinely unseen
            # categories at inference time.
            encoder = OneHotEncoder(
                handle_unknown="ignore",
                drop="first",
                sparse_output=False,
            )

            train_encoded = encoder.fit_transform(
                train_df[categorical_cols].astype(str)
            )
            test_encoded = encoder.transform(
                test_df[categorical_cols].astype(str)
            )

            encoded_cols = encoder.get_feature_names_out(categorical_cols)

            train_df = train_df.drop(columns=categorical_cols)
            test_df = test_df.drop(columns=categorical_cols)

            train_encoded_df = pd.DataFrame(
                train_encoded, columns=encoded_cols, index=train_df.index
            )
            test_encoded_df = pd.DataFrame(
                test_encoded, columns=encoded_cols, index=test_df.index
            )

            train_df = pd.concat([train_df, train_encoded_df], axis=1)
            test_df = pd.concat([test_df, test_encoded_df], axis=1)

            for col in categorical_cols:
                fe_state.encoded_columns[col] = "onehot"

            fe_state.fitted_onehot_encoder = encoder
            fe_state.onehot_encoded_columns = categorical_cols

        elif config.encoding_method == "label":

            for col in categorical_cols:
                train_values = train_df[col].astype(str)
                categories = {
                    value: code
                    for code, value in enumerate(sorted(train_values.unique()))
                }
                # Reserve one extra code for categories seen only at
                # test/inference time and never during training.
                unseen_code = len(categories)

                train_df[col] = train_values.map(categories)
                test_df[col] = (
                    test_df[col]
                    .astype(str)
                    .map(categories)
                    .fillna(unseen_code)
                    .astype(int)
                )

                fe_state.encoded_columns[col] = "label"

                label_map = dict(categories)
                label_map["__unseen__"] = unseen_code
                fe_state.fitted_label_encoding_maps[col] = label_map

        fe_state.transformations_applied.append(
            f"Encoded categorical columns: {categorical_cols} using "
            f"{config.encoding_method} (fit on train split only)."
        )

        return train_df, test_df

    def _scale_numerical(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_column: str | None,
        config: FeatureEngineeringOutput,
        fe_state,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:

        features_df = self._feature_columns(train_df, target_column)

        numeric_cols = features_df.select_dtypes(
            include=[np.number]
        ).columns.tolist()

        if not numeric_cols:
            return train_df, test_df

        if config.scaling_method == "standard":
            scaler = StandardScaler()
        elif config.scaling_method == "minmax":
            scaler = MinMaxScaler()
        elif config.scaling_method == "robust":
            scaler = RobustScaler()
        else:
            return train_df, test_df

        scaler.fit(train_df[numeric_cols])

        train_df[numeric_cols] = scaler.transform(train_df[numeric_cols])
        test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])

        fe_state.scaled_columns = numeric_cols
        fe_state.fitted_scaler = scaler

        fe_state.transformations_applied.append(
            f"Scaled numerical columns (fit on train split only): "
            f"{numeric_cols} using {config.scaling_method} scaling."
        )

        return train_df, test_df