"""
Dataset Snapshot Builder.

Computes a reference feature-distribution snapshot from the RAW
training dataframe — not the post-feature-engineering one. This
matters: ModelRegistryAgent bundles the fitted feature-engineering
transformers with the registered model precisely so `/predict/<id>`
(server/api/routes/predict.py) can accept raw input rows directly.
Live prediction traffic is therefore raw, and drift must be measured
against the same raw distribution, not the encoded/scaled training
matrix.

Pure, read-only, no LLM — mirrors DatasetProfiler's contract
(tools/model_selection/dataset_profiler.py): takes PipelineState,
returns a plain dict, never mutates anything. Called once from
orchestration_service.py at the point a Deployment row is created,
not from within DeploymentAgent — agents never touch Postgres in this
codebase; orchestration_service.py is the sole persistence boundary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from state.pipeline_state import PipelineState

# Cap on distinct category values retained per categorical column, to
# keep the JSONB blob bounded for high-cardinality columns (e.g. free
# text / IDs that slipped through as "categorical").
_MAX_CATEGORIES = 20


class DatasetSnapshotBuilder:
    """
    Builds the `feature_statistics` payload persisted on DatasetSnapshot.
    """

    def build(self, state: "PipelineState") -> dict[str, Any] | None:
        """
        Returns a dict with `num_rows`, `target_column`, and
        `feature_statistics`, or None if there is no raw dataframe to
        snapshot (e.g. dataset was never loaded, or has since been
        released from memory).
        """
        df = state.dataset.dataframe
        if df is None or df.empty:
            return None

        target_column = state.validation.target_column
        feature_columns = [c for c in df.columns if c != target_column]

        feature_statistics: dict[str, Any] = {}
        for col in feature_columns:
            series = df[col]
            if pd.api.types.is_numeric_dtype(series):
                feature_statistics[col] = self._numerical_stats(series)
            else:
                feature_statistics[col] = self._categorical_stats(series)

        return {
            "num_rows": int(len(df)),
            "target_column": target_column,
            "feature_statistics": feature_statistics,
        }

    @staticmethod
    def _numerical_stats(series: pd.Series) -> dict[str, Any]:
        clean = series.dropna()
        return {
            "type": "numerical",
            "mean": float(clean.mean()) if not clean.empty else None,
            "std": float(clean.std()) if len(clean) > 1 else 0.0,
            "min": float(clean.min()) if not clean.empty else None,
            "max": float(clean.max()) if not clean.empty else None,
        }

    @staticmethod
    def _categorical_stats(series: pd.Series) -> dict[str, Any]:
        clean = series.dropna().astype(str)
        if clean.empty:
            return {"type": "categorical", "frequencies": {}}

        proportions = clean.value_counts(normalize=True)
        top = proportions.iloc[:_MAX_CATEGORIES]
        return {
            "type": "categorical",
            "frequencies": {str(k): float(v) for k, v in top.items()},
        }
