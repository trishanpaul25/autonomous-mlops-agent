"""
Bundled Pipeline Model — MLflow pyfunc wrapper.

Wraps the fitted feature-transform pipeline (FeatureTransformReplay)
together with the fitted best estimator into a single MLflow pyfunc
model. Whoever loads this model later (Deployment Agent, a notebook,
`mlflow models serve`) can call .predict() with a raw pandas DataFrame
shaped like the ORIGINAL dataset — not a pre-transformed one — and get
predictions straight back, with no knowledge of the feature engineering
pipeline required on the caller's side.

Design notes
------------
* No LLM involvement — deterministic transform + predict only.
* Kept import-light at module scope: only mlflow.pyfunc and pandas are
  imported unconditionally, since this module gets pickled and
  reloaded inside whatever process eventually serves the model.
* Assumes the deployment environment has this project's `tools` and
  `schemas` packages importable (FeatureTransformReplay depends on
  them). This is intentional for now — the registry is designed to
  feed this project's own Deployment Agent, not to be a fully
  environment-agnostic artifact. Worth revisiting if a truly external
  serving environment is ever needed.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

try:
    import mlflow.pyfunc
except ImportError as exc:  # pragma: no cover - surfaced clearly by the agent instead
    raise ImportError(
        "BundledPipelineModel requires the 'mlflow' package. "
        "Install it with: pip install mlflow"
    ) from exc

from tools.model_registry.feature_pipeline_replay import FeatureTransformReplay


class BundledPipelineModel(mlflow.pyfunc.PythonModel):
    """
    A single deployable unit: fitted feature transforms + fitted estimator.
    """

    def __init__(
        self,
        replay: FeatureTransformReplay,
        estimator: Any,
        task_type: str | None,
        return_proba: bool = True,
    ) -> None:
        self.replay = replay
        self.estimator = estimator
        self.task_type = (task_type or "").lower()
        self.return_proba = return_proba

    def predict(self, context, model_input, params: dict | None = None) -> pd.DataFrame:
        """
        Parameters
        ----------
        model_input : pd.DataFrame
            Raw input rows, shaped like the original (pre-feature-
            engineering) dataset. If a dict/list is passed instead
            (e.g. from a JSON API request), it's coerced to a
            DataFrame first.

        Returns
        -------
        pd.DataFrame
            Column 'prediction', plus 'probability_class_i' columns
            for each class if this is classification and the estimator
            supports predict_proba.
        """
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)

        transformed = self.replay.transform(model_input)

        predictions = self.estimator.predict(transformed)
        result = pd.DataFrame({"prediction": predictions}, index=model_input.index)

        if self.return_proba and "classification" in self.task_type:
            predict_proba = getattr(self.estimator, "predict_proba", None)
            if predict_proba is not None:
                try:
                    proba = predict_proba(transformed)
                    for class_idx in range(proba.shape[1]):
                        result[f"probability_class_{class_idx}"] = proba[:, class_idx]
                except Exception:
                    # Probability estimation failing shouldn't take down
                    # an otherwise-valid prediction — degrade silently.
                    pass

        return result