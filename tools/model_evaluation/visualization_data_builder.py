"""
Visualization Data Builder for the Model Evaluation Agent.

Prepares plot-ready data structures for future visualization consumers
(API endpoints, dashboards, Jupyter notebooks).

Design Principles
-----------------
* No Rendering: this module never creates figures, axes, or any plotting
  artefacts. It only produces JSON-serializable Python dicts of lists.
* Serializable Output: all numpy arrays are converted to plain Python
  lists before being placed into the output dict. This ensures the
  data can flow through PipelineState without serialization errors.
* Error Isolation: each chart type is built inside its own try/except.
  A failure in one chart never prevents others from being built.
* Conditional Building: each build_* method is safe to call with None
  inputs. If required inputs are absent, the method returns an empty dict.

Supported Charts
----------------
Classification:
  - Confusion Matrix (all classification tasks)
  - ROC Curve (binary + multiclass OvR)
  - Precision-Recall Curve (binary + multiclass OvR)
  - Feature Importance (models with feature_importances_ attribute)

Regression:
  - Residual Plot data (y_true, y_pred, residuals)
  - Feature Importance (models with feature_importances_ attribute)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from utils.logger import logger


class VisualizationDataBuilder:
    """
    Builds JSON-serializable visualization data structures for a single model.

    All public methods return plain Python dicts. Numpy arrays are always
    converted to lists before being included in the output.
    """

    def build_all(
        self,
        model: Any,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray | None,
        task_type: str,
        feature_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Build all applicable visualization data for a single model.

        Dispatches to the appropriate chart builders based on task type
        and available data. All failures are caught and logged.

        Parameters
        ----------
        model : Any
            Fitted sklearn-compatible estimator.
        y_true : np.ndarray
            Ground-truth target values.
        y_pred : np.ndarray
            Model predictions.
        y_proba : np.ndarray | None
            Class probability estimates, or None.
        task_type : str
            ML task type ('binary_classification', etc.)
        feature_names : list[str] | None
            Feature column names for feature importance charts.

        Returns
        -------
        dict[str, Any]
            Keyed by chart type. All values are JSON-serializable.
        """
        normalized = (task_type or "").lower().strip()
        charts: dict[str, Any] = {}

        is_classification = "classification" in normalized
        is_regression = normalized == "regression"

        if is_classification:
            charts["confusion_matrix"] = self.build_confusion_matrix(
                y_true, y_pred
            )
            if y_proba is not None:
                charts["roc_curve"] = self.build_roc_curve(
                    y_true, y_proba, normalized
                )
                charts["pr_curve"] = self.build_pr_curve(
                    y_true, y_proba, normalized
                )

        if is_regression:
            charts["residual_data"] = self.build_residual_data(y_true, y_pred)

        charts["feature_importance"] = self.build_feature_importance(
            model, feature_names
        )

        return {k: v for k, v in charts.items() if v}

    def build_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, Any]:
        """
        Build confusion matrix data.

        Parameters
        ----------
        y_true : np.ndarray
            Ground-truth labels.
        y_pred : np.ndarray
            Predicted labels.

        Returns
        -------
        dict with keys:
            "labels"  — list of class labels (sorted)
            "matrix"  — 2D list (rows=actual, cols=predicted)
        """
        try:
            from sklearn.metrics import confusion_matrix

            labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
            cm = confusion_matrix(y_true, y_pred, labels=labels)

            return {
                "labels": [str(lbl) for lbl in labels],
                "matrix": cm.tolist(),
            }
        except Exception as exc:
            logger.warning(
                "VisualizationDataBuilder: confusion_matrix failed — %s", exc
            )
            return {}

    def build_roc_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        task_type: str,
    ) -> dict[str, Any]:
        """
        Build ROC curve data.

        For binary classification: single FPR/TPR curve.
        For multiclass: one curve per class (OvR strategy).

        Parameters
        ----------
        y_true : np.ndarray
            Ground-truth labels.
        y_proba : np.ndarray
            Probability matrix from predict_proba().
        task_type : str
            Task type string for binary vs. multiclass dispatch.

        Returns
        -------
        dict with keys:
            Binary: "fpr", "tpr", "auc"
            Multiclass: list of {"class", "fpr", "tpr", "auc"} dicts
        """
        try:
            from sklearn.metrics import roc_curve, roc_auc_score
            from sklearn.preprocessing import label_binarize

            unique_classes = np.unique(y_true)
            is_binary = len(unique_classes) == 2

            if is_binary:
                pos_proba = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
                fpr, tpr, _ = roc_curve(y_true, pos_proba)
                auc = float(roc_auc_score(y_true, pos_proba))
                return {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                    "auc": round(auc, 6),
                }

            # Multiclass: OvR per class
            classes = unique_classes.tolist()
            y_bin = label_binarize(y_true, classes=classes)
            curves = []
            for i, cls in enumerate(classes):
                try:
                    fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
                    auc = float(roc_auc_score(y_bin[:, i], y_proba[:, i]))
                    curves.append({
                        "class": str(cls),
                        "fpr": fpr.tolist(),
                        "tpr": tpr.tolist(),
                        "auc": round(auc, 6),
                    })
                except Exception:
                    continue
            return {"curves": curves}

        except Exception as exc:
            logger.warning(
                "VisualizationDataBuilder: roc_curve failed — %s", exc
            )
            return {}

    def build_pr_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        task_type: str,
    ) -> dict[str, Any]:
        """
        Build Precision-Recall curve data.

        For binary classification: single precision/recall curve.
        For multiclass: one curve per class (OvR strategy).

        Parameters
        ----------
        y_true : np.ndarray
            Ground-truth labels.
        y_proba : np.ndarray
            Probability matrix from predict_proba().
        task_type : str
            Task type string for dispatch.

        Returns
        -------
        dict with keys:
            Binary: "precision", "recall", "average_precision"
            Multiclass: list of {"class", "precision", "recall", "avg_precision"}
        """
        try:
            from sklearn.metrics import (
                precision_recall_curve,
                average_precision_score,
            )
            from sklearn.preprocessing import label_binarize

            unique_classes = np.unique(y_true)
            is_binary = len(unique_classes) == 2

            if is_binary:
                pos_proba = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
                precision, recall, _ = precision_recall_curve(y_true, pos_proba)
                avg_precision = float(average_precision_score(y_true, pos_proba))
                return {
                    "precision": precision.tolist(),
                    "recall": recall.tolist(),
                    "average_precision": round(avg_precision, 6),
                }

            # Multiclass: OvR per class
            classes = unique_classes.tolist()
            y_bin = label_binarize(y_true, classes=classes)
            curves = []
            for i, cls in enumerate(classes):
                try:
                    prec, rec, _ = precision_recall_curve(
                        y_bin[:, i], y_proba[:, i]
                    )
                    avg_prec = float(
                        average_precision_score(y_bin[:, i], y_proba[:, i])
                    )
                    curves.append({
                        "class": str(cls),
                        "precision": prec.tolist(),
                        "recall": rec.tolist(),
                        "average_precision": round(avg_prec, 6),
                    })
                except Exception:
                    continue
            return {"curves": curves}

        except Exception as exc:
            logger.warning(
                "VisualizationDataBuilder: pr_curve failed — %s", exc
            )
            return {}

    def build_residual_data(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, Any]:
        """
        Build residual plot data for regression models.

        Parameters
        ----------
        y_true : np.ndarray
            Ground-truth continuous target values.
        y_pred : np.ndarray
            Predicted values.

        Returns
        -------
        dict with keys:
            "y_true"     — list of actual values
            "y_pred"     — list of predicted values
            "residuals"  — list of (y_true - y_pred)
        """
        try:
            residuals = (y_true - y_pred).tolist()
            return {
                "y_true": y_true.tolist(),
                "y_pred": y_pred.tolist(),
                "residuals": residuals,
            }
        except Exception as exc:
            logger.warning(
                "VisualizationDataBuilder: residual_data failed — %s", exc
            )
            return {}

    def build_feature_importance(
        self,
        model: Any,
        feature_names: list[str] | None,
    ) -> dict[str, Any]:
        """
        Build feature importance data if the model exposes it.

        Works with any model that has a `feature_importances_` attribute
        (Random Forest, Gradient Boosting, XGBoost, LightGBM, etc.).
        Returns an empty dict for models that don't support this
        (Logistic Regression, SVM, KNN, etc.).

        Parameters
        ----------
        model : Any
            Fitted sklearn-compatible estimator.
        feature_names : list[str] | None
            Feature column names. Falls back to 'feature_0', 'feature_1'
            if None.

        Returns
        -------
        dict with keys:
            "features"    — list of feature name strings
            "importances" — list of importance values (same order)
        """
        try:
            importances = getattr(model, "feature_importances_", None)
            if importances is None:
                return {}

            importances_list = np.array(importances).tolist()
            n = len(importances_list)

            if feature_names is None or len(feature_names) != n:
                names = [f"feature_{i}" for i in range(n)]
            else:
                names = list(feature_names)

            # Sort by importance descending for readability
            paired = sorted(
                zip(names, importances_list),
                key=lambda x: x[1],
                reverse=True,
            )
            sorted_names, sorted_importances = zip(*paired) if paired else ([], [])

            return {
                "features": list(sorted_names),
                "importances": list(sorted_importances),
            }
        except Exception as exc:
            logger.warning(
                "VisualizationDataBuilder: feature_importance failed — %s", exc
            )
            return {}
