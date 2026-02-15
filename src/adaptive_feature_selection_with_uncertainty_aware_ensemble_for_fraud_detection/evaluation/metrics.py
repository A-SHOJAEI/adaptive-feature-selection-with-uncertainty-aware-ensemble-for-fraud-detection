"""Evaluation metrics for fraud detection."""

import logging
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve
)

logger = logging.getLogger(__name__)


class FraudMetrics:
    """Comprehensive metrics for fraud detection evaluation."""

    @staticmethod
    def compute_all_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        feature_mask: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Compute all evaluation metrics.

        Args:
            y_true: True binary labels.
            y_pred: Predicted binary labels.
            y_proba: Predicted probabilities (for AUC metrics).
            feature_mask: Optional binary mask of selected features.

        Returns:
            Dictionary of metric names to values.
        """
        metrics = {}

        # Classification metrics
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
        metrics["f1_score"] = f1_score(y_true, y_pred, zero_division=0)

        # Confusion matrix metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics["true_positives"] = int(tp)
        metrics["false_positives"] = int(fp)
        metrics["true_negatives"] = int(tn)
        metrics["false_negatives"] = int(fn)
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # AUC metrics (if probabilities provided)
        if y_proba is not None:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
                metrics["pr_auc"] = average_precision_score(y_true, y_proba)
            except ValueError as e:
                logger.warning(f"Could not compute AUC metrics: {e}")
                metrics["roc_auc"] = 0.0
                metrics["pr_auc"] = 0.0

        # Feature efficiency (if feature mask provided)
        if feature_mask is not None:
            metrics["features_selected"] = int(feature_mask.sum())
            metrics["feature_efficiency"] = float(feature_mask.sum() / len(feature_mask))
        else:
            metrics["features_selected"] = 0
            metrics["feature_efficiency"] = 1.0

        return metrics

    @staticmethod
    def compute_threshold_metrics(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        thresholds: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """Compute metrics at different decision thresholds.

        Args:
            y_true: True binary labels.
            y_proba: Predicted probabilities.
            thresholds: Optional custom thresholds to evaluate.

        Returns:
            Dictionary containing threshold-based metrics.
        """
        # ROC curve
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)

        # Precision-recall curve
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba)

        results = {
            "roc": {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": roc_thresholds.tolist()
            },
            "pr": {
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "thresholds": pr_thresholds.tolist()
            }
        }

        # Evaluate at custom thresholds if provided
        if thresholds is not None:
            threshold_results = []
            for threshold in thresholds:
                y_pred = (y_proba >= threshold).astype(int)
                threshold_results.append({
                    "threshold": float(threshold),
                    "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                    "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                    "f1_score": float(f1_score(y_true, y_pred, zero_division=0))
                })
            results["custom_thresholds"] = threshold_results

        return results

    @staticmethod
    def compute_per_class_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Compute metrics separately for each class.

        Args:
            y_true: True binary labels.
            y_pred: Predicted binary labels.

        Returns:
            Dictionary with per-class metrics.
        """
        # Class 0 (legitimate transactions)
        class_0_mask = y_true == 0
        if class_0_mask.sum() > 0:
            class_0_metrics = {
                "support": int(class_0_mask.sum()),
                "precision": float(precision_score(
                    y_true[class_0_mask],
                    y_pred[class_0_mask],
                    pos_label=0,
                    zero_division=0
                )),
                "recall": float(recall_score(
                    y_true[class_0_mask],
                    y_pred[class_0_mask],
                    pos_label=0,
                    zero_division=0
                )),
                "f1_score": float(f1_score(
                    y_true[class_0_mask],
                    y_pred[class_0_mask],
                    pos_label=0,
                    zero_division=0
                ))
            }
        else:
            class_0_metrics = {"support": 0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0}

        # Class 1 (fraudulent transactions)
        class_1_mask = y_true == 1
        if class_1_mask.sum() > 0:
            class_1_metrics = {
                "support": int(class_1_mask.sum()),
                "precision": float(precision_score(y_true[class_1_mask], y_pred[class_1_mask], zero_division=0)),
                "recall": float(recall_score(y_true[class_1_mask], y_pred[class_1_mask], zero_division=0)),
                "f1_score": float(f1_score(y_true[class_1_mask], y_pred[class_1_mask], zero_division=0))
            }
        else:
            class_1_metrics = {"support": 0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0}

        return {
            "legitimate": class_0_metrics,
            "fraud": class_1_metrics
        }

    @staticmethod
    def print_metrics_summary(metrics: Dict[str, float]) -> None:
        """Print a formatted summary of metrics.

        Args:
            metrics: Dictionary of metric names to values.
        """
        print("\n" + "=" * 60)
        print("FRAUD DETECTION METRICS SUMMARY")
        print("=" * 60)

        # Main metrics
        print("\nMain Metrics:")
        print(f"  ROC-AUC:          {metrics.get('roc_auc', 0):.4f}")
        print(f"  PR-AUC:           {metrics.get('pr_auc', 0):.4f}")
        print(f"  F1 Score:         {metrics.get('f1_score', 0):.4f}")
        print(f"  Accuracy:         {metrics.get('accuracy', 0):.4f}")

        # Precision/Recall
        print("\nPrecision & Recall:")
        print(f"  Precision:        {metrics.get('precision', 0):.4f}")
        print(f"  Recall:           {metrics.get('recall', 0):.4f}")
        print(f"  Specificity:      {metrics.get('specificity', 0):.4f}")

        # Confusion matrix
        print("\nConfusion Matrix:")
        print(f"  True Positives:   {metrics.get('true_positives', 0)}")
        print(f"  False Positives:  {metrics.get('false_positives', 0)}")
        print(f"  True Negatives:   {metrics.get('true_negatives', 0)}")
        print(f"  False Negatives:  {metrics.get('false_negatives', 0)}")

        # Feature efficiency
        if "feature_efficiency" in metrics:
            print("\nFeature Selection:")
            print(f"  Features Used:    {metrics.get('features_selected', 0)}")
            print(f"  Feature Efficiency: {metrics.get('feature_efficiency', 1):.2%}")

        print("=" * 60 + "\n")
