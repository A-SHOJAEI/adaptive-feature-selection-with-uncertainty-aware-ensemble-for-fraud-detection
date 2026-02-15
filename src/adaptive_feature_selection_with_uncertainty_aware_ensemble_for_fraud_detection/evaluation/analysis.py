"""Results analysis and visualization utilities."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)


class ResultsAnalyzer:
    """Analyzes and visualizes fraud detection results."""

    def __init__(self, output_dir: Path) -> None:
        """Initialize the results analyzer.

        Args:
            output_dir: Directory for saving analysis outputs.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set plotting style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)

    def plot_roc_curve(
        self,
        fpr: np.ndarray,
        tpr: np.ndarray,
        roc_auc: float,
        save_name: str = "roc_curve.png"
    ) -> None:
        """Plot ROC curve.

        Args:
            fpr: False positive rates.
            tpr: True positive rates.
            roc_auc: ROC-AUC score.
            save_name: Filename for saving the plot.
        """
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"ROC curve saved to {save_path}")

    def plot_precision_recall_curve(
        self,
        precision: np.ndarray,
        recall: np.ndarray,
        pr_auc: float,
        save_name: str = "pr_curve.png"
    ) -> None:
        """Plot precision-recall curve.

        Args:
            precision: Precision values.
            recall: Recall values.
            pr_auc: PR-AUC score.
            save_name: Filename for saving the plot.
        """
        plt.figure(figsize=(8, 8))
        plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.3f})', linewidth=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14)
        plt.legend(loc="lower left", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"PR curve saved to {save_path}")

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        save_name: str = "confusion_matrix.png"
    ) -> None:
        """Plot confusion matrix heatmap.

        Args:
            cm: Confusion matrix.
            save_name: Filename for saving the plot.
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Legitimate', 'Fraud'],
            yticklabels=['Legitimate', 'Fraud'],
            cbar_kws={'label': 'Count'}
        )
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title('Confusion Matrix', fontsize=14)
        plt.tight_layout()

        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Confusion matrix saved to {save_path}")

    def plot_feature_importance(
        self,
        feature_importance: np.ndarray,
        feature_names: Optional[list] = None,
        top_k: int = 20,
        save_name: str = "feature_importance.png"
    ) -> None:
        """Plot top feature importance scores.

        Args:
            feature_importance: Feature importance values.
            feature_names: Optional feature names.
            top_k: Number of top features to plot.
            save_name: Filename for saving the plot.
        """
        # Get top k features
        if len(feature_importance) > top_k:
            top_indices = np.argsort(feature_importance)[-top_k:]
        else:
            top_indices = np.argsort(feature_importance)

        top_scores = feature_importance[top_indices]

        if feature_names is not None:
            top_names = [feature_names[i] for i in top_indices]
        else:
            top_names = [f"Feature {i}" for i in top_indices]

        # Plot
        plt.figure(figsize=(10, max(6, len(top_indices) * 0.3)))
        y_pos = np.arange(len(top_names))
        plt.barh(y_pos, top_scores, color='steelblue')
        plt.yticks(y_pos, top_names, fontsize=9)
        plt.xlabel('Importance Score', fontsize=12)
        plt.title(f'Top {len(top_indices)} Feature Importance', fontsize=14)
        plt.tight_layout()

        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Feature importance plot saved to {save_path}")

    def plot_threshold_analysis(
        self,
        threshold_metrics: list,
        save_name: str = "threshold_analysis.png"
    ) -> None:
        """Plot metrics vs decision threshold.

        Args:
            threshold_metrics: List of dicts with threshold, precision, recall, f1.
            save_name: Filename for saving the plot.
        """
        thresholds = [m['threshold'] for m in threshold_metrics]
        precisions = [m['precision'] for m in threshold_metrics]
        recalls = [m['recall'] for m in threshold_metrics]
        f1_scores = [m['f1_score'] for m in threshold_metrics]

        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, precisions, label='Precision', linewidth=2, marker='o')
        plt.plot(thresholds, recalls, label='Recall', linewidth=2, marker='s')
        plt.plot(thresholds, f1_scores, label='F1 Score', linewidth=2, marker='^')
        plt.xlabel('Decision Threshold', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Metrics vs Decision Threshold', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Threshold analysis saved to {save_path}")

    def save_metrics_json(
        self,
        metrics: Dict[str, Any],
        filename: str = "metrics.json"
    ) -> None:
        """Save metrics to JSON file.

        Args:
            metrics: Dictionary of metrics.
            filename: Output filename.
        """
        save_path = self.output_dir / filename
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {save_path}")

    def generate_full_report(
        self,
        metrics: Dict[str, Any],
        threshold_metrics: Optional[Dict[str, Any]] = None,
        feature_importance: Optional[np.ndarray] = None,
        feature_names: Optional[list] = None
    ) -> None:
        """Generate a complete analysis report.

        Args:
            metrics: Dictionary of evaluation metrics.
            threshold_metrics: Optional threshold-based metrics.
            feature_importance: Optional feature importance scores.
            feature_names: Optional feature names.
        """
        logger.info("Generating full analysis report...")

        # Save metrics JSON
        self.save_metrics_json(metrics, "evaluation_metrics.json")

        # Plot ROC curve
        if "roc" in metrics:
            self.plot_roc_curve(
                np.array(metrics["roc"]["fpr"]),
                np.array(metrics["roc"]["tpr"]),
                metrics.get("roc_auc", 0.0)
            )

        # Plot PR curve
        if "pr" in metrics:
            self.plot_precision_recall_curve(
                np.array(metrics["pr"]["precision"]),
                np.array(metrics["pr"]["recall"]),
                metrics.get("pr_auc", 0.0)
            )

        # Plot confusion matrix
        if all(k in metrics for k in ["true_positives", "false_positives", "true_negatives", "false_negatives"]):
            cm = np.array([
                [metrics["true_negatives"], metrics["false_positives"]],
                [metrics["false_negatives"], metrics["true_positives"]]
            ])
            self.plot_confusion_matrix(cm)

        # Plot threshold analysis
        if threshold_metrics and "custom_thresholds" in threshold_metrics:
            self.plot_threshold_analysis(threshold_metrics["custom_thresholds"])

        # Plot feature importance
        if feature_importance is not None:
            self.plot_feature_importance(feature_importance, feature_names)

        logger.info(f"Full report generated in {self.output_dir}")
