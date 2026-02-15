#!/usr/bin/env python
"""Evaluation script for fraud detection model.

This script:
- Loads a trained model from checkpoint
- Evaluates on test/validation set
- Computes comprehensive metrics
- Generates analysis visualizations
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np

# Add project root and src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adaptive_feature_selection_with_uncertainty_aware_ensemble_for_fraud_detection.data import (
    FraudDataLoader
)
from adaptive_feature_selection_with_uncertainty_aware_ensemble_for_fraud_detection.evaluation import (
    FraudMetrics,
    ResultsAnalyzer
)
from adaptive_feature_selection_with_uncertainty_aware_ensemble_for_fraud_detection.models import (
    AdaptiveUncertaintyFraudDetector
)
from adaptive_feature_selection_with_uncertainty_aware_ensemble_for_fraud_detection.utils import (
    set_random_seeds
)

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration.

    Args:
        log_level: Logging level.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler()]
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate fraud detection model"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="models/checkpoints/best_model.pkl",
        help="Path to trained model checkpoint"
    )

    parser.add_argument(
        "--preprocessor-path",
        type=str,
        default="models/checkpoints/preprocessor.pkl",
        help="Path to fitted preprocessor"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory for saving evaluation results"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val"],
        help="Which split to evaluate on"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    return parser.parse_args()


def main() -> None:
    """Main evaluation function."""
    # Parse arguments
    args = parse_args()

    # Setup logging
    setup_logging(args.log_level)

    logger.info("Starting fraud detection evaluation")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Output directory: {args.output_dir}")

    # Set random seeds
    set_random_seeds(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load model
        logger.info("Loading trained model...")
        model = AdaptiveUncertaintyFraudDetector.load(args.model_path)

        # Load preprocessor
        logger.info("Loading preprocessor...")
        preprocessor = joblib.load(args.preprocessor_path)

        # Load data
        logger.info("Loading evaluation data...")
        data_loader = FraudDataLoader(random_state=args.seed)
        X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = data_loader.load_data()

        # Preprocess
        X_train = preprocessor.transform(X_train_raw)
        X_val = preprocessor.transform(X_val_raw)
        X_test = preprocessor.transform(X_test_raw)

        # Select evaluation split
        if args.split == "test":
            X_eval = X_test
            y_eval = y_test.values
            split_name = "Test"
        else:
            X_eval = X_val
            y_eval = y_val.values
            split_name = "Validation"

        logger.info(f"Evaluating on {split_name} set: {len(X_eval)} samples")

        # Get predictions
        logger.info("Generating predictions...")
        y_pred_proba_full = model.predict_proba(X_eval)
        y_pred_proba = y_pred_proba_full[:, 1]  # Fraud probability
        y_pred = model.predict(X_eval)

        # Compute metrics
        logger.info("Computing metrics...")
        metrics = FraudMetrics.compute_all_metrics(
            y_eval,
            y_pred,
            y_pred_proba,
            feature_mask=model.feature_mask_
        )

        # Print summary
        FraudMetrics.print_metrics_summary(metrics)

        # Compute threshold-based metrics
        threshold_metrics = FraudMetrics.compute_threshold_metrics(
            y_eval,
            y_pred_proba,
            thresholds=np.arange(0.1, 1.0, 0.1)
        )

        # Compute per-class metrics
        per_class_metrics = FraudMetrics.compute_per_class_metrics(y_eval, y_pred)

        # Combine all metrics
        all_metrics = {
            **metrics,
            **threshold_metrics,
            "per_class": per_class_metrics
        }

        # Save metrics JSON
        metrics_path = output_dir / "evaluation_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")

        # Generate visualizations
        logger.info("Generating analysis visualizations...")
        analyzer = ResultsAnalyzer(output_dir)

        # Get feature importance if available
        feature_importance = model.get_feature_importance()
        feature_names = preprocessor.get_feature_names()

        # Generate full report
        analyzer.generate_full_report(
            all_metrics,
            threshold_metrics,
            feature_importance,
            feature_names
        )

        logger.info("Evaluation completed successfully!")

        # Create summary table
        print("\n" + "=" * 80)
        print("SUMMARY RESULTS")
        print("=" * 80)
        print(f"{'Metric':<25} {'Value':>15}")
        print("-" * 80)
        print(f"{'ROC-AUC':<25} {metrics.get('roc_auc', 0):>15.4f}")
        print(f"{'PR-AUC':<25} {metrics.get('pr_auc', 0):>15.4f}")
        print(f"{'F1 Score':<25} {metrics.get('f1_score', 0):>15.4f}")
        print(f"{'Feature Efficiency':<25} {metrics.get('feature_efficiency', 1):>15.2%}")
        print("=" * 80 + "\n")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.error("Please train the model first using: python scripts/train.py")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
