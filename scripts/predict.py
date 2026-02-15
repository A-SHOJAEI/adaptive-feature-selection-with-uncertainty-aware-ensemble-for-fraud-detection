#!/usr/bin/env python
"""Prediction script for fraud detection.

This script loads a trained model and performs inference on new data.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Add project root and src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adaptive_feature_selection_with_uncertainty_aware_ensemble_for_fraud_detection.models import (
    AdaptiveUncertaintyFraudDetector
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
        description="Predict fraud using trained model"
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
        "--input",
        type=str,
        required=True,
        help="Path to input CSV file or '-' for stdin"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output CSV file (default: stdout)"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for fraud classification"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    return parser.parse_args()


def load_input_data(input_path: str) -> pd.DataFrame:
    """Load input data from file or stdin.

    Args:
        input_path: Path to input file or '-' for stdin.

    Returns:
        DataFrame with input features.
    """
    if input_path == "-":
        logger.info("Reading input from stdin...")
        data = pd.read_csv(sys.stdin)
    else:
        logger.info(f"Reading input from {input_path}...")
        data = pd.read_csv(input_path)

    logger.info(f"Loaded {len(data)} samples")
    return data


def main() -> None:
    """Main prediction function."""
    # Parse arguments
    args = parse_args()

    # Setup logging
    setup_logging(args.log_level)

    logger.info("Starting fraud detection prediction")

    try:
        # Load model
        logger.info(f"Loading model from {args.model_path}...")
        model = AdaptiveUncertaintyFraudDetector.load(args.model_path)

        # Load preprocessor
        logger.info(f"Loading preprocessor from {args.preprocessor_path}...")
        preprocessor = joblib.load(args.preprocessor_path)

        # Load input data
        data = load_input_data(args.input)

        # Preprocess
        logger.info("Preprocessing input data...")
        X = preprocessor.transform(data)

        # Get predictions
        logger.info("Generating predictions...")
        y_pred_proba_full = model.predict_proba(X)
        y_pred_proba = y_pred_proba_full[:, 1]  # Fraud probability

        # Apply threshold
        y_pred = (y_pred_proba >= args.threshold).astype(int)

        # Also get uncertainty estimates if available
        try:
            y_pred_proba_with_unc, uncertainty = model.predict_proba(
                X, return_uncertainty=True
            )
            has_uncertainty = True
        except Exception:
            uncertainty = None
            has_uncertainty = False

        # Create output DataFrame
        output_data = pd.DataFrame({
            "prediction": y_pred,
            "fraud_probability": y_pred_proba,
            "fraud_score": y_pred_proba
        })

        if has_uncertainty and uncertainty is not None:
            output_data["uncertainty"] = uncertainty

        # Add confidence (inverse of uncertainty or direct from probability)
        if has_uncertainty and uncertainty is not None:
            output_data["confidence"] = 1 - uncertainty
        else:
            # Use probability distance from decision boundary as confidence
            output_data["confidence"] = np.abs(y_pred_proba - 0.5) * 2

        # Add interpretation
        output_data["prediction_label"] = output_data["prediction"].map(
            {0: "LEGITIMATE", 1: "FRAUD"}
        )

        # Save or print output
        if args.output:
            logger.info(f"Saving predictions to {args.output}...")
            output_data.to_csv(args.output, index=False)
            logger.info("Predictions saved successfully")
        else:
            # Print to stdout
            print("\n" + "=" * 80)
            print("FRAUD DETECTION PREDICTIONS")
            print("=" * 80)
            print(output_data.to_string(index=False))
            print("=" * 80 + "\n")

        # Print summary statistics
        n_fraud = y_pred.sum()
        n_total = len(y_pred)
        avg_prob = y_pred_proba.mean()

        logger.info("\nPrediction Summary:")
        logger.info(f"  Total samples:     {n_total}")
        logger.info(f"  Predicted fraud:   {n_fraud} ({n_fraud/n_total:.1%})")
        logger.info(f"  Avg fraud prob:    {avg_prob:.3f}")

        if has_uncertainty and uncertainty is not None:
            avg_uncertainty = uncertainty.mean()
            logger.info(f"  Avg uncertainty:   {avg_uncertainty:.3f}")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.error("Please ensure model and preprocessor are trained and paths are correct")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
