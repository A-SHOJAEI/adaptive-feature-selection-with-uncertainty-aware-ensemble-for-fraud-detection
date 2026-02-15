#!/usr/bin/env python
"""Training script for adaptive uncertainty fraud detection.

This script trains the full model with:
- Adaptive feature selection via RL
- Uncertainty-aware ensemble voting
- Focal-uncertainty loss for imbalanced learning
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root and src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adaptive_feature_selection_with_uncertainty_aware_ensemble_for_fraud_detection.training import (
    FraudDetectionTrainer
)
from adaptive_feature_selection_with_uncertainty_aware_ensemble_for_fraud_detection.utils import (
    load_config,
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
        description="Train adaptive uncertainty fraud detection model"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory for saving outputs"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    parser.add_argument(
        "--use-mlflow",
        action="store_true",
        help="Enable MLflow tracking"
    )

    return parser.parse_args()


def main() -> None:
    """Main training function."""
    # Parse arguments
    args = parse_args()

    # Setup logging
    setup_logging(args.log_level)

    logger.info("Starting fraud detection training pipeline")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Random seed: {args.seed}")

    # Set random seeds for reproducibility
    set_random_seeds(args.seed)

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    # Optional MLflow tracking
    mlflow_run = None
    if args.use_mlflow:
        try:
            import mlflow
            mlflow.set_experiment("fraud_detection")
            mlflow_run = mlflow.start_run()
            mlflow.log_params({
                "config_file": args.config,
                "random_seed": args.seed,
                **config.get("model", {})
            })
            logger.info("MLflow tracking enabled")
        except Exception as e:
            logger.warning(f"MLflow tracking not available: {e}")
            mlflow_run = None

    try:
        # Initialize trainer
        trainer = FraudDetectionTrainer(
            config=config,
            output_dir=Path(args.output_dir),
            random_state=args.seed
        )

        # Run training
        results = trainer.train()

        # Log to MLflow if enabled
        if mlflow_run is not None:
            try:
                mlflow.log_artifact(str(Path(args.output_dir) / "results" / "training_results.json"))
                logger.info("Results logged to MLflow")
            except Exception as e:
                logger.warning(f"Failed to log to MLflow: {e}")

        logger.info("Training completed successfully!")
        logger.info(f"Model saved to: {results['model_path']}")
        logger.info(f"Preprocessor saved to: {results['preprocessor_path']}")

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if mlflow_run is not None:
            try:
                mlflow.end_run()
            except Exception:
                pass


if __name__ == "__main__":
    main()
