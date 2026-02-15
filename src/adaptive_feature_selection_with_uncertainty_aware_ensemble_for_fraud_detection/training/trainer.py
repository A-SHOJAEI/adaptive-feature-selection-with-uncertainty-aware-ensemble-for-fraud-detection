"""Training loop with early stopping and model checkpointing."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ..data import FraudDataLoader, FraudPreprocessor
from ..models import AdaptiveUncertaintyFraudDetector

logger = logging.getLogger(__name__)


class FraudDetectionTrainer:
    """Trainer for fraud detection models with checkpointing and early stopping."""

    def __init__(
        self,
        config: Dict[str, Any],
        output_dir: Path,
        random_state: int = 42
    ) -> None:
        """Initialize the trainer.

        Args:
            config: Configuration dictionary.
            output_dir: Directory for saving outputs.
            random_state: Random seed for reproducibility.
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.random_state = random_state

        # Create output directories
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.results_dir = self.output_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Components
        self.data_loader: Optional[FraudDataLoader] = None
        self.preprocessor: Optional[FraudPreprocessor] = None
        self.model: Optional[AdaptiveUncertaintyFraudDetector] = None

        # Training state
        self.best_val_score: float = -np.inf
        self.training_history: Dict[str, list] = {
            "val_roc_auc": [],
            "val_pr_auc": [],
            "val_f1": []
        }

    def train(self) -> Dict[str, Any]:
        """Run the full training pipeline.

        Returns:
            Dictionary containing training results and metrics.
        """
        logger.info("=" * 80)
        logger.info("Starting fraud detection training pipeline")
        logger.info("=" * 80)

        # 1. Load and preprocess data
        logger.info("\nStep 1: Loading and preprocessing data...")
        X_train, X_val, X_test, y_train, y_val, y_test = self._load_data()

        # 2. Initialize model
        logger.info("\nStep 2: Initializing model...")
        self.model = self._create_model()

        # 3. Train model
        logger.info("\nStep 3: Training model...")
        self.model.fit(
            X_train, y_train,
            X_val, y_val,
            verbose=True
        )

        # 4. Save best model
        logger.info("\nStep 4: Saving best model...")
        model_path = self.checkpoint_dir / "best_model.pkl"
        self.model.save(model_path)

        # 5. Save preprocessor
        preprocessor_path = self.checkpoint_dir / "preprocessor.pkl"
        import joblib
        joblib.dump(self.preprocessor, preprocessor_path)
        logger.info(f"Preprocessor saved to {preprocessor_path}")

        # 6. Collect results
        results = {
            "model_path": str(model_path),
            "preprocessor_path": str(preprocessor_path),
            "config": self.config,
            "data_stats": {
                "n_train": len(X_train),
                "n_val": len(X_val),
                "n_test": len(X_test),
                "n_features": X_train.shape[1],
                "fraud_rate_train": float(y_train.mean()),
                "fraud_rate_val": float(y_val.mean()),
                "fraud_rate_test": float(y_test.mean())
            }
        }

        # Save results
        results_path = self.results_dir / "training_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Training results saved to {results_path}")

        logger.info("\n" + "=" * 80)
        logger.info("Training pipeline complete!")
        logger.info("=" * 80)

        return results

    def _load_data(self) -> tuple:
        """Load and preprocess data.

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test).
        """
        # Load raw data
        data_config = self.config.get("data", {})
        self.data_loader = FraudDataLoader(
            data_dir=data_config.get("data_dir"),
            random_state=self.random_state
        )

        X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = \
            self.data_loader.load_data(
                test_size=data_config.get("test_size", 0.2),
                val_size=data_config.get("val_size", 0.1)
            )

        # Preprocess
        preprocess_config = self.config.get("preprocessing", {})
        self.preprocessor = FraudPreprocessor(
            numerical_strategy=preprocess_config.get("numerical_strategy", "median"),
            categorical_strategy=preprocess_config.get("categorical_strategy", "most_frequent"),
            scale=preprocess_config.get("scale", True)
        )

        X_train = self.preprocessor.fit_transform(X_train_raw)
        X_val = self.preprocessor.transform(X_val_raw)
        X_test = self.preprocessor.transform(X_test_raw)

        logger.info(f"Preprocessed data shapes: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

        return X_train, X_val, X_test, y_train.values, y_val.values, y_test.values

    def _create_model(self) -> AdaptiveUncertaintyFraudDetector:
        """Create model from configuration.

        Returns:
            Initialized model instance.
        """
        model_config = self.config.get("model", {})

        model = AdaptiveUncertaintyFraudDetector(
            feature_budget=model_config.get("feature_budget", 0.6),
            focal_alpha=model_config.get("focal_alpha", 0.25),
            focal_gamma=model_config.get("focal_gamma", 2.0),
            uncertainty_weight=model_config.get("uncertainty_weight", 0.1),
            n_estimators=model_config.get("n_estimators", 500),
            learning_rate=model_config.get("learning_rate", 0.05),
            max_depth=model_config.get("max_depth", 6),
            use_feature_selection=model_config.get("use_feature_selection", True),
            use_uncertainty_weighting=model_config.get("use_uncertainty_weighting", True),
            random_state=self.random_state
        )

        logger.info("Model configuration:")
        for key, value in model_config.items():
            logger.info(f"  {key}: {value}")

        return model
