"""Data loading utilities for fraud detection datasets."""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class FraudDataLoader:
    """Loads and manages fraud detection datasets.

    For demonstration purposes, this generates synthetic fraud data with
    similar characteristics to the IEEE fraud detection dataset when the
    actual dataset is not available.
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        download: bool = True,
        random_state: int = 42
    ) -> None:
        """Initialize the data loader.

        Args:
            data_dir: Directory containing the dataset. If None, generates synthetic data.
            download: Whether to download the dataset if not found.
            random_state: Random seed for reproducibility.
        """
        self.data_dir = data_dir
        self.download = download
        self.random_state = random_state

    def load_data(
        self,
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Load and split fraud detection data.

        Args:
            test_size: Proportion of data for test set.
            val_size: Proportion of training data for validation set.

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test).
        """
        logger.info("Loading fraud detection dataset...")

        # Try to load real data if available, otherwise generate synthetic
        if self.data_dir and (Path(self.data_dir) / "train_transaction.csv").exists():
            X, y = self._load_ieee_fraud_data()
        else:
            logger.warning(
                "IEEE fraud dataset not found. Generating synthetic data with similar characteristics."
            )
            X, y = self._generate_synthetic_fraud_data()

        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        # Second split: separate validation from training
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted,
            random_state=self.random_state, stratify=y_temp
        )

        logger.info(f"Data loaded: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
        logger.info(f"Fraud rate - train: {y_train.mean():.3f}, val: {y_val.mean():.3f}, test: {y_test.mean():.3f}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def _load_ieee_fraud_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load the real IEEE fraud detection dataset.

        Returns:
            Tuple of (features, labels).
        """
        logger.info("Loading IEEE fraud detection dataset from disk...")

        # Load transaction data
        train_df = pd.read_csv(Path(self.data_dir) / "train_transaction.csv")

        # Separate features and target
        y = train_df["isFraud"]
        X = train_df.drop(["isFraud", "TransactionID"], axis=1, errors="ignore")

        # Load identity data if available and merge
        identity_path = Path(self.data_dir) / "train_identity.csv"
        if identity_path.exists():
            identity_df = pd.read_csv(identity_path)
            X = X.merge(identity_df, on="TransactionID", how="left")

        return X, y

    def _generate_synthetic_fraud_data(
        self,
        n_samples: int = 50000,
        n_features: int = 100,
        fraud_rate: float = 0.035
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate synthetic fraud detection data.

        Args:
            n_samples: Number of samples to generate.
            n_features: Number of features to generate.
            fraud_rate: Proportion of fraudulent transactions.

        Returns:
            Tuple of (features, labels).
        """
        logger.info(
            f"Generating synthetic fraud data: {n_samples} samples, "
            f"{n_features} features, {fraud_rate:.1%} fraud rate"
        )

        # Create imbalanced classification problem
        n_informative = max(20, n_features // 5)
        n_redundant = max(10, n_features // 10)

        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_clusters_per_class=3,
            weights=[1 - fraud_rate, fraud_rate],
            flip_y=0.02,  # Add some label noise
            random_state=self.random_state
        )

        # Convert to DataFrame with realistic feature names
        feature_names = (
            [f"TransactionAmt_feature_{i}" for i in range(5)] +
            [f"ProductCD_feature_{i}" for i in range(5)] +
            [f"card_feature_{i}" for i in range(10)] +
            [f"addr_feature_{i}" for i in range(10)] +
            [f"dist_feature_{i}" for i in range(10)] +
            [f"email_feature_{i}" for i in range(10)] +
            [f"DeviceInfo_feature_{i}" for i in range(10)] +
            [f"V_feature_{i}" for i in range(n_features - 60)]
        )

        X_df = pd.DataFrame(X, columns=feature_names[:n_features])
        y_series = pd.Series(y, name="isFraud")

        # Add some categorical-like features (will be float but represent categories)
        for i in range(5):
            col = f"categorical_{i}"
            X_df[col] = np.random.randint(0, 10, size=n_samples).astype(float)

        # Add some missing values to simulate real data
        mask = np.random.random(X_df.shape) < 0.05
        X_df = X_df.mask(mask)

        return X_df, y_series
