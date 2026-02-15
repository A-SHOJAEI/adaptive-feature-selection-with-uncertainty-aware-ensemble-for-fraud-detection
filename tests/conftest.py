"""Pytest fixtures for fraud detection tests."""

import sys
from pathlib import Path

# Add project root and src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification


@pytest.fixture
def random_state() -> int:
    """Random state for reproducibility.

    Returns:
        Random seed value.
    """
    return 42


@pytest.fixture
def synthetic_fraud_data(random_state: int) -> tuple:
    """Generate synthetic fraud detection data.

    Args:
        random_state: Random seed.

    Returns:
        Tuple of (X, y) where X is features and y is labels.
    """
    X, y = make_classification(
        n_samples=1000,
        n_features=50,
        n_informative=20,
        n_redundant=10,
        n_clusters_per_class=2,
        weights=[0.965, 0.035],  # 3.5% fraud rate
        flip_y=0.02,
        random_state=random_state
    )

    # Convert to DataFrame
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="isFraud")

    # Add some missing values
    mask = np.random.RandomState(random_state).random(X_df.shape) < 0.05
    X_df = X_df.mask(mask)

    return X_df, y_series


@pytest.fixture
def small_fraud_data(random_state: int) -> tuple:
    """Generate small synthetic fraud data for quick tests.

    Args:
        random_state: Random seed.

    Returns:
        Tuple of (X, y).
    """
    X, y = make_classification(
        n_samples=200,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        weights=[0.9, 0.1],
        random_state=random_state
    )

    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y)

    return X_df, y_series


@pytest.fixture
def sample_config() -> dict:
    """Sample configuration for testing.

    Returns:
        Configuration dictionary.
    """
    return {
        "data": {
            "test_size": 0.2,
            "val_size": 0.1
        },
        "preprocessing": {
            "numerical_strategy": "median",
            "categorical_strategy": "most_frequent",
            "scale": True
        },
        "model": {
            "feature_budget": 0.6,
            "focal_alpha": 0.25,
            "focal_gamma": 2.0,
            "uncertainty_weight": 0.1,
            "n_estimators": 50,  # Reduced for testing
            "learning_rate": 0.1,
            "max_depth": 4,
            "use_feature_selection": True,
            "use_uncertainty_weighting": True
        }
    }
