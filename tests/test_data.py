"""Tests for data loading and preprocessing."""

import numpy as np
import pandas as pd
import pytest

from adaptive_feature_selection_with_uncertainty_aware_ensemble_for_fraud_detection.data import (
    FraudDataLoader,
    FraudPreprocessor
)


class TestFraudDataLoader:
    """Tests for FraudDataLoader."""

    def test_init(self, random_state: int) -> None:
        """Test initialization."""
        loader = FraudDataLoader(random_state=random_state)
        assert loader.random_state == random_state
        assert loader.data_dir is None

    def test_load_synthetic_data(self, random_state: int) -> None:
        """Test loading synthetic data."""
        loader = FraudDataLoader(random_state=random_state)
        X_train, X_val, X_test, y_train, y_val, y_test = loader.load_data()

        # Check shapes
        assert len(X_train) > 0
        assert len(X_val) > 0
        assert len(X_test) > 0
        assert len(X_train) == len(y_train)
        assert len(X_val) == len(y_val)
        assert len(X_test) == len(y_test)

        # Check feature consistency
        assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1]

        # Check labels are binary
        assert set(y_train.unique()).issubset({0, 1})
        assert set(y_val.unique()).issubset({0, 1})
        assert set(y_test.unique()).issubset({0, 1})

    def test_fraud_rate(self, random_state: int) -> None:
        """Test that fraud rate is approximately correct."""
        loader = FraudDataLoader(random_state=random_state)
        _, _, _, y_train, _, _ = loader.load_data()

        fraud_rate = y_train.mean()
        assert 0.01 < fraud_rate < 0.1  # Should be imbalanced

    def test_split_sizes(self, random_state: int) -> None:
        """Test that split sizes are correct."""
        loader = FraudDataLoader(random_state=random_state)
        X_train, X_val, X_test, _, _, _ = loader.load_data(
            test_size=0.2, val_size=0.1
        )

        total = len(X_train) + len(X_val) + len(X_test)
        test_ratio = len(X_test) / total
        val_ratio = len(X_val) / total

        # Allow some tolerance due to stratification
        assert 0.15 < test_ratio < 0.25
        assert 0.05 < val_ratio < 0.15


class TestFraudPreprocessor:
    """Tests for FraudPreprocessor."""

    def test_init(self) -> None:
        """Test initialization."""
        preprocessor = FraudPreprocessor()
        assert preprocessor.numerical_strategy == "median"
        assert preprocessor.scale is True

    def test_fit_transform(self, synthetic_fraud_data: tuple) -> None:
        """Test fit and transform."""
        X, _ = synthetic_fraud_data
        preprocessor = FraudPreprocessor()

        X_transformed = preprocessor.fit_transform(X)

        assert isinstance(X_transformed, np.ndarray)
        assert X_transformed.shape[0] == len(X)
        assert X_transformed.shape[1] > 0

    def test_transform_consistency(self, synthetic_fraud_data: tuple) -> None:
        """Test that transform is consistent."""
        X, _ = synthetic_fraud_data
        preprocessor = FraudPreprocessor()

        X_transformed_1 = preprocessor.fit_transform(X)
        X_transformed_2 = preprocessor.transform(X)

        np.testing.assert_array_almost_equal(X_transformed_1, X_transformed_2)

    def test_handles_missing_values(self, random_state: int) -> None:
        """Test handling of missing values."""
        # Create data with missing values
        X = pd.DataFrame({
            'a': [1, 2, np.nan, 4, 5],
            'b': [np.nan, 2, 3, 4, 5],
            'c': [1, 2, 3, 4, 5]
        })

        preprocessor = FraudPreprocessor()
        X_transformed = preprocessor.fit_transform(X)

        # No NaN in output
        assert not np.isnan(X_transformed).any()

    def test_scaling(self, synthetic_fraud_data: tuple) -> None:
        """Test that scaling is applied."""
        X, _ = synthetic_fraud_data
        preprocessor = FraudPreprocessor(scale=True)

        X_transformed = preprocessor.fit_transform(X)

        # Check that data is roughly centered and scaled
        # (RobustScaler may not exactly center at 0)
        assert np.abs(np.median(X_transformed, axis=0)).max() < 5

    def test_no_scaling(self, synthetic_fraud_data: tuple) -> None:
        """Test without scaling."""
        X, _ = synthetic_fraud_data
        preprocessor = FraudPreprocessor(scale=False)

        X_transformed = preprocessor.fit_transform(X)

        # Output should still be numeric
        assert X_transformed.dtype in [np.float32, np.float64]

    def test_get_feature_names(self, synthetic_fraud_data: tuple) -> None:
        """Test getting feature names."""
        X, _ = synthetic_fraud_data
        preprocessor = FraudPreprocessor()

        preprocessor.fit(X)
        feature_names = preprocessor.get_feature_names()

        assert len(feature_names) > 0
        assert all(isinstance(name, str) for name in feature_names)
