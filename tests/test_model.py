"""Tests for model components."""

import numpy as np
import pytest

from adaptive_feature_selection_with_uncertainty_aware_ensemble_for_fraud_detection.models import (
    AdaptiveUncertaintyFraudDetector,
    FeatureSelectionAgent,
    FocalUncertaintyLoss,
    TemperatureScaler,
    UncertaintyEnsemble
)
from adaptive_feature_selection_with_uncertainty_aware_ensemble_for_fraud_detection.data import (
    FraudPreprocessor
)


class TestFocalUncertaintyLoss:
    """Tests for FocalUncertaintyLoss."""

    def test_init(self) -> None:
        """Test initialization."""
        loss = FocalUncertaintyLoss(alpha=0.25, gamma=2.0, uncertainty_weight=0.1)
        assert loss.alpha == 0.25
        assert loss.gamma == 2.0
        assert loss.uncertainty_weight == 0.1

    def test_call(self) -> None:
        """Test loss computation."""
        loss = FocalUncertaintyLoss()

        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0.1, 0.9, 0.2, 0.8, 0.3])

        loss_value = loss(y_true, y_pred)

        assert isinstance(loss_value, float)
        assert loss_value > 0

    def test_with_uncertainty(self) -> None:
        """Test loss computation with uncertainty."""
        loss = FocalUncertaintyLoss(uncertainty_weight=0.1)

        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0.1, 0.9, 0.2, 0.8])
        uncertainty = np.array([0.1, 0.2, 0.15, 0.25])

        loss_with_unc = loss(y_true, y_pred, uncertainty)
        loss_without_unc = loss(y_true, y_pred, None)

        # Loss with uncertainty should be different
        assert loss_with_unc != loss_without_unc

    def test_gradient(self) -> None:
        """Test gradient computation."""
        loss = FocalUncertaintyLoss()

        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0.1, 0.9, 0.2, 0.8])

        grad = loss.gradient(y_true, y_pred)

        assert grad.shape == y_pred.shape
        assert not np.isnan(grad).any()

    def test_hessian(self) -> None:
        """Test hessian computation."""
        loss = FocalUncertaintyLoss()

        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0.1, 0.9, 0.2, 0.8])

        hess = loss.hessian(y_true, y_pred)

        assert hess.shape == y_pred.shape
        assert (hess > 0).all()  # Hessian should be positive


class TestFeatureSelectionAgent:
    """Tests for FeatureSelectionAgent."""

    def test_init(self, random_state: int) -> None:
        """Test initialization."""
        agent = FeatureSelectionAgent(
            n_features=50,
            feature_budget=0.6,
            random_state=random_state
        )
        assert agent.n_features == 50
        assert agent.feature_budget == 0.6

    def test_select_features(self, random_state: int) -> None:
        """Test feature selection."""
        agent = FeatureSelectionAgent(
            n_features=50,
            feature_budget=0.6,
            random_state=random_state
        )

        mask = agent.select_features(deterministic=True)

        assert mask.shape == (50,)
        assert mask.dtype == bool
        # Should select approximately 60% of features
        assert 20 < mask.sum() < 40

    def test_update_policy(self, random_state: int) -> None:
        """Test policy update."""
        agent = FeatureSelectionAgent(
            n_features=20,
            random_state=random_state
        )

        # Select features
        mask = agent.select_features()

        # Update with positive reward
        agent.update_policy(reward=1.0)

        assert len(agent.reward_history) == 1
        assert agent.reward_history[0] == 1.0

    def test_get_feature_importance(self, random_state: int) -> None:
        """Test getting feature importance."""
        agent = FeatureSelectionAgent(
            n_features=20,
            random_state=random_state
        )

        importance = agent.get_feature_importance()

        assert importance.shape == (20,)
        assert (importance >= 0).all()
        assert (importance <= 1).all()


class TestTemperatureScaler:
    """Tests for TemperatureScaler."""

    def test_init(self) -> None:
        """Test initialization."""
        scaler = TemperatureScaler()
        assert scaler.temperature == 1.0

    def test_fit(self, random_state: int) -> None:
        """Test fitting temperature."""
        np.random.seed(random_state)

        logits = np.random.randn(100)
        labels = (logits > 0).astype(int)

        scaler = TemperatureScaler()
        scaler.fit(logits, labels)

        assert scaler.temperature > 0

    def test_transform(self, random_state: int) -> None:
        """Test transforming logits."""
        np.random.seed(random_state)

        logits = np.random.randn(50)

        scaler = TemperatureScaler()
        scaler.temperature = 1.5

        probs = scaler.transform(logits)

        assert probs.shape == logits.shape
        assert (probs >= 0).all()
        assert (probs <= 1).all()


class TestAdaptiveUncertaintyFraudDetector:
    """Tests for AdaptiveUncertaintyFraudDetector."""

    def test_init(self, random_state: int) -> None:
        """Test initialization."""
        model = AdaptiveUncertaintyFraudDetector(
            feature_budget=0.6,
            n_estimators=50,
            random_state=random_state
        )

        assert model.feature_budget == 0.6
        assert model.n_estimators == 50

    def test_fit(self, small_fraud_data: tuple, random_state: int) -> None:
        """Test model fitting."""
        X, y = small_fraud_data

        # Preprocess
        preprocessor = FraudPreprocessor()
        X_processed = preprocessor.fit_transform(X)

        # Split
        split_idx = int(0.8 * len(X_processed))
        X_train = X_processed[:split_idx]
        y_train = y.values[:split_idx]
        X_val = X_processed[split_idx:]
        y_val = y.values[split_idx:]

        # Train
        model = AdaptiveUncertaintyFraudDetector(
            n_estimators=50,
            random_state=random_state
        )

        model.fit(X_train, y_train, X_val, y_val, verbose=False)

        assert model.ensemble is not None
        assert len(model.base_models) > 0

    def test_predict_proba(self, small_fraud_data: tuple, random_state: int) -> None:
        """Test probability prediction."""
        X, y = small_fraud_data

        preprocessor = FraudPreprocessor()
        X_processed = preprocessor.fit_transform(X)

        split_idx = int(0.8 * len(X_processed))
        X_train = X_processed[:split_idx]
        y_train = y.values[:split_idx]

        model = AdaptiveUncertaintyFraudDetector(
            n_estimators=50,
            random_state=random_state
        )
        model.fit(X_train, y_train, verbose=False)

        # Predict
        probs = model.predict_proba(X_train)

        assert probs.shape == (len(X_train), 2)
        assert (probs >= 0).all()
        assert (probs <= 1).all()
        # Probabilities should sum to 1
        np.testing.assert_array_almost_equal(probs.sum(axis=1), 1.0)

    def test_predict(self, small_fraud_data: tuple, random_state: int) -> None:
        """Test binary prediction."""
        X, y = small_fraud_data

        preprocessor = FraudPreprocessor()
        X_processed = preprocessor.fit_transform(X)

        split_idx = int(0.8 * len(X_processed))
        X_train = X_processed[:split_idx]
        y_train = y.values[:split_idx]

        model = AdaptiveUncertaintyFraudDetector(
            n_estimators=50,
            random_state=random_state
        )
        model.fit(X_train, y_train, verbose=False)

        # Predict
        preds = model.predict(X_train)

        assert preds.shape == (len(X_train),)
        assert set(preds).issubset({0, 1})

    def test_feature_selection(self, small_fraud_data: tuple, random_state: int) -> None:
        """Test that feature selection is applied."""
        X, y = small_fraud_data

        preprocessor = FraudPreprocessor()
        X_processed = preprocessor.fit_transform(X)

        model = AdaptiveUncertaintyFraudDetector(
            feature_budget=0.5,
            use_feature_selection=True,
            n_estimators=50,
            random_state=random_state
        )

        model.fit(X_processed, y.values, verbose=False)

        # Check that feature mask exists and selects subset
        assert model.feature_mask_ is not None
        assert model.feature_mask_.sum() < len(model.feature_mask_)

    def test_get_feature_importance(self, small_fraud_data: tuple, random_state: int) -> None:
        """Test getting feature importance."""
        X, y = small_fraud_data

        preprocessor = FraudPreprocessor()
        X_processed = preprocessor.fit_transform(X)

        model = AdaptiveUncertaintyFraudDetector(
            use_feature_selection=True,
            n_estimators=50,
            random_state=random_state
        )
        model.fit(X_processed, y.values, verbose=False)

        importance = model.get_feature_importance()

        assert importance is not None
        assert len(importance) == X_processed.shape[1]
