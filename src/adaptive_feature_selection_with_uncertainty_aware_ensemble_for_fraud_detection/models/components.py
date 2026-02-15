"""Custom components for uncertainty-aware fraud detection.

This module contains the novel contributions:
1. FeatureSelectionAgent: RL agent for adaptive feature selection
2. UncertaintyEnsemble: Calibrated ensemble with uncertainty-based voting
3. FocalUncertaintyLoss: Custom loss combining focal loss with uncertainty
4. TemperatureScaling: Calibration for uncertainty estimation
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.special import softmax
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


class FocalUncertaintyLoss:
    """Custom focal loss combined with uncertainty regularization.

    This loss jointly optimizes for:
    1. Class imbalance (via focal loss)
    2. Prediction confidence (via uncertainty penalty)

    The focal loss down-weights easy examples and focuses on hard ones,
    while the uncertainty term encourages confident predictions.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        uncertainty_weight: float = 0.1
    ) -> None:
        """Initialize the focal uncertainty loss.

        Args:
            alpha: Weighting factor for class imbalance (higher = more weight on minority class).
            gamma: Focusing parameter (higher = more focus on hard examples).
            uncertainty_weight: Weight for uncertainty regularization term.
        """
        self.alpha = alpha
        self.gamma = gamma
        self.uncertainty_weight = uncertainty_weight

    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        uncertainty: Optional[np.ndarray] = None
    ) -> float:
        """Compute the focal uncertainty loss.

        Args:
            y_true: True labels (0 or 1).
            y_pred: Predicted probabilities.
            uncertainty: Optional uncertainty estimates (higher = more uncertain).

        Returns:
            Scalar loss value.
        """
        # Clip predictions to prevent log(0)
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Compute focal loss
        # For positive class (fraud)
        focal_positive = -self.alpha * np.power(1 - y_pred, self.gamma) * np.log(y_pred)
        # For negative class (legitimate)
        focal_negative = -(1 - self.alpha) * np.power(y_pred, self.gamma) * np.log(1 - y_pred)

        # Combine based on true labels
        focal_loss = np.where(y_true == 1, focal_positive, focal_negative)

        # Add uncertainty regularization if provided
        if uncertainty is not None:
            # Penalize high uncertainty (encourages confident predictions)
            uncertainty_term = self.uncertainty_weight * np.mean(uncertainty)
            total_loss = np.mean(focal_loss) + uncertainty_term
        else:
            total_loss = np.mean(focal_loss)

        return total_loss

    def gradient(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """Compute gradient for boosting algorithms.

        Args:
            y_true: True labels.
            y_pred: Predicted probabilities.

        Returns:
            Gradient array.
        """
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Gradient of focal loss
        pt = np.where(y_true == 1, y_pred, 1 - y_pred)
        alpha_t = np.where(y_true == 1, self.alpha, 1 - self.alpha)

        grad = alpha_t * (
            self.gamma * np.power(1 - pt, self.gamma - 1) * np.log(pt) +
            np.power(1 - pt, self.gamma) / pt
        )

        # Adjust sign based on label
        grad = np.where(y_true == 1, -grad, grad)

        return grad

    def hessian(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """Compute hessian for boosting algorithms.

        Args:
            y_true: True labels.
            y_pred: Predicted probabilities.

        Returns:
            Hessian array.
        """
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Approximate hessian (using Fisher information)
        hess = y_pred * (1 - y_pred)

        return np.maximum(hess, 1e-6)


class FeatureSelectionAgent:
    """Reinforcement learning agent for adaptive feature selection.

    Uses a simple policy gradient approach where:
    - State: current transaction features + model uncertainties
    - Action: binary mask selecting subset of features
    - Reward: prediction accuracy - feature cost

    The agent learns to select informative features while minimizing cost.
    """

    def __init__(
        self,
        n_features: int,
        feature_budget: float = 0.6,
        exploration_rate: float = 0.2,
        learning_rate: float = 0.01,
        random_state: int = 42
    ) -> None:
        """Initialize the feature selection agent.

        Args:
            n_features: Total number of features available.
            feature_budget: Target proportion of features to select.
            exploration_rate: Probability of random exploration.
            learning_rate: Learning rate for policy updates.
            random_state: Random seed for reproducibility.
        """
        self.n_features = n_features
        self.feature_budget = feature_budget
        self.exploration_rate = exploration_rate
        self.learning_rate = learning_rate
        self.random_state = random_state

        # Policy parameters: logits for each feature
        np.random.seed(random_state)
        self.policy_logits = np.zeros(n_features)

        # Track feature selection history
        self.selection_history: List[np.ndarray] = []
        self.reward_history: List[float] = []

    def select_features(
        self,
        uncertainty: Optional[np.ndarray] = None,
        deterministic: bool = False
    ) -> np.ndarray:
        """Select features using current policy.

        Args:
            uncertainty: Optional per-feature uncertainty estimates.
            deterministic: If True, use greedy selection; else sample.

        Returns:
            Binary mask indicating selected features.
        """
        # Compute selection probabilities
        if uncertainty is not None:
            # Boost probability for high-uncertainty features
            adjusted_logits = self.policy_logits + 0.5 * uncertainty
        else:
            adjusted_logits = self.policy_logits

        probs = 1 / (1 + np.exp(-adjusted_logits))  # Sigmoid

        if deterministic:
            # Greedy: select top features by probability
            n_select = int(self.feature_budget * self.n_features)
            mask = np.zeros(self.n_features, dtype=bool)
            top_indices = np.argsort(probs)[-n_select:]
            mask[top_indices] = True
        else:
            # Stochastic: sample from Bernoulli
            if np.random.random() < self.exploration_rate:
                # Random exploration
                n_select = int(self.feature_budget * self.n_features)
                mask = np.zeros(self.n_features, dtype=bool)
                selected = np.random.choice(
                    self.n_features, size=n_select, replace=False
                )
                mask[selected] = True
            else:
                # Sample from policy
                mask = np.random.random(self.n_features) < probs

                # Ensure at least some features are selected
                if mask.sum() < 3:
                    top_indices = np.argsort(probs)[-3:]
                    mask[top_indices] = True

        self.selection_history.append(mask)
        return mask

    def update_policy(
        self,
        reward: float
    ) -> None:
        """Update policy based on received reward.

        Args:
            reward: Reward signal (higher is better).
        """
        if not self.selection_history:
            return

        self.reward_history.append(reward)

        # Simple policy gradient update
        last_mask = self.selection_history[-1]

        # Baseline: moving average of rewards
        baseline = np.mean(self.reward_history[-10:]) if len(self.reward_history) > 1 else 0

        # Advantage
        advantage = reward - baseline

        # Update logits: increase for selected features if reward is good
        update = self.learning_rate * advantage * (last_mask.astype(float) - 0.5)
        self.policy_logits += update

    def get_feature_importance(self) -> np.ndarray:
        """Get learned feature importance scores.

        Returns:
            Importance scores for each feature.
        """
        return 1 / (1 + np.exp(-self.policy_logits))


class TemperatureScaler:
    """Temperature scaling for probability calibration.

    Learns a temperature parameter that scales logits to produce
    better-calibrated probability estimates.
    """

    def __init__(self) -> None:
        """Initialize temperature scaler."""
        self.temperature: float = 1.0

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        max_iter: int = 100
    ) -> "TemperatureScaler":
        """Fit temperature parameter on validation data.

        Args:
            logits: Raw model outputs (before sigmoid).
            labels: True binary labels.
            max_iter: Maximum optimization iterations.

        Returns:
            Self for method chaining.
        """
        from scipy.optimize import minimize_scalar

        def neg_log_likelihood(temp: float) -> float:
            """Negative log-likelihood for optimization."""
            scaled_probs = 1 / (1 + np.exp(-logits / temp))
            scaled_probs = np.clip(scaled_probs, 1e-7, 1 - 1e-7)

            nll = -np.mean(
                labels * np.log(scaled_probs) +
                (1 - labels) * np.log(1 - scaled_probs)
            )
            return nll

        # Optimize temperature
        result = minimize_scalar(
            neg_log_likelihood,
            bounds=(0.1, 10.0),
            method="bounded"
        )

        self.temperature = result.x
        logger.info(f"Fitted temperature: {self.temperature:.3f}")

        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to logits.

        Args:
            logits: Raw model outputs.

        Returns:
            Calibrated probabilities.
        """
        scaled_logits = logits / self.temperature
        return 1 / (1 + np.exp(-scaled_logits))


class UncertaintyEnsemble:
    """Ensemble that weights votes by epistemic uncertainty estimates.

    Each base model's prediction is weighted by its confidence (inverse uncertainty).
    This gives more weight to models that are certain about their predictions.
    """

    def __init__(
        self,
        models: List[BaseEstimator],
        use_temperature_scaling: bool = True
    ) -> None:
        """Initialize the uncertainty ensemble.

        Args:
            models: List of fitted base models.
            use_temperature_scaling: Whether to apply temperature scaling.
        """
        self.models = models
        self.use_temperature_scaling = use_temperature_scaling
        self.temperature_scalers: List[TemperatureScaler] = [
            TemperatureScaler() for _ in models
        ]

    def fit_calibration(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> "UncertaintyEnsemble":
        """Fit temperature scaling on validation data.

        Args:
            X_val: Validation features.
            y_val: Validation labels.

        Returns:
            Self for method chaining.
        """
        if not self.use_temperature_scaling:
            return self

        logger.info("Fitting temperature scaling for ensemble calibration...")

        for i, (model, scaler) in enumerate(zip(self.models, self.temperature_scalers)):
            # Get raw predictions (logits)
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_val)[:, 1]
                # Convert probabilities back to logits
                probs = np.clip(probs, 1e-7, 1 - 1e-7)
                logits = np.log(probs / (1 - probs))
            else:
                # Handle XGBoost Booster objects that need DMatrix
                try:
                    import xgboost as xgb
                    if isinstance(model, xgb.Booster):
                        dmatrix = xgb.DMatrix(X_val)
                        probs = model.predict(dmatrix)
                        probs = np.clip(probs, 1e-7, 1 - 1e-7)
                        logits = np.log(probs / (1 - probs))
                    else:
                        logits = model.predict(X_val)
                except ImportError:
                    logits = model.predict(X_val)

            scaler.fit(logits, y_val)

        return self

    def predict_proba(
        self,
        X: np.ndarray,
        return_uncertainty: bool = False
    ) -> np.ndarray:
        """Predict probabilities with uncertainty-weighted voting.

        Args:
            X: Features to predict.
            return_uncertainty: If True, also return uncertainty estimates.

        Returns:
            Predicted probabilities (and optionally uncertainties).
        """
        all_probs = []
        all_uncertainties = []

        for i, model in enumerate(self.models):
            # Get predictions
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)[:, 1]
            else:
                # Handle XGBoost Booster objects that need DMatrix
                try:
                    import xgboost as xgb
                    if isinstance(model, xgb.Booster):
                        dmatrix = xgb.DMatrix(X)
                        probs = model.predict(dmatrix)
                    else:
                        probs = 1 / (1 + np.exp(-model.predict(X)))
                except ImportError:
                    probs = 1 / (1 + np.exp(-model.predict(X)))

            # Apply temperature scaling
            if self.use_temperature_scaling:
                # Convert to logits, scale, convert back
                probs_clipped = np.clip(probs, 1e-7, 1 - 1e-7)
                logits = np.log(probs_clipped / (1 - probs_clipped))
                probs = self.temperature_scalers[i].transform(logits)

            all_probs.append(probs)

            # Estimate uncertainty (entropy-based)
            uncertainty = -probs * np.log(probs + 1e-7) - (1 - probs) * np.log(1 - probs + 1e-7)
            all_uncertainties.append(uncertainty)

        all_probs = np.array(all_probs)  # Shape: (n_models, n_samples)
        all_uncertainties = np.array(all_uncertainties)

        # Weight by inverse uncertainty (more weight to confident predictions)
        weights = 1 / (all_uncertainties + 1e-6)
        weights = weights / weights.sum(axis=0, keepdims=True)

        # Weighted average
        ensemble_probs = np.sum(all_probs * weights, axis=0)

        if return_uncertainty:
            # Aggregate uncertainty
            ensemble_uncertainty = np.mean(all_uncertainties, axis=0)
            return ensemble_probs, ensemble_uncertainty

        return ensemble_probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels.

        Args:
            X: Features to predict.

        Returns:
            Binary predictions.
        """
        probs = self.predict_proba(X)
        return (probs > 0.5).astype(int)
