"""Main adaptive uncertainty-aware fraud detection model."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

from .components import (
    FeatureSelectionAgent,
    FocalUncertaintyLoss,
    UncertaintyEnsemble
)

logger = logging.getLogger(__name__)


class AdaptiveUncertaintyFraudDetector(BaseEstimator, ClassifierMixin):
    """Adaptive feature selection with uncertainty-aware ensemble for fraud detection.

    This model combines three novel components:
    1. RL-based adaptive feature selection per transaction
    2. Uncertainty-aware ensemble with temperature-scaled calibration
    3. Focal-uncertainty loss for imbalanced learning

    The model trains multiple base learners (LightGBM, XGBoost, CatBoost) and
    combines them using uncertainty-weighted voting.
    """

    def __init__(
        self,
        feature_budget: float = 0.6,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        uncertainty_weight: float = 0.1,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        use_feature_selection: bool = True,
        use_uncertainty_weighting: bool = True,
        random_state: int = 42
    ) -> None:
        """Initialize the adaptive uncertainty fraud detector.

        Args:
            feature_budget: Proportion of features to select (0-1).
            focal_alpha: Focal loss alpha parameter.
            focal_gamma: Focal loss gamma parameter.
            uncertainty_weight: Weight for uncertainty regularization.
            n_estimators: Number of boosting rounds.
            learning_rate: Learning rate for base models.
            max_depth: Maximum tree depth.
            use_feature_selection: Whether to use adaptive feature selection.
            use_uncertainty_weighting: Whether to use uncertainty-weighted ensemble.
            random_state: Random seed for reproducibility.
        """
        self.feature_budget = feature_budget
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.uncertainty_weight = uncertainty_weight
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.use_feature_selection = use_feature_selection
        self.use_uncertainty_weighting = use_uncertainty_weighting
        self.random_state = random_state

        # Components
        self.focal_loss = FocalUncertaintyLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            uncertainty_weight=uncertainty_weight
        )
        self.feature_agent: Optional[FeatureSelectionAgent] = None
        self.ensemble: Optional[UncertaintyEnsemble] = None
        self.base_models: List[BaseEstimator] = []

        # Metadata
        self.n_features_: Optional[int] = None
        self.classes_ = np.array([0, 1])
        self.feature_mask_: Optional[np.ndarray] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True
    ) -> "AdaptiveUncertaintyFraudDetector":
        """Fit the adaptive uncertainty fraud detector.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features for early stopping and calibration.
            y_val: Validation labels.
            verbose: Whether to log training progress.

        Returns:
            Self for method chaining.
        """
        if verbose:
            logger.info("Training adaptive uncertainty fraud detector...")

        self.n_features_ = X_train.shape[1]

        # Initialize feature selection agent
        if self.use_feature_selection:
            self.feature_agent = FeatureSelectionAgent(
                n_features=self.n_features_,
                feature_budget=self.feature_budget,
                random_state=self.random_state
            )
            # Select features for training
            self.feature_mask_ = self.feature_agent.select_features(deterministic=True)
            X_train_selected = X_train[:, self.feature_mask_]
            X_val_selected = X_val[:, self.feature_mask_] if X_val is not None else None
        else:
            self.feature_mask_ = np.ones(self.n_features_, dtype=bool)
            X_train_selected = X_train
            X_val_selected = X_val

        if verbose:
            logger.info(
                f"Selected {self.feature_mask_.sum()} / {self.n_features_} features "
                f"({self.feature_mask_.sum() / self.n_features_:.1%})"
            )

        # Train base models
        self.base_models = self._train_base_models(
            X_train_selected, y_train,
            X_val_selected, y_val,
            verbose
        )

        # Create uncertainty ensemble
        self.ensemble = UncertaintyEnsemble(
            models=self.base_models,
            use_temperature_scaling=self.use_uncertainty_weighting
        )

        # Calibrate ensemble on validation set
        if X_val is not None and y_val is not None:
            self.ensemble.fit_calibration(X_val_selected, y_val)

            # Update feature selection agent based on validation performance
            if self.use_feature_selection and self.feature_agent is not None:
                val_probs = self.ensemble.predict_proba(X_val_selected)
                val_loss = self.focal_loss(y_val, val_probs)
                # Reward is negative loss (higher is better)
                reward = -val_loss
                self.feature_agent.update_policy(reward)

                if verbose:
                    logger.info(f"Validation focal loss: {val_loss:.4f}")

        if verbose:
            logger.info("Training complete!")

        return self

    def _train_base_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        verbose: bool
    ) -> List[BaseEstimator]:
        """Train ensemble of base models.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features.
            y_val: Validation labels.
            verbose: Whether to log progress.

        Returns:
            List of trained base models.
        """
        models = []

        # Compute class weights for imbalance
        fraud_rate = y_train.mean()
        scale_pos_weight = (1 - fraud_rate) / fraud_rate

        # 1. LightGBM with custom focal loss
        if verbose:
            logger.info("Training LightGBM...")

        lgb_params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "num_leaves": 2 ** self.max_depth,
            "learning_rate": self.learning_rate,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "max_depth": self.max_depth,
            "min_child_samples": 20,
            "scale_pos_weight": scale_pos_weight,
            "verbose": -1,
            "random_state": self.random_state
        }

        train_data = lgb.Dataset(X_train, label=y_train)
        valid_sets = [train_data]
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)

        lgb_model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=self.n_estimators,
            valid_sets=valid_sets,
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        models.append(lgb_model)

        # 2. XGBoost
        if verbose:
            logger.info("Training XGBoost...")

        xgb_params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": scale_pos_weight,
            "tree_method": "hist",
            "random_state": self.random_state
        }

        dtrain = xgb.DMatrix(X_train, label=y_train)
        evals = [(dtrain, "train")]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, "val"))

        xgb_model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=self.n_estimators,
            evals=evals,
            early_stopping_rounds=50,
            verbose_eval=False
        )
        models.append(xgb_model)

        # 3. CatBoost
        if verbose:
            logger.info("Training CatBoost...")

        cat_model = CatBoostClassifier(
            iterations=self.n_estimators,
            learning_rate=self.learning_rate,
            depth=self.max_depth,
            loss_function="Logloss",
            eval_metric="AUC",
            auto_class_weights="Balanced",
            random_seed=self.random_state,
            verbose=False
        )

        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = (X_val, y_val)

        cat_model.fit(
            X_train, y_train,
            eval_set=eval_set,
            early_stopping_rounds=50,
            verbose=False
        )
        models.append(cat_model)

        return models

    def predict_proba(
        self,
        X: np.ndarray,
        return_uncertainty: bool = False
    ) -> np.ndarray:
        """Predict fraud probabilities.

        Args:
            X: Features to predict.
            return_uncertainty: Whether to also return uncertainty estimates.

        Returns:
            Predicted probabilities (and optionally uncertainties).
        """
        if self.ensemble is None:
            raise ValueError("Model must be fitted before prediction")

        # Apply feature selection
        if self.feature_mask_ is not None:
            X_selected = X[:, self.feature_mask_]
        else:
            X_selected = X

        # Get ensemble predictions
        if return_uncertainty:
            probs, uncertainty = self.ensemble.predict_proba(
                X_selected, return_uncertainty=True
            )
            # Return full probability matrix for sklearn compatibility
            probs_full = np.column_stack([1 - probs, probs])
            return probs_full, uncertainty
        else:
            probs = self.ensemble.predict_proba(X_selected)
            # Return full probability matrix
            return np.column_stack([1 - probs, probs])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary fraud labels.

        Args:
            X: Features to predict.

        Returns:
            Binary predictions (0 = legitimate, 1 = fraud).
        """
        probs = self.predict_proba(X)
        return (probs[:, 1] > 0.5).astype(int)

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get learned feature importance from RL agent.

        Returns:
            Feature importance scores, or None if feature selection not used.
        """
        if self.feature_agent is not None:
            return self.feature_agent.get_feature_importance()
        return None

    def save(self, path: Path) -> None:
        """Save the model to disk.

        Args:
            path: Path to save the model.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save entire model
        joblib.dump(self, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "AdaptiveUncertaintyFraudDetector":
        """Load a model from disk.

        Args:
            path: Path to load the model from.

        Returns:
            Loaded model instance.
        """
        model = joblib.load(path)
        logger.info(f"Model loaded from {path}")
        return model
