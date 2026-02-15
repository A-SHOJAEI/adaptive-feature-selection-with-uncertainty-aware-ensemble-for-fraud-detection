"""Data preprocessing for fraud detection."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, LabelEncoder

logger = logging.getLogger(__name__)


class FraudPreprocessor:
    """Preprocesses fraud detection data with robust scaling and imputation."""

    def __init__(
        self,
        numerical_strategy: str = "median",
        categorical_strategy: str = "most_frequent",
        scale: bool = True
    ) -> None:
        """Initialize the preprocessor.

        Args:
            numerical_strategy: Imputation strategy for numerical features.
            categorical_strategy: Imputation strategy for categorical features.
            scale: Whether to apply robust scaling to features.
        """
        self.numerical_strategy = numerical_strategy
        self.categorical_strategy = categorical_strategy
        self.scale = scale

        self.numerical_imputer: Optional[SimpleImputer] = None
        self.categorical_imputer: Optional[SimpleImputer] = None
        self.scaler: Optional[RobustScaler] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_names: Optional[List[str]] = None
        self.numerical_features: Optional[List[str]] = None
        self.categorical_features: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame) -> "FraudPreprocessor":
        """Fit the preprocessor on training data.

        Args:
            X: Training features.

        Returns:
            Self for method chaining.
        """
        logger.info("Fitting preprocessor on training data...")

        self.feature_names = list(X.columns)

        # Identify numerical and categorical features
        self.numerical_features = list(X.select_dtypes(include=[np.number]).columns)
        self.categorical_features = list(X.select_dtypes(include=["object", "category"]).columns)

        logger.info(
            f"Feature types: {len(self.numerical_features)} numerical, "
            f"{len(self.categorical_features)} categorical"
        )

        # Fit numerical imputer
        if self.numerical_features:
            self.numerical_imputer = SimpleImputer(strategy=self.numerical_strategy)
            self.numerical_imputer.fit(X[self.numerical_features])

        # Fit categorical imputer and label encoders
        if self.categorical_features:
            self.categorical_imputer = SimpleImputer(strategy=self.categorical_strategy)
            self.categorical_imputer.fit(X[self.categorical_features].astype(str))

            for col in self.categorical_features:
                self.label_encoders[col] = LabelEncoder()
                # Handle missing values before encoding
                values = X[col].fillna("missing").astype(str)
                self.label_encoders[col].fit(values)

        # Fit scaler on all features
        if self.scale:
            X_transformed = self._transform_features(X)
            self.scaler = RobustScaler()
            self.scaler.fit(X_transformed)

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform features using fitted preprocessor.

        Args:
            X: Features to transform.

        Returns:
            Transformed feature array.
        """
        X_transformed = self._transform_features(X)

        if self.scale and self.scaler is not None:
            X_transformed = self.scaler.transform(X_transformed)

        return X_transformed

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit the preprocessor and transform features.

        Args:
            X: Features to fit and transform.

        Returns:
            Transformed feature array.
        """
        return self.fit(X).transform(X)

    def _transform_features(self, X: pd.DataFrame) -> np.ndarray:
        """Transform features with imputation and encoding.

        Args:
            X: Features to transform.

        Returns:
            Transformed feature array.
        """
        parts = []

        # Transform numerical features
        if self.numerical_features and self.numerical_imputer is not None:
            X_num = self.numerical_imputer.transform(X[self.numerical_features])
            parts.append(X_num)

        # Transform categorical features
        if self.categorical_features and self.categorical_imputer is not None:
            X_cat_imputed = self.categorical_imputer.transform(
                X[self.categorical_features].astype(str)
            )

            # Encode categorical features
            X_cat_encoded = []
            for i, col in enumerate(self.categorical_features):
                encoder = self.label_encoders[col]
                values = X_cat_imputed[:, i]

                # Handle unseen categories
                encoded = np.zeros(len(values), dtype=int)
                for j, val in enumerate(values):
                    if val in encoder.classes_:
                        encoded[j] = encoder.transform([val])[0]
                    else:
                        # Assign to "missing" class or 0
                        encoded[j] = 0

                X_cat_encoded.append(encoded)

            if X_cat_encoded:
                parts.append(np.column_stack(X_cat_encoded))

        # Combine all features
        if parts:
            return np.hstack(parts)
        else:
            return X.values

    def get_feature_names(self) -> List[str]:
        """Get the names of all features after preprocessing.

        Returns:
            List of feature names.
        """
        if self.feature_names is None:
            return []

        names = []
        if self.numerical_features:
            names.extend(self.numerical_features)
        if self.categorical_features:
            names.extend(self.categorical_features)

        return names

    def inverse_transform_feature_mask(
        self,
        mask: np.ndarray
    ) -> Dict[str, bool]:
        """Convert a feature mask to feature name dictionary.

        Args:
            mask: Binary mask indicating selected features.

        Returns:
            Dictionary mapping feature names to selection status.
        """
        feature_names = self.get_feature_names()
        if len(feature_names) != len(mask):
            logger.warning(
                f"Feature mask length {len(mask)} does not match "
                f"feature count {len(feature_names)}"
            )
            return {}

        return {name: bool(selected) for name, selected in zip(feature_names, mask)}
