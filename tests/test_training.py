"""Tests for training utilities."""

import tempfile
from pathlib import Path

import pytest

from adaptive_feature_selection_with_uncertainty_aware_ensemble_for_fraud_detection.training import (
    FraudDetectionTrainer
)
from adaptive_feature_selection_with_uncertainty_aware_ensemble_for_fraud_detection.evaluation import (
    FraudMetrics
)
import numpy as np


class TestFraudDetectionTrainer:
    """Tests for FraudDetectionTrainer."""

    def test_init(self, sample_config: dict) -> None:
        """Test initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = FraudDetectionTrainer(
                config=sample_config,
                output_dir=Path(tmpdir),
                random_state=42
            )

            assert trainer.config == sample_config
            assert trainer.random_state == 42
            assert trainer.checkpoint_dir.exists()
            assert trainer.results_dir.exists()

    def test_train(self, sample_config: dict) -> None:
        """Test full training pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = FraudDetectionTrainer(
                config=sample_config,
                output_dir=Path(tmpdir),
                random_state=42
            )

            results = trainer.train()

            # Check results
            assert "model_path" in results
            assert "preprocessor_path" in results
            assert "data_stats" in results

            # Check files were created
            assert Path(results["model_path"]).exists()
            assert Path(results["preprocessor_path"]).exists()

    def test_create_model(self, sample_config: dict) -> None:
        """Test model creation from config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = FraudDetectionTrainer(
                config=sample_config,
                output_dir=Path(tmpdir),
                random_state=42
            )

            model = trainer._create_model()

            assert model is not None
            assert model.feature_budget == sample_config["model"]["feature_budget"]
            assert model.n_estimators == sample_config["model"]["n_estimators"]


class TestFraudMetrics:
    """Tests for FraudMetrics."""

    def test_compute_all_metrics(self, random_state: int) -> None:
        """Test computing all metrics."""
        np.random.seed(random_state)

        # Generate fake predictions
        n_samples = 100
        y_true = np.random.randint(0, 2, n_samples)
        y_pred = np.random.randint(0, 2, n_samples)
        y_proba = np.random.random(n_samples)

        metrics = FraudMetrics.compute_all_metrics(
            y_true, y_pred, y_proba
        )

        # Check required metrics exist
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "roc_auc" in metrics
        assert "pr_auc" in metrics

        # Check metric values are valid
        for key, value in metrics.items():
            if isinstance(value, float):
                assert 0 <= value <= 1 or key in ["true_positives", "false_positives",
                                                    "true_negatives", "false_negatives"]

    def test_compute_all_metrics_with_feature_mask(self, random_state: int) -> None:
        """Test metrics with feature mask."""
        np.random.seed(random_state)

        y_true = np.random.randint(0, 2, 100)
        y_pred = np.random.randint(0, 2, 100)
        y_proba = np.random.random(100)
        feature_mask = np.random.random(50) > 0.5

        metrics = FraudMetrics.compute_all_metrics(
            y_true, y_pred, y_proba, feature_mask
        )

        assert "feature_efficiency" in metrics
        assert "features_selected" in metrics
        assert metrics["features_selected"] == feature_mask.sum()

    def test_compute_threshold_metrics(self, random_state: int) -> None:
        """Test threshold-based metrics."""
        np.random.seed(random_state)

        y_true = np.random.randint(0, 2, 100)
        y_proba = np.random.random(100)

        threshold_metrics = FraudMetrics.compute_threshold_metrics(
            y_true, y_proba, thresholds=np.array([0.3, 0.5, 0.7])
        )

        assert "roc" in threshold_metrics
        assert "pr" in threshold_metrics
        assert "custom_thresholds" in threshold_metrics
        assert len(threshold_metrics["custom_thresholds"]) == 3

    def test_compute_per_class_metrics(self, random_state: int) -> None:
        """Test per-class metrics."""
        np.random.seed(random_state)

        y_true = np.random.randint(0, 2, 100)
        y_pred = np.random.randint(0, 2, 100)

        per_class = FraudMetrics.compute_per_class_metrics(y_true, y_pred)

        assert "legitimate" in per_class
        assert "fraud" in per_class

        for class_name in ["legitimate", "fraud"]:
            assert "support" in per_class[class_name]
            assert "precision" in per_class[class_name]
            assert "recall" in per_class[class_name]
            assert "f1_score" in per_class[class_name]

    def test_print_metrics_summary(self, random_state: int, capsys) -> None:
        """Test printing metrics summary."""
        np.random.seed(random_state)

        y_true = np.random.randint(0, 2, 100)
        y_pred = np.random.randint(0, 2, 100)
        y_proba = np.random.random(100)

        metrics = FraudMetrics.compute_all_metrics(y_true, y_pred, y_proba)

        FraudMetrics.print_metrics_summary(metrics)

        captured = capsys.readouterr()
        assert "FRAUD DETECTION METRICS SUMMARY" in captured.out
        assert "ROC-AUC" in captured.out
        assert "F1 Score" in captured.out
