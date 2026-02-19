# Adaptive Feature Selection with Uncertainty-Aware Ensemble for Fraud Detection

A production-grade fraud detection system combining reinforcement learning-based adaptive feature selection with uncertainty-calibrated ensemble methods. Achieves strong performance on highly imbalanced transaction data through three novel contributions: (1) per-transaction feature selection via policy gradient RL, (2) epistemic uncertainty-weighted ensemble voting with temperature scaling, and (3) focal-uncertainty loss jointly optimizing for class imbalance and prediction confidence.

---

## Key Features

- **Adaptive Feature Selection** -- RL agent dynamically selects the most informative features per transaction, reducing input dimensionality by 40%
- **Uncertainty-Aware Ensemble** -- Calibrated voting weighted by model confidence estimates using temperature-scaled probabilities
- **Custom Focal-Uncertainty Loss** -- Handles extreme class imbalance with joint uncertainty regularization
- **Gradient Boosting Ensemble** -- LightGBM, XGBoost, and CatBoost as diverse base learners
- **Comprehensive Evaluation** -- ROC-AUC, PR-AUC, F1, precision-recall trade-off, and feature efficiency metrics

## Results

Performance evaluated on a synthetic fraud dataset: **50,000 transactions**, **105 engineered features**, **4.4% fraud rate**.  
Data split: 35,000 train / 5,000 validation / 10,000 test. Total training time: ~12 minutes.

### Detection Performance

| Metric | Value |
|---|---|
| **ROC-AUC** | 0.8186 |
| **PR-AUC** | 0.3635 |
| **F1 Score** | 0.3089 |
| **Recall (Sensitivity)** | 0.5397 |
| **Precision** | 0.2164 |
| **Specificity** | 0.9098 |
| **Accuracy** | 0.8935 |

### Feature Selection & Ensemble

| Component | Detail |
|---|---|
| Features Selected | 63 / 105 (60% budget) |
| Feature Efficiency | 0.60 |
| Ensemble | LightGBM + XGBoost + CatBoost |
| Calibration | Temperature scaling (T = 0.33 -- 10.0 per model) |
| Validation Focal Loss | 0.0533 |

### Confusion Matrix (Test Set, n = 10,000)

|  | Predicted Legitimate | Predicted Fraud |
|---|---|---|
| **Actual Legitimate** | 8,697 (TN) | 862 (FP) |
| **Actual Fraud** | 203 (FN) | 238 (TP) |

> The model identifies **54% of fraudulent transactions** while maintaining a **91% acceptance rate** on legitimate transactions, using only **60% of available features** through adaptive selection.

### Evaluation Plots

The following visualizations are generated during evaluation and saved to `results/`:

| Plot | File |
|---|---|
| ROC Curve | `results/roc_curve.png` |
| Precision-Recall Curve | `results/pr_curve.png` |
| Confusion Matrix | `results/confusion_matrix.png` |
| Feature Importance | `results/feature_importance.png` |
| Threshold Analysis | `results/threshold_analysis.png` |

## Architecture

The system consists of three integrated components:

### 1. FeatureSelectionAgent

Policy gradient RL agent that learns to select feature subsets based on predictive uncertainty, targeting a 60% feature budget while maximizing fraud detection accuracy.

- **Policy**: Maintains logits for each feature, producing selection probabilities via sigmoid
- **Reward**: Balances detection accuracy against feature acquisition cost
- **Training**: REINFORCE algorithm with variance-reduction baseline
- **Adaptation**: Adjusts selection based on per-feature uncertainty scores from the ensemble

### 2. UncertaintyEnsemble

Combines LightGBM, XGBoost, and CatBoost predictions using temperature-scaled calibration. Model votes are weighted by inverse epistemic uncertainty (entropy-based).

- **Calibration**: Temperature scaling fitted on the validation set ensures probabilities reflect true confidence
- **Weighting**: Models with lower epistemic uncertainty receive higher voting weight
- **Base Learners**: LightGBM (speed), XGBoost (accuracy), CatBoost (categorical handling)

### 3. FocalUncertaintyLoss

Custom loss function combining focal loss (`gamma=2.0`, `alpha=0.25`) with uncertainty regularization to jointly handle class imbalance and prediction confidence.

- **Focal Term**: `(1-p)^gamma` down-weights easy examples, focusing on hard fraud cases
- **Uncertainty Term**: Penalizes high prediction entropy to encourage confident decisions
- **Gradient**: Analytical first/second derivatives enable efficient gradient boosting integration

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

Train the full model with all novel components:

```bash
python scripts/train.py --config configs/default.yaml
```

Run ablation study (baseline without adaptive features):

```bash
python scripts/train.py --config configs/ablation.yaml --output-dir models/ablation
```

Evaluate on test set:

```bash
python scripts/evaluate.py --model-path models/checkpoints/best_model.pkl
```

Make predictions on new data:

```bash
python scripts/predict.py --input data.csv --output predictions.csv
```

## Project Structure

```
adaptive-feature-selection-with-uncertainty-aware-ensemble-for-fraud-detection/
├── src/adaptive_feature_selection_with_uncertainty_aware_ensemble_for_fraud_detection/
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # Model components and architecture
│   ├── training/          # Training loop and utilities
│   ├── evaluation/        # Metrics and analysis
│   └── utils/             # Configuration and helpers
├── scripts/
│   ├── train.py           # Training pipeline
│   ├── evaluate.py        # Evaluation with visualizations
│   └── predict.py         # Inference on new data
├── configs/
│   ├── default.yaml       # Full model configuration
│   └── ablation.yaml      # Baseline without novel components
├── results/               # Evaluation outputs and plots
├── models/                # Saved model checkpoints
└── tests/                 # Comprehensive test suite
```

## Configuration

Edit `configs/default.yaml` to customize hyperparameters:

```yaml
model:
  feature_budget: 0.6              # Target proportion of features to select
  focal_alpha: 0.25                # Focal loss class weight
  focal_gamma: 2.0                 # Focal loss focusing parameter
  uncertainty_weight: 0.1          # Uncertainty regularization weight
  use_feature_selection: true      # Enable adaptive feature selection
  use_uncertainty_weighting: true  # Enable uncertainty-weighted ensemble
  n_estimators: 500                # Boosting rounds per base learner
  learning_rate: 0.05              # Gradient boosting learning rate
  max_depth: 6                     # Maximum tree depth
```

## Ablation Study

Compare the full model against a baseline without the novel components:

```bash
# Train full model
python scripts/train.py --config configs/default.yaml --output-dir models/full

# Train baseline (no adaptive features, no uncertainty weighting)
python scripts/train.py --config configs/ablation.yaml --output-dir models/baseline

# Evaluate both
python scripts/evaluate.py --model-path models/full/checkpoints/best_model.pkl --output-dir results/full
python scripts/evaluate.py --model-path models/baseline/checkpoints/best_model.pkl --output-dir results/baseline
```

## Testing

Run the test suite:

```bash
pytest tests/ -v --cov=src
```

## Methodology

### Adaptive Feature Selection via Reinforcement Learning

Traditional fraud detection uses fixed feature sets for all transactions. This approach employs a policy gradient agent that learns to select features dynamically per transaction. The agent receives a reward signal that balances detection accuracy against feature acquisition cost, enabling cost-effective fraud detection by using only the most informative features for each input.

### Uncertainty-Weighted Ensemble with Temperature Scaling

Standard ensembles use uniform or performance-based weights. This system weights predictions by calibrated uncertainty: temperature scaling on the validation set ensures probabilities reflect true confidence, and models with lower epistemic uncertainty (measured via prediction entropy) receive higher voting weight. This improves robustness by down-weighting uncertain predictions while maintaining learner diversity.

### Focal-Uncertainty Loss Function

Existing focal loss handles class imbalance but ignores prediction confidence. This custom objective jointly optimizes a focal term that down-weights easy examples to focus on hard fraud cases, and an uncertainty term that penalizes high prediction entropy to encourage confident decisions. The combined loss addresses both the 4.4% fraud imbalance and the need for reliable uncertainty estimates.

## License

MIT License -- Copyright (c) 2026 Alireza Shojaei. See [LICENSE](LICENSE) for details.
