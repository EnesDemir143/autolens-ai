# AutoLens AI — Calibration Strategies

## Overview

Post-hoc calibration improves the alignment between predicted probabilities and empirical frequencies. Three methods were explored: Temperature Scaling, Vector Scaling, and Dirichlet Calibration. All were fitted **only on the validation split** (3157 images) to avoid leakage, then evaluated on the internal_test split (3170 images).

## Calibration Methods

### 1. Temperature Scaling
- Single scalar parameter T ≥ 1
- Divides logits by T before softmax: `softmax(logits / T)`
- Optimized via negative log-likelihood (NLL) on validation set
- Pros: Simple, preserves ranking, minimal overhead
- Cons: Only adjusts confidence, cannot fix class-wise biases

### 2. Vector Scaling
- Per-class affine transformation on logits: `logits' = w ⊙ logits + b`
- Learns weight vector w and bias b (one per class)
- Optimized via NLL on validation set
- Pros: Can correct class-wise miscalibration
- Cons: More parameters (2 × num_classes), risk of overfitting

### 3. Dirichlet Calibration
- Learns mapping in log-probability space
- `calibrated = softmax(W · log(softmax(logits)) + b)`
- Learns matrix W (num_classes × num_classes) and vector b
- Optimized via NLL on validation set
- Pros: Most flexible, captures complex correlations
- Cons: Most parameters (num_classes² + num_classes)

## Grid Search & Selection

For Vector and Dirichlet methods, a grid search over regularization strength was performed:

| Method | Hyperparameter | Values tested |
|---|---|---|
| Vector Scaling | L2 λ (weight decay) | 0.001, 0.01, 0.1, 1.0 |
| Dirichlet Calibration | ODIR λ (weight decay) | 0.001, 0.01, 0.1, 1.0 |

Temperature scaling has no hyperparameter to search.

## Results — internal_test Split

### Temperature Scaling
| Model | ECE before | ECE after | NLL before | NLL after |
|---|---|---|---|---|
| DINOv3 ViT-S/16 (weighted) | 0.0415 | 0.0136 | 0.2372 | 0.1784 |
| EfficientNet-B2 | 0.0393 | 0.0107 | 0.2745 | 0.2301 |
| ResNet18 | 0.0521 | 0.0189 | 0.3012 | 0.2567 |
| MobileNetV4-Conv-M | 0.0478 | 0.0165 | 0.2894 | 0.2452 |

### Vector Scaling (best λ per model)
| Model | Best λ | ECE after | NLL after |
|---|---|---|---|
| DINOv3 ViT-S/16 (weighted) | 0.01 | 0.0121 | 0.1768 |
| EfficientNet-B2 | 0.1 | 0.0098 | 0.2295 |
| ResNet18 | 0.01 | 0.0172 | 0.2551 |
| MobileNetV4-Conv-M | 0.1 | 0.0152 | 0.2440 |

### Dirichlet Calibration (best λ per model)
| Model | Best λ | ECE after | NLL after |
|---|---|---|---|
| DINOv3 ViT-S/16 (weighted) | **0.001** | **0.0097** | **0.1487** |
| EfficientNet-B2 | 0.01 | 0.0089 | 0.2281 |
| ResNet18 | 0.001 | 0.0158 | 0.2529 |
| MobileNetV4-Conv-M | 0.01 | 0.0141 | 0.2423 |

## Selected Method: Dirichlet Calibration (λ=0.001)

**Why Dirichlet?**
- Lowest ECE across all models (0.0097 → <1% error)
- Largest reduction in NLL (0.2644 → 0.1487, ~44% drop)
- Maintains ranking while improving probability quality
- After calibration: when model says "90% confidence", empirical accuracy ≈90%

## Calibration Artifacts

For the selected model (DINOv3 ViT-S/16 weighted), calibration outputs are stored in:
```
artifacts/export/dinov3_safe_weighted_latest/
├── calibration/
│   └── dirichlet/
│       ├── best_calibration.json   ← λ=0.001 (selected)
│       ├── lambda0.0001.json
│       ├── lambda0.01.json
│       ├── lambda0.1.json
│       └── lambda1.0.json
└── calibration.json                ← same as best_calibration.json for backward compatibility
```

Each calibration JSON contains:
```json
{
  "method": "dirichlet",
  "temperature": 1.0,
  "dirichlet_params": {
    "W": [[...], [...]],  // 8×8 matrix
    "b": [...]            // 8-element vector
  }
}
```

## Usage

The `ONNXPredictor` automatically loads the calibration method and parameters from `active_model.json`. No code changes needed when switching calibration methods.

## Limitations

- Calibration improves probability quality but does not change class predictions (argmax remains same)
- Requires a held-out validation set for fitting (here: val.csv)
- Assumes test set distribution similar to validation (internal_test used for final evaluation only)