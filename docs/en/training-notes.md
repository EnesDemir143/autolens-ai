# AutoLens AI — Training Notes

## Overview

This document summarizes the training configurations, hyperparameters, and experimental details for all models trained in the AutoLens AI project. The goal is to provide full reproducibility of the training pipelines reported in the model comparison.

## Training Framework

- **PyTorch Lightning** (v2.6.1) for structured training loops
- **Hydra** (v1.3.2) for configuration management
- **Optuna** (v4.8.0) used only for hyperparameter sweeps in ablations (not final runs)
- **Weights & Biases** (v0.26.1) for experiment tracking (optional via `--wandb` flag)
- **Device**: Apple Silicon MPS with CPU fallback; all configs support `accelerator: "auto"`

## Common Training Settings

Unless overridden in a specific config, the following settings apply to all experiments:

| Setting | Value |
|---|---|
| **Optimizer** | AdamW |
| **Learning Rate** | 1e-4 |
| **Weight Decay** | 1e-4 |
| **Batch Size** | 32 (adjusted per model to fit memory) |
| **Epochs** | 100 (early stopping patience=10) |
| **Scheduler** | None (constant LR) |
| **Loss Function** | CrossEntropyLoss with label smoothing (0.05) |
| **EMA** | Exponential Moving Average (decay=0.999) used for DINOv3 runs |
| **Precision** | 32-bit (FP16 not used due to stability concerns) |
| **Gradient Clipping** | None |
| **Validation Interval** | Every epoch |
| **Log Every N Steps** | 50 |

## Image Preprocessing

All models receive the same preprocessing pipeline, adjusted for input size:

1. Resize to `(resize_size, resize_size)` where `resize_size = image_size + 32`
2. Center crop to `(image_size, image_size)`
3. Convert to RGB
4. Normalize with dataset-specific mean and std (computed from train split)
5. Convert to float32 tensor in NCHW format

The mean and std values are stored in `artifacts/dataset/stats.json` and are:
- **Mean**: [0.4429, 0.4354, 0.4370]
- **Std**: [0.2456, 0.2421, 0.2449]

These values were computed from the combined training splits of all sources and are used for all models.

## Model-Specific Configurations

### 1. DINOv3 ViT-S/16 (Backbone: vit_small_patch16_dinov3)

**Configs:**
- `baseline_0_dinov3_vits16.yaml` — no augmentation
- `baseline_0_dinov3_vits16_lora.yaml` — same as above with LoRA (rank=8, alpha=16)
- `dinov3_safe_focal_aug.yaml` — safe augmentation + focal loss
- `dinov3_safe_weighted_aug.yaml` — safe augmentation + weighted loss (selected)

**Specifics:**
- Input size: 256×256
- Augmentation (safe): horizontal flip (±0.1), rotate (±5°), brightness/contrast (±0.1), saturation (±0.1), hue (±0.05)
- Label smoothing: 0.05
- EMA: enabled (decay=0.999)
- LoRA: applied to query and value projections in attention modules

### 2. EfficientNet-B2 (Backbone: tf_efficientnet_b2_ns)

**Config:** `baseline_0_efficientnet_b2.yaml`

**Specifics:**
- Input size: 224×224
- Augmentation: none (baseline 0)
- Dropout: 0.3 (inherited from timm model)
- Label smoothing: 0.05
- EMA: not used

### 3. ResNet18 (Backbone: resnet18)

**Configs:**
- `baseline_0_resnet18.yaml` — no augmentation
- `baseline_1_resnet18_augmented.yaml` — light augmentation
- `baseline_2_resnet18_weighted_sampler.yaml` — weighted sampler (selected for Baseline 2)

**Specifics:**
- Input size: 224×224
- Augmentation (light): horizontal flip (0.5), rotate (±10°), brightness/contrast (±0.2)
- Label smoothing: 0.05
- EMA: not used
- Weighted sampler: samples inversely proportional to class frequencies (Baseline 2 only)

### 4. MobileNetV4-Conv-M (Backbone: mobilenetv4_conv_medium)

**Config:** `baseline_0_mobilenetv4.yaml`

**Specifics:**
- Input size: 224×224
- Augmentation: none (baseline 0)
- Label smoothing: 0.05
- EMA: not used

## Ablation Studies

Several ablations were conducted to inform the final design:

### Loss Function Comparison
- **Cross Entropy (CE)**: baseline
- **CE + Label Smoothing (LS)**: improves calibration, slight accuracy gain
- **Focal Loss**: focuses on hard examples, helps with class imbalance
- **Weighted CE**: inverse class frequency weighting, best for minority classes
- **LS + Weighted CE**: combines benefits of both (final choice for DINOv3)

### Augmentation Strategies
- **None**: highest train accuracy, risk of overfitting
- **Safe**: geometric + photometric transforms that preserve vehicle semantics
- **Strong**: includes cutout, solarize, etc. — led to lower validation performance
- **Choice**: safe augmentation for DINOv3, none for CNN baselines (to isolate architecture impact)

### EMA Impact
- **Without EMA**: faster convergence, slightly noisier validation curve
- **With EMA**: smoother validation, better generalization to test set
- **Decision**: enabled for DINOv3 runs due to smaller effective dataset size

## Hardware & Runtime

All training was conducted on an Apple MacBook Pro with M2 Pro chip (16GB RAM).

| Model | Params | Sec/epoch (train) | Max Mem | Epochs to convergence |
|---|---|---|---|---|
| DINOv3 ViT-S/16 | 21.6M | 18.2 s | 10.2 GB | 29 |
| EfficientNet-B2 | ~9M | 6.1 s | 4.1 GB | 16 |
| ResNet18 | ~11M | 6.8 s | 4.5 GB | 18 |
| MobileNetV4-Conv-M | ~9M | 5.9 s | 4.0 GB | 22 |

*Times measured with batch size 32, num_workers=4, on MPS.*

## Reproducibility

To reproduce any training run:

```bash
uv run python scripts/train.py --config configs/experiments/<config_name>.yaml [--wandb]
```

Example:
```bash
uv run python scripts/train.py --config configs/experiments/dinov3_safe_weighted_aug.yaml --wandb
```

The script will:
1. Load the Hydra config
2. Compute dataset statistics if not cached (`artifacts/dataset/stats.json`)
3. Initialize the PyTorch Lightning trainer
4. Train until early stopping (patience=10) on validation loss
5. Save the best checkpoint (by val/loss) to `checkpoints/<run_name>/`
6. Optionally log to Weights & Biases if `--wandb` flag is provided

## Checkpoint Naming

Checkpoints follow the pattern:
```
checkpoints/
└─ <config_name>_<timestamp>/
   ├─ last.ckpt
   ├─ best-{epoch}-{val_loss:.4f}.ckpt
   └─ ... (logs, hydra config, etc.)
```

Example: `checkpoints/dinov3_safe_weighted_aug_20260510_131319/`

## Post-Training Export

After training, the following pipeline is used to create deployment artifacts:

```bash
make checkpoint-to-safetensors \
   CHECKPOINT=checkpoints/<run_name>/best-*.ckpt \
   EXPORT_DIR=artifacts/export/<model_name>_latest

make export-model \
   SAFETENSORS=artifacts/export/<model_name>_latest/model.safetensors \
   METADATA=artifacts/export/<model_name>_latest/metadata.json \
   ONNX=artifacts/export/<model_name>_latest/model.onnx

make calibrate-temperature \
   ONNX=artifacts/export/<model_name>_latest/model.onnx \
   METADATA=artifacts/export/<model_name>_latest/metadata.json

make prepare-demo-artifact \
   METADATA=artifacts/export/<model_name>_latest/metadata.json \
   ONNX=artifacts/export/<model_name>_latest/model.onnx \
   CALIBRATION=artifacts/export/<model_name>_latest/calibration.json
```

The final `artifacts/demo/active_model.json` points to the exported artifact.

## Conclusion

The training pipeline emphasized reproducibility, fairness in comparison, and adherence to the 95 MB deployment constraint. The selected model (DINOv3 ViT-S/16 with weighted loss and safe augmentation) achieves the best balance of accuracy, generalization, and deployment readiness.