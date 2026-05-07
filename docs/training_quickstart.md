# Training Quick Start Guide

## Prerequisites

1. Phase 2 dataset splits must exist:
```bash
ls -lh artifacts/dataset/splits/
# Should show: train.csv, val.csv, internal_test.csv, all_splits.csv
```

2. Verify environment:
```bash
uv run python -c "from autolens_ai.training import print_device_info; print_device_info()"
# Should detect MPS on Apple Silicon
```

## Training Commands

### Option 1: Using Makefile (Recommended)

```bash
# Train all Baseline 0 models (ResNet18, MobileNetV4, EfficientNet-B2)
make train-baseline-0

# Train Baseline 1 (augmentation comparison)
make train-baseline-1

# Train Baseline 2 (weighted sampler comparison)
make train-baseline-2

# Run all baselines sequentially
make train-all-baselines

# With W&B logging
make train-baseline-0 WANDB=--wandb
```

### Option 2: Direct Script Execution

```bash
# Single model training
uv run python scripts/train.py --config configs/experiments/baseline_0_resnet18.yaml

# With W&B logging
uv run python scripts/train.py --config configs/experiments/baseline_0_resnet18.yaml --wandb

# Custom W&B project
uv run python scripts/train.py \
  --config configs/experiments/baseline_0_resnet18.yaml \
  --wandb \
  --wandb-project my-project-name
```

## What Gets Saved

### Checkpoints

Each experiment saves to its own directory:
- `checkpoints/baseline_0_resnet18/`
- `checkpoints/baseline_0_mobilenetv4/`
- `checkpoints/baseline_0_efficientnet_b2/`
- `checkpoints/baseline_1_resnet18_augmented/`
- `checkpoints/baseline_2_resnet18_weighted_sampler/`

Files saved:
- `best-*.ckpt` - Top 3 checkpoints by val/f1_macro
- `best_loss-*.ckpt` - Best checkpoint by val/loss
- `last.ckpt` - Last epoch checkpoint

### Metrics Logged

**Training:**
- train/loss
- train/acc

**Validation:**
- val/loss
- val/acc
- val/f1_macro (primary ranking metric)
- val/f1_weighted
- val/precision
- val/recall

**Test:**
- test/loss
- test/acc
- test/f1_macro
- test/f1_weighted
- test/precision
- test/recall
- test/confusion_matrix

### Logs

**Without W&B:**
- CSV logs in `lightning_logs/version_X/`

**With W&B:**
- Online dashboard at wandb.ai
- Local cache in `wandb/`

## Monitoring Training

### Progress Bar

Rich progress bar shows:
- Current epoch
- Training/validation loss
- Training/validation accuracy
- Estimated time remaining

### Early Stopping

Training stops automatically if val/loss doesn't improve for 10 epochs.

### Learning Rate Schedule

ReduceLROnPlateau reduces LR by 0.5 when val/loss plateaus (patience=3).

## Expected Training Time (Estimates)

On Apple Silicon M2 Pro with MPS:
- ResNet18: ~2-3 hours (50 epochs)
- MobileNetV4: ~3-4 hours
- EfficientNet-B2: ~4-5 hours

Early stopping may reduce actual time.

## Troubleshooting

### MPS Not Detected

```bash
# Check PyTorch MPS availability
uv run python -c "import torch; print(torch.backends.mps.is_available())"
```

### Out of Memory

Reduce batch size in config:
```yaml
batch_size: 16  # or 8
```

### W&B Login

```bash
uv run wandb login
# Or set WANDB_API_KEY environment variable
```

### Dataset Not Found

Verify paths in config match your dataset location:
```yaml
data_root: artifacts/dataset/raw
train_csv: artifacts/dataset/splits/train.csv
```

## Next Steps After Training

1. Compare results across models:
   - Check val/f1_macro scores
   - Review per-class F1 from test confusion matrix
   - Compare artifact sizes

2. Select best model for Phase 4 comparison with DINOv3

3. Document findings in Phase 3 evidence for final report
