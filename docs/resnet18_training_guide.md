# ResNet18 Training: Step-by-Step Guide

## Prerequisites Check

```bash
# 1. Verify you're in project root
pwd
# Should show: /Users/enesdemir/Documents/autolens_ai

# 2. Check dataset splits exist
ls -lh artifacts/dataset/splits/
# Should show: train.csv, val.csv, internal_test.csv, all_splits.csv

# 3. Check MPS availability
uv run python -c "from autolens_ai.training import print_device_info; print_device_info()"
# Should show: MPS available: True, Selected device: mps
```

## Step 1: Merge Branch to Main

```bash
# Check current branch
git branch

# If on feat/baseline-training-pipeline, merge to main
git checkout main
git merge feat/baseline-training-pipeline

# Verify merge
git log --oneline -5
```

## Step 2: (Optional) W&B Setup

```bash
# Login to W&B (one-time)
uv run wandb login
# Enter your API key when prompted

# Or skip if you don't want W&B logging
```

## Step 3: Smoke Test (1 Epoch)

```bash
# Edit config for quick test
# Open: configs/experiments/baseline_0_resnet18.yaml
# Change: max_epochs: 1

# Run smoke test
uv run python scripts/train.py \
  --config configs/experiments/baseline_0_resnet18.yaml

# Expected output:
# - Device info (MPS detected)
# - Dataset stats computed (or loaded from cache)
# - Class counts and weights
# - Training progress bar
# - Validation metrics
# - Test metrics
# - Checkpoint saved

# Check checkpoint created
ls -lh checkpoints/baseline_0_resnet18_*/
```

## Step 4: Verify Smoke Test

```bash
# Check logs
ls lightning_logs/version_0/

# Check metrics
cat lightning_logs/version_0/metrics.csv

# Check model size
make check-model-size

# Expected: ResNet18 ~45 MB (under 95 MB limit ✓)
```

## Step 5: Full Training (50 Epochs)

```bash
# Restore config
# Open: configs/experiments/baseline_0_resnet18.yaml
# Change: max_epochs: 50

# Option A: Without W&B
uv run python scripts/train.py \
  --config configs/experiments/baseline_0_resnet18.yaml \
  --run-id baseline0_resnet18_full

# Option B: With W&B (recommended)
uv run python scripts/train.py \
  --config configs/experiments/baseline_0_resnet18.yaml \
  --run-id baseline0_resnet18_full \
  --wandb

# Or use Makefile (trains all 3 models)
make train-baseline-0 WANDB=--wandb
```

## Step 6: Monitor Training

### Terminal Output

```
Epoch 15/50 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:05:30
train/loss: 0.4567  train/acc: 0.8234
val/loss: 0.5123    val/acc: 0.7891    val/f1_macro: 0.7512
LR: 0.001000
```

### W&B Dashboard (if enabled)

1. Go to: wandb.ai/your-username/autolens-ai
2. Find run: baseline_0_no_augmentation
3. Watch live:
   - Loss curves
   - Accuracy curves
   - F1 scores
   - Learning rate

### What to Watch For

**Good signs:**
- Loss decreasing steadily
- Accuracy increasing
- Val/F1 > 0.6 after 10 epochs
- No NaN values

**Bad signs:**
- Loss = NaN (reduce LR or increase gradient clipping)
- Loss not decreasing (check data/labels)
- Val/F1 < 0.5 after 20 epochs (dataset issue)

## Step 7: Training Complete

Expected time: **1.5-2 hours** on M2 Pro with bf16

```bash
# Check final checkpoints
ls -lh checkpoints/baseline_0_resnet18_baseline0_resnet18_full/

# Should see:
# - best-epoch=XX-val_f1_macro=0.XXXX.ckpt (top 3)
# - best_loss-epoch=XX-val_loss=0.XXXX.ckpt
# - last.ckpt
```

## Step 8: Evaluate Results

```bash
# Check model size
make check-model-size

# Load best checkpoint
python
>>> import torch
>>> ckpt = torch.load("checkpoints/baseline_0_resnet18_baseline0_resnet18_full/best-epoch=XX-val_f1_macro=0.XXXX.ckpt")
>>> print(f"Epoch: {ckpt['epoch']}")
>>> print(f"Val F1: {ckpt['callbacks']['ModelCheckpoint']['best_model_score']}")
>>> print(f"Hyperparameters: {ckpt['hyper_parameters']}")
```

### Expected Results

**Good baseline:**
- Val F1 macro: 0.65-0.75
- Val accuracy: 0.70-0.80
- Test F1 similar to val F1 (±0.02)

**Per-class F1 (expected):**
- SUV, SEDAN, HATCHBACK: 0.75-0.85 (majority classes)
- VAN, PICK UP: 0.65-0.75 (medium)
- MICRO, STATION WAGON, OPEN WHEEL: 0.50-0.65 (minority, harder)

## Step 9: Next Steps

### If Results Good (F1 > 0.65)

```bash
# Train other models
make train-baseline-0 WANDB=--wandb
# This trains MobileNetV4 and EfficientNet-B2 too

# Or train augmentation variant
make train-baseline-1 WANDB=--wandb
```

### If Results Poor (F1 < 0.60)

1. **Check dataset:**
   ```bash
   # Review class distribution
   python
   >>> import pandas as pd
   >>> df = pd.read_csv("artifacts/dataset/splits/train.csv")
   >>> print(df['normalized_label'].value_counts())
   ```

2. **Check labels:**
   - Are labels correct?
   - Any mislabeled images?
   - Class definitions clear?

3. **Try hyperparameter tuning:**
   ```bash
   # Find better LR
   uv run python scripts/train.py \
     --config configs/experiments/baseline_0_resnet18.yaml \
     --find-lr
   
   # Update config with suggested LR
   # Re-train
   ```

### If Training Crashes

**Out of Memory:**
```yaml
# Edit config
batch_size: 16  # or 8
precision: "32-true"  # disable mixed precision
```

**Loss = NaN:**
```yaml
# Edit config
gradient_clip_val: 0.5  # more aggressive clipping
learning_rate: 0.0001  # lower LR
```

**MPS Error:**
```yaml
# Edit config
precision: "32-true"  # disable bf16
# Or set: accelerator: "cpu"
```

## Step 10: Document Results

```bash
# Create results file
cat > results/baseline_0_resnet18.txt << EOF
Model: ResNet18
Config: baseline_0_resnet18.yaml
Run ID: baseline0_resnet18_full
Date: $(date)

Results:
- Val F1 macro: [FILL]
- Val accuracy: [FILL]
- Test F1 macro: [FILL]
- Test accuracy: [FILL]
- Model size: [FILL] MB
- Training time: [FILL] hours
- Best epoch: [FILL]

Notes:
- [Any observations]
EOF
```

## Troubleshooting

### Dataset stats not computing

```bash
# Manually compute
make compute-stats

# Check output
cat artifacts/dataset/stats.json
```

### Checkpoints not saving

```bash
# Check disk space
df -h

# Check permissions
ls -la checkpoints/
```

### W&B not logging

```bash
# Check login
uv run wandb status

# Re-login
uv run wandb login

# Or train without W&B
# Remove --wandb flag
```

## Summary

**Minimum steps:**
1. ✅ Check prerequisites
2. ✅ Merge branch
3. ✅ Smoke test (1 epoch)
4. ✅ Full training (50 epochs)
5. ✅ Evaluate results

**Time:** ~2 hours training + 30 min setup/evaluation

**Next:** Train other models or proceed to Phase 4 (DINOv3)
