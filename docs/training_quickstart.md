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

3. Dataset statistics (automatic):
- First training run computes mean/std from train split
- Saved to `artifacts/dataset/stats.json`
- Subsequent runs reuse cached stats
- To recompute: `rm artifacts/dataset/stats.json`

## Training Commands

### Quick Start (Recommended)

```bash
# Train all Baseline 0 models (ResNet18, MobileNetV4, EfficientNet-B2)
make train-baseline-0

# With W&B logging
make train-baseline-0 WANDB=--wandb

# Train single model
uv run python scripts/train.py --config configs/experiments/baseline_0_resnet18.yaml

# With custom run ID
uv run python scripts/train.py \
  --config configs/experiments/baseline_0_resnet18.yaml \
  --run-id my_experiment_v1
```

### All Training Targets

```bash
# Baseline 0: No augmentation (3 models)
make train-baseline-0

# Baseline 1: Light augmentation (ResNet18)
make train-baseline-1

# Baseline 2: Weighted sampler (ResNet18)
make train-baseline-2

# Run all baselines sequentially
make train-all-baselines
```

### Resume Training

If training stops (crash, early stopping, manual interrupt):

```bash
# Using Makefile
make resume-training \
  CONFIG=configs/experiments/baseline_0_resnet18.yaml \
  RUN_ID=20260507_234800

# Direct command
uv run python scripts/train.py \
  --config configs/experiments/baseline_0_resnet18.yaml \
  --run-id 20260507_234800 \
  --resume

# With W&B
make resume-training \
  CONFIG=configs/experiments/baseline_0_resnet18.yaml \
  RUN_ID=20260507_234800 \
  WANDB=--wandb
```

**How it works:**
- Loads `last.ckpt` from specified run
- Continues from saved epoch (e.g., epoch 30 → 100)
- Preserves optimizer state, LR schedule, metrics
- Saves to same checkpoint directory

### Available Flags

```bash
--config PATH          # Required: Path to experiment config YAML
--wandb                # Enable W&B logging
--wandb-project NAME   # W&B project name (default: autolens-ai)
--run-id ID            # Custom run identifier (default: timestamp)
--resume               # Resume from last checkpoint (requires --run-id)
```

## What Gets Saved

### Checkpoints (Per Run)

Each run creates timestamped directory:
```
checkpoints/
├── baseline_0_resnet18_20260507_234800/
│   ├── best-epoch=15-val_f1_macro=0.7234.ckpt  # Top 3 by F1
│   ├── best-epoch=18-val_f1_macro=0.7456.ckpt
│   ├── best-epoch=22-val_f1_macro=0.7512.ckpt
│   ├── best_loss-epoch=12-val_loss=0.4567.ckpt # Best by loss
│   └── last.ckpt                                # Last epoch (for resume)
├── baseline_0_resnet18_20260507_235200/         # Second run
└── baseline_0_mobilenetv4_20260508_001500/      # Different model
```

**Checkpoint contents:**
- Model weights (state_dict)
- Optimizer state
- LR scheduler state
- Epoch number
- **Hyperparameters** (model_name, lr, batch_size, etc.)
- Training metrics history

### Dataset Statistics

```json
// artifacts/dataset/stats.json
{
  "mean": [0.462, 0.451, 0.438],
  "std": [0.251, 0.247, 0.245],
  "description": "Dataset normalization statistics computed from train split"
}
```

### Metrics Logged

**Training (every epoch):**
- train/loss
- train/acc

**Validation (every epoch):**
- val/loss
- val/acc
- val/f1_macro ⭐ (primary ranking metric)
- val/f1_weighted
- val/precision
- val/recall

**Test (end of training):**
- test/loss
- test/acc
- test/f1_macro
- test/f1_weighted
- test/precision
- test/recall
- test/confusion_matrix (8×8 normalized)

### Logs

**Without W&B:**
- CSV logs: `lightning_logs/version_X/metrics.csv`
- Hyperparameters: `lightning_logs/version_X/hparams.yaml`

**With W&B:**
- Online dashboard: wandb.ai/your-username/autolens-ai
- Local cache: `wandb/`
- Run comparison, hyperparameter tracking, confusion matrices

## Monitoring Training

### Progress Bar (Rich)

```
Epoch 15/50 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:05:30
train/loss: 0.4567  train/acc: 0.8234
val/loss: 0.5123    val/acc: 0.7891    val/f1_macro: 0.7512
```

### Early Stopping

- Monitors: `val/loss`
- Patience: 10 epochs
- Stops automatically if no improvement

### Learning Rate Schedule

- Optimizer: AdamW (lr=0.001, weight_decay=0.0001)
- Scheduler: ReduceLROnPlateau
  - Reduces LR by 0.5 when val/loss plateaus
  - Patience: 3 epochs

### Callbacks

- **ModelCheckpoint**: Saves top 3 by F1, best by loss, last
- **EarlyStopping**: Stops if val/loss doesn't improve (patience=10)
- **LearningRateMonitor**: Logs LR changes
- **RichProgressBar**: Beautiful terminal progress

## Expected Training Time

On Apple Silicon M2 Pro with MPS:

| Model | Epochs | Time (est.) |
|-------|--------|-------------|
| ResNet18 | 50 | 2-3 hours |
| MobileNetV4 Conv Medium | 50 | 3-4 hours |
| EfficientNet-B2 | 50 | 4-5 hours |

**Note:** Early stopping may reduce actual time.

## Comparing Runs

### W&B Dashboard

1. Go to wandb.ai/your-username/autolens-ai
2. Select multiple runs
3. Compare:
   - Val/F1 curves
   - Loss curves
   - Confusion matrices
   - Hyperparameters side-by-side

### Local Checkpoints

```bash
# List all runs
ls -lh checkpoints/

# Check best F1 for each run
grep "val_f1_macro" checkpoints/*/best-*.ckpt

# Load checkpoint in Python
import torch
ckpt = torch.load("checkpoints/baseline_0_resnet18_20260507_234800/best-*.ckpt")
print(ckpt['hyper_parameters'])
print(ckpt['epoch'])
```

## Troubleshooting

### MPS Not Detected

```bash
# Check PyTorch MPS availability
uv run python -c "import torch; print(torch.backends.mps.is_available())"

# If False, training will use CPU (slower)
```

### Out of Memory

Reduce batch size in config:
```yaml
batch_size: 16  # or 8
```

### W&B Login

```bash
# Login once
uv run wandb login

# Or set environment variable
export WANDB_API_KEY=your_key_here
```

### Dataset Not Found

Verify paths in config match your dataset location:
```yaml
data_root: artifacts/dataset/raw
train_csv: artifacts/dataset/splits/train.csv
val_csv: artifacts/dataset/splits/val.csv
test_csv: artifacts/dataset/splits/internal_test.csv
```

### Resume Not Working

Check that:
1. `--run-id` matches existing checkpoint directory
2. `last.ckpt` exists in that directory
3. Config file is the same as original run

```bash
# List available runs
ls checkpoints/

# Check if last.ckpt exists
ls checkpoints/baseline_0_resnet18_20260507_234800/last.ckpt
```

## Advanced Usage

### Custom Hyperparameters

Edit config file before training:
```yaml
# configs/experiments/my_custom_config.yaml
learning_rate: 0.0001  # Lower LR
batch_size: 64         # Larger batch
max_epochs: 100        # More epochs
```

### Multiple Runs with Different Settings

```bash
# Run 1: Default
uv run python scripts/train.py \
  --config configs/experiments/baseline_0_resnet18.yaml \
  --run-id default

# Run 2: Lower LR (edit config first)
uv run python scripts/train.py \
  --config configs/experiments/baseline_0_resnet18_lr0001.yaml \
  --run-id lr0001

# Run 3: Larger batch (edit config first)
uv run python scripts/train.py \
  --config configs/experiments/baseline_0_resnet18_bs64.yaml \
  --run-id bs64
```

### Recompute Dataset Stats

```bash
# Delete cached stats
rm artifacts/dataset/stats.json

# Next training will recompute
make train-baseline-0
```

## Next Steps After Training

1. **Compare Results:**
   - Check val/f1_macro scores across models
   - Review per-class F1 from test confusion matrix
   - Compare model sizes (must be < 95 MB)

2. **Select Best Model:**
   - Primary metric: macro F1
   - Secondary: per-class F1 (especially MICRO, STATION WAGON, OPEN WHEEL/F1)
   - Tertiary: model size and inference speed

3. **Document Findings:**
   - Save metrics to `docs/phase_reports/`
   - Note which model performs best
   - Identify weak classes for Phase 4 improvement

4. **Phase 4 Preparation:**
   - Best CNN baseline becomes comparison point
   - DINOv3 will be compared against these baselines
   - Keep checkpoint paths for final model selection
