# Baseline Training Runbook

**Phase:** 3 — Baseline Training Pipeline  
**Requirements:** TRN-01, TRN-02, TRN-03, TRN-06, TRN-07  
**Phase Decisions:** D-01 through D-07

## Overview

This runbook defines the experiment order for Phase 3 CNN baseline training. These three model families (MobileNetV4 Conv Medium, EfficientNet-B2, ResNet18) serve as baseline/comparison models. Phase 4 will add DINOv3 non-LoRA and optional LoRA as the fourth model family for final comparison.

## Model Families

| Model | Requirement | Source | Purpose |
|-------|-------------|--------|---------|
| MobileNetV4 Conv Medium | TRN-01 | timm | Efficient mobile architecture baseline |
| EfficientNet-B2 | TRN-02 | timm | Compound scaling baseline |
| ResNet18 | TRN-03 | torchvision | Classic control model (< 95 MB) |

## Experiment Sequence

### Baseline 0: No Augmentation (Required)

**Goal:** Establish interpretable reference point for label quality, class imbalance, and model capacity.

**Configuration:**
- Dataset: Phase 2 outlier-unfiltered split
- Preprocessing: Resize 256 → Center crop 224 → Normalize (train split mean/std)
- Augmentation: None (deterministic for train/val/test)
- Imbalance strategy: Class-weighted CrossEntropy (inverse frequency, max weight cap 5.0)
- Models: All three (ResNet18, MobileNetV4, EfficientNet-B2)

**Config files:**
- `configs/experiments/baseline_0_resnet18.yaml`
- `configs/experiments/baseline_0_mobilenetv4.yaml`
- `configs/experiments/baseline_0_efficientnet_b2.yaml`

**Metrics to capture:**
- Macro F1, Weighted F1, Per-class F1
- Accuracy, Precision, Recall
- Normalized 8×8 confusion matrix
- Training/validation loss curves
- Class distribution and balance report

### Baseline 1: Light Augmentation (Required)

**Goal:** Compare augmentation impact against Baseline 0.

**Configuration:**
- Same as Baseline 0, except:
  - Augmentation: RandomResizedCrop (scale 0.8-1.0) + RandomHorizontalFlip (p=0.5) + ColorJitter (brightness/contrast/saturation 0.1)
- Models: Start with ResNet18, extend to others if time permits

**Config file:**
- `configs/experiments/baseline_1_resnet18_augmented.yaml`

**Comparison evidence:**
- Baseline 0 vs Baseline 1 macro F1
- Per-class F1 changes (especially MICRO, STATION WAGON, OPEN WHEEL/F1)
- Overfitting indicators (train/val gap)

### Baseline 2: Weighted Sampler vs Class-Weighted Loss (Required)

**Goal:** Compare imbalance handling strategies.

**Configuration:**
- Same as Baseline 0, except:
  - Use WeightedRandomSampler instead of class-weighted loss
  - No augmentation (for fair comparison with Baseline 0)
- Models: ResNet18

**Config file:**
- `configs/experiments/baseline_2_resnet18_weighted_sampler.yaml`

**Comparison evidence:**
- Baseline 0 (class weights) vs Baseline 2 (sampler) macro F1
- Per-class F1 for weak classes
- Training stability (loss curves)

### Baseline 3: Outlier-Filtered Split (Optional, Low Priority)

**Goal:** Evaluate if outlier/near-duplicate filtering improves metrics without excessive data loss.

**When to run:**
- Only if Baseline 0-2 results show leakage evidence
- Only if weak classes (MICRO, STATION WAGON) have sufficient remaining samples
- Only if time remains before Phase 4

**Configuration:**
- Generate filtered split artifact from Phase 2 outlier detection
- Compare against Baseline 0 using same preprocessing/model
- Document class count changes and source coverage impact

**Comparison evidence:**
- Class counts before/after filtering
- Macro F1, per-class F1 changes
- Source diversity impact

## Training Procedure

### 1. Environment Setup

```bash
# Verify device
uv run python -c "from autolens_ai.training import print_device_info; print_device_info()"

# Verify dataset splits exist
ls -lh artifacts/dataset/splits/
```

### 2. Run Baseline 0 (All Models)

```bash
# ResNet18
uv run python scripts/train.py --config configs/experiments/baseline_0_resnet18.yaml

# MobileNetV4
uv run python scripts/train.py --config configs/experiments/baseline_0_mobilenetv4.yaml

# EfficientNet-B2
uv run python scripts/train.py --config configs/experiments/baseline_0_efficientnet_b2.yaml
```

### 3. Run Baseline 1 (Augmentation)

```bash
uv run python scripts/train.py --config configs/experiments/baseline_1_resnet18_augmented.yaml
```

### 4. Run Baseline 2 (Weighted Sampler)

```bash
uv run python scripts/train.py --config configs/experiments/baseline_2_resnet18_weighted_sampler.yaml
```

### 5. Compare Results

Generate comparison report with:
- Model comparison table (macro F1, weighted F1, accuracy, artifact size)
- Per-class F1 heatmap
- Normalized confusion matrices
- Training curves overlay

## Success Criteria

From `.planning/phases/03-baseline-training-pipeline/03-VALIDATION.md`:

1. ✓ Datamodule import succeeds
2. ✓ Model factory lists MobileNetV4/EfficientNet-B2/ResNet18
3. ✓ Lightning trainer config has early stopping/checkpointing
4. ✓ MPS/CPU fallback code exists
5. ✓ Baseline runbook exists
6. ✓ Baseline 0 config uses no train augmentation
7. ✓ Follow-up configs separate augmentation and sampler variants

## Phase 4 Handoff

After Phase 3 completion:
- Three CNN baseline models trained and evaluated
- Baseline 0 (no augmentation) metrics established
- Augmentation and imbalance strategy comparisons complete
- Ready for DINOv3 non-LoRA and optional LoRA comparison in Phase 4

## Notes

- F1-score (macro) is the primary ranking metric
- Final model artifact must be < 95 MB
- DINOv3 is the intended main model; CNNs are baselines for comparison
- Keep training code explainable for demo questions
- Log all experiments to W&B or local evidence files
