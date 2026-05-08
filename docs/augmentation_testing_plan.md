# Augmentation Testing Plan

## Purpose

Before running full training with augmentation (Baseline 1), validate that augmentation pipeline works correctly and produces reasonable transformations.

## Why Important

- Incorrect augmentation can hurt performance
- Too aggressive augmentation destroys important features
- Too weak augmentation provides no benefit
- Visual inspection catches bugs early

## Test Procedure

### 1. Visual Inspection Script

Create a script to visualize augmented samples:

```python
# scripts/test_augmentation.py
import matplotlib.pyplot as plt
from autolens_ai.training import AutoLensDataset, PreprocessConfig

# Load dataset with augmentation
config = PreprocessConfig()
transform = config.get_train_transform(augment=True)

dataset = AutoLensDataset(
    csv_path="artifacts/dataset/splits/train.csv",
    root_dir="artifacts/dataset/raw",
    transform=transform,
)

# Visualize 16 augmented versions of same image
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
for i, ax in enumerate(axes.flat):
    img, label = dataset[0]  # Same image, different augmentation
    ax.imshow(img.permute(1, 2, 0))
    ax.axis('off')
plt.savefig("augmentation_test.png")
```

### 2. Check List

- [ ] Images are still recognizable after augmentation
- [ ] Color jitter is not too extreme
- [ ] Random crop doesn't cut off important parts
- [ ] Horizontal flip makes sense (cars are symmetric)
- [ ] No artifacts or corruption

### 3. Baseline 0 First

**IMPORTANT:** Always run Baseline 0 (no augmentation) first!

Reasons:
1. Establishes baseline performance
2. Validates dataset quality
3. Checks for label errors
4. Provides comparison point

### 4. Augmentation Comparison

After Baseline 0 completes:

```bash
# Run Baseline 1 (with augmentation)
make train-baseline-1

# Compare metrics:
# - If F1 improves: augmentation helps
# - If F1 drops: augmentation too aggressive or dataset already good
# - If F1 same: augmentation neutral
```

### 5. Augmentation Tuning (If Needed)

If Baseline 1 underperforms, try:

**Weaker augmentation:**
```yaml
# configs/experiments/baseline_1_weak_aug.yaml
transforms.RandomResizedCrop(224, scale=(0.9, 1.0))  # Less crop
transforms.ColorJitter(brightness=0.05, contrast=0.05)  # Less jitter
```

**Stronger augmentation:**
```yaml
# configs/experiments/baseline_1_strong_aug.yaml
transforms.RandomResizedCrop(224, scale=(0.7, 1.0))  # More crop
transforms.ColorJitter(brightness=0.2, contrast=0.2)  # More jitter
transforms.RandomRotation(10)  # Add rotation
```

## Current Augmentation (Baseline 1)

```python
# Light augmentation (conservative)
transforms.RandomResizedCrop(224, scale=(0.8, 1.0))
transforms.RandomHorizontalFlip(p=0.5)
transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)
```

**Assessment:** Conservative, safe for car classification.

## Decision Tree

```
Start
  ↓
Run Baseline 0 (no aug)
  ↓
Baseline 0 F1 > 0.7?
  ├─ Yes → Run Baseline 1 (light aug)
  │         ↓
  │       Baseline 1 F1 > Baseline 0?
  │         ├─ Yes → Use augmentation ✓
  │         └─ No → Skip augmentation, use Baseline 0
  │
  └─ No → Fix dataset/model first
            (augmentation won't help bad data)
```

## Notes

- Augmentation is applied **only to training set**
- Validation and test always use deterministic preprocessing
- This ensures fair comparison
- Augmentation increases effective dataset size
- Most useful when dataset is small (<10k samples)

## Status

- [ ] Augmentation test script created
- [ ] Visual inspection completed
- [ ] Baseline 0 completed
- [ ] Baseline 1 ready to run
- [ ] Results compared

## Recommendation

**For AutoLens AI:**
- Dataset size: ~32k samples (medium)
- Class imbalance: Yes (MICRO, STATION WAGON weak)
- Augmentation: **Recommended** but validate with Baseline 0 first

**Expected outcome:**
- Baseline 0: F1 ~0.65-0.75 (depends on data quality)
- Baseline 1: F1 +0.02-0.05 improvement (if augmentation helps)
