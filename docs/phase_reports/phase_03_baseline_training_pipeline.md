# Phase 3 Change Log — Baseline Training Pipeline

**Tarih:** 2026-05-07  
**Phase:** 03 — Baseline Training Pipeline  
**Durum:** Tamamlandı

## Kısa Özet

Phase 3'te PyTorch Lightning tabanlı CNN baseline training altyapısı kuruldu:

- AutoLensDataset ve AutoLensDataModule ile Phase 2 split CSV'lerinden veri yükleme
- MobileNetV4 Conv Medium, EfficientNet-B2, ResNet18 model factory
- AutoLensClassifier Lightning module (Accuracy, F1, Precision, Recall, Confusion Matrix metrikleri)
- MPS/CPU auto-detection ve fallback device handling
- ModelCheckpoint, EarlyStopping, LearningRateMonitor callbacks
- 5 experiment config (Baseline 0-2 varyantları)
- Comprehensive baseline training runbook

## Commit'ler

| Commit | Amaç |
|---|---|
| `924bc19` | Plan 01: Dataset ve preprocessing altyapısı — Phase 2 split CSV'lerinden veri yükleme, class-weighted loss ve weighted sampler desteği |
| `eed178c` | Plan 02: CNN model factory — MobileNetV4, EfficientNet-B2, ResNet18 baseline modelleri |
| `19c6775` | Plan 03: Lightning module ve device handling — Comprehensive metrics, MPS/CPU fallback, callbacks |
| `f317d71` | Plan 04: Experiment configs ve runbook — Baseline 0-2 varyantları, training procedure documentation |

## Değişen / Eklenen Dosyalar

### Training Infrastructure

#### `src/autolens_ai/training/dataset.py` *(yeni)*

Ne yaptığı:

- Phase 2 split CSV'lerinden image path ve label yükleme
- PIL Image loading ve transform uygulama
- Class distribution hesaplama (class weight computation için)

Neden:

- Phase 2 split artifact'larını doğrudan consume etmek için
- Class imbalance handling için train split class counts gerekli

#### `src/autolens_ai/training/datamodule.py` *(yeni)*

Ne yaptığı:

- Lightning DataModule implementation
- Train/val/test dataloader'lar
- Class-weighted loss ve WeightedRandomSampler desteği
- Inverse frequency class weight computation (max cap ile)

Neden:

- Lightning training loop için standart data interface
- MICRO, STATION WAGON gibi weak class'lar için imbalance handling

#### `src/autolens_ai/training/preprocessing.py` *(yeni)*

Ne yaptığı:

- PreprocessConfig: Resize 256, center crop 224, normalization
- Baseline 0: No augmentation (deterministic)
- Baseline 1+: Light augmentation (RandomResizedCrop, HorizontalFlip, ColorJitter)

Neden:

- Phase decision D-06: No-augmentation baseline first
- Train/val/test preprocessing consistency

#### `src/autolens_ai/training/lightning_module.py` *(yeni)*

Ne yaptığı:

- AutoLensClassifier Lightning module
- Class-weighted CrossEntropyLoss
- Comprehensive metrics: Accuracy, F1 (macro/weighted), Precision, Recall, Confusion Matrix
- AdamW optimizer + ReduceLROnPlateau scheduler

Neden:

- Assignment requirement: Accuracy, Precision, Recall, F1, normalized confusion matrix
- F1-score primary ranking metric

#### `src/autolens_ai/training/utils.py` *(yeni)*

Ne yaptığı:

- get_device(): MPS > CUDA > CPU auto-detection
- create_trainer(): Lightning Trainer factory with callbacks
- print_device_info(): Device capability reporting

Neden:

- Phase decision D-01: MPS/CPU fallback required
- Consistent callback setup (ModelCheckpoint, EarlyStopping, LearningRateMonitor)

### Model Factory

#### `src/autolens_ai/models/factory.py` *(yeni)*

Ne yaptığı:

- create_model(): MobileNetV4 Conv Medium, EfficientNet-B2, ResNet18
- timm ve torchvision integration
- Pretrained weight loading ve classifier head replacement

Neden:

- Requirements TRN-01, TRN-02, TRN-03
- Phase decision D-02, D-03: timm for MobileNetV4/EfficientNet, torchvision for ResNet18

### Experiment Configs

#### `configs/experiments/baseline_0_*.yaml` *(yeni)*

Ne yaptığı:

- Baseline 0 configs for ResNet18, MobileNetV4, EfficientNet-B2
- No augmentation, class-weighted loss
- Resize 256, center crop 224

Neden:

- Phase decision D-06: Deterministic baseline first
- Interpretable reference point for label quality and model capacity

#### `configs/experiments/baseline_1_resnet18_augmented.yaml` *(yeni)*

Ne yaptığı:

- Light augmentation variant (RandomResizedCrop, HorizontalFlip, ColorJitter)
- Same imbalance handling as Baseline 0

Neden:

- Phase decision D-07: Augmentation as explicit experiment variant
- Compare augmentation impact on macro F1 and per-class F1

#### `configs/experiments/baseline_2_resnet18_weighted_sampler.yaml` *(yeni)*

Ne yaptığı:

- WeightedRandomSampler instead of class-weighted loss
- No augmentation (fair comparison with Baseline 0)

Neden:

- Phase decision D-07: Imbalance strategy comparison
- Evaluate sampler vs loss weighting for weak classes

### Documentation

#### `docs/baseline_training_runbook.md` *(yeni)*

Ne yaptığı:

- Experiment sequence: Baseline 0 → Baseline 1 → Baseline 2 → (optional) Baseline 3
- Training procedure and commands
- Comparison metrics: macro F1, weighted F1, per-class F1, normalized confusion matrix
- Phase 4 handoff notes (DINOv3 as fourth model family)

Neden:

- Phase decision D-07: Separate no-augmentation, augmentation, imbalance, and outlier variants
- Document experiment rationale for Phase 6 IEEE report

#### `scripts/train.py` *(yeni)*

Ne yaptığı:

- Training script with YAML config loading
- Class weight computation from train split
- Trainer creation and fit/test execution

Neden:

- Executable training entry point
- Config-driven experiment execution

---

## Doğrulama

Çalışan kontroller:

```bash
# Phase 3 validation checks
uv run python /tmp/phase3_validation.py
# ✓ 1. Datamodule import succeeds
# ✓ 2. Model factory lists MobileNetV4/EfficientNet-B2/ResNet18
# ✓ 3. Lightning trainer has early stopping/checkpointing
# ✓ 4. MPS/CPU fallback code exists (detected: mps)
# ✓ 5. Baseline runbook exists
# ✓ 6. Baseline 0 config uses no augmentation, resize 256, crop 224
# ✓ 7. Follow-up configs separate augmentation and sampler variants

# Quality gates
uv run pytest
# 5 passed in 0.04s

uv run ruff check src/ scripts/ tests/
# All passed

uv run mypy src/
# Success: no issues found in 12 source files
```

Sınırlı kalan / dış bağımlılık gerektiren kontroller:

- Actual training run: Phase 3 sadece infrastructure, training Phase 4'te yapılacak
- W&B logging: Credentials henüz setup edilmedi

## Ekstra (Orijinal Plan Dışı)

| Ekstra | Açıklama |
|---|---|
| Type ignore annotations | mypy type stub issues (pandas, torchvision) için gerekli |
| ReduceLROnPlateau verbose removal | PyTorch 2.11.0'da verbose parameter deprecated |

## Notlar

- MPS device detected and working (Apple Silicon M2 Pro)
- DINOv3 intentionally deferred to Phase 4 (Phase decision D-05)
- Outlier filtering marked as low-priority optional (Phase decision D-07)
- Class weight max cap default: 5.0 (tunable if gradient instability occurs)
- F1-score (macro) is primary ranking metric
- Final model artifact must be < 95 MB

## Sonraki Adım

- Phase 4: Main DINOv3 Model and Selection
- Implement DINOv3 ViT-S/16 non-LoRA path first
- Compare four model families: MobileNetV4, EfficientNet-B2, ResNet18, DINOv3
- Select final model by macro F1, per-class behavior, artifact size, and speed
- Optional: LoRA fine-tuning variant after non-LoRA path works
