# Phase 4 Validation Strategy

**Phase:** 4 — Main DINOv3 Model and Selection
**Created:** 2026-05-06
**Branch:** `feat/model-comparison-and-selection`

## Required Evidence

- metric suite computes required metrics.
- plots are generated to artifacts directory.
- DINOv3 main-path access/fallback is documented.
- benchmark table contains four model-family results: MobileNetV4 Conv Medium, EfficientNet-B2, ResNet18, and DINOv3 non-LoRA, with optional DINOv3 LoRA only after non-LoRA works.
- metrics include macro F1, weighted F1, balanced accuracy, MCC, per-class metrics, normalized confusion matrix, size, and latency.
- selected final model is DINOv3 unless blocked/inferior on F1, size, or latency; selected artifact is under 95 MB or mitigation documented.
