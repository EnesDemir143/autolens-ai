---
library_name: onnxruntime
pipeline_tag: image-classification
tags:
  - computer-vision
  - image-classification
  - cars
  - onnx
  - safetensors
license: other
---

# AutoLens EfficientNet-B2 Candidate

Current strongest CNN candidate for the AutoLens AI demo artifact lane.

## Model

- Architecture: EfficientNet-B2
- Classes: 8 vehicle body types
- Source run: `baseline_0_efficientnet_b2_20260509_135313`
- Source checkpoint: `checkpoints/baseline_0_efficientnet_b2_20260509_135313/best-04-0.8994.ckpt`
- Export format: `model.safetensors` plus `metadata.json`, with ONNX for deployment

## Classes

- SUV
- VAN
- STATION WAGON
- MICRO
- OPEN WHEEL / F1
- SEDAN
- HATCHBACK
- PICK UP

## Preprocessing

- RGB input
- Resize: 256
- Center crop / model input: 224 x 224
- Mean: `[0.4429, 0.4354, 0.437]`
- Std: `[0.2456, 0.2421, 0.2449]`

## Internal Test Results

Evaluated on held-out internal test set (3170 samples, no data leakage).

- Accuracy: **0.9202**
- F1-macro: **0.8982**
- F1-weighted: 0.9201
- Precision-macro: 0.9056
- Recall-macro: 0.8923

### Per-class Performance

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| SUV | 0.9213 | 0.9090 | 0.9151 | 670 |
| VAN | 0.9450 | 0.9824 | 0.9634 | 455 |
| STATION WAGON | 0.8033 | 0.7424 | 0.7717 | 66 |
| MICRO | 0.9545 | 0.9545 | 0.9545 | 22 |
| OPEN WHEEL / F1 | 0.9948 | 0.9846 | 0.9897 | 586 |
| SEDAN | 0.9023 | 0.9282 | 0.9151 | 836 |
| HATCHBACK | 0.7662 | 0.8008 | 0.7831 | 266 |
| PICK UP | 0.9574 | 0.8364 | 0.8929 | 269 |

## Temperature Scaling (Calibration)

Post-training calibration uses temperature scaling. Temperature was fit on the validation set only; the before/after values below are measured on the held-out internal test set.

- Temperature: **1.7799**
- Internal test ECE before: 0.0393 → after: **0.0107**
- Internal test NLL before: 0.2745 → after: **0.2301**
- Internal test mean confidence before: 0.9595 → after: **0.9168**
- Internal test accuracy: **0.9202** (3170 samples)

Calibration parameters are stored in `calibration.json`. The ONNX model outputs raw logits; apply temperature scaling at inference time.

## Artifact sizes

- `model.safetensors`: 29.722 MB
- `model.onnx`: 29.371 MB

## Files

- `model.safetensors` — weights-only model artifact
- `metadata.json` — architecture, classes, preprocessing, source run, and artifact metadata
- `model.onnx` — ONNX Runtime inference artifact
- `model.simplified.onnx` — optional simplified ONNX graph when available
- `size_report.json` / `size_check.json` — artifact size and ONNX smoke evidence

## Intended use

Educational demo and report evidence for classifying uploaded vehicle images into the 8 AutoLens body-type classes.

## Limitations

The dataset is assembled from public/open sources and may contain domain bias. Similar body styles such as hatchback, station wagon, and sedan can be ambiguous. The merged raw dataset is not redistributed in this model repository.
