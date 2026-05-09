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

# AutoLens MobileNetV4 Baseline

Lightweight CNN baseline for 8-class vehicle body type classification.

## Model

- Architecture: MobileNetV4 Conv Medium
- Classes: 8 vehicle body types
- Source run: `baseline_0_mobilenetv4_20260509_103508`
- Source checkpoint: `checkpoints/baseline_0_mobilenetv4_20260509_103508/best-22-0.8350.ckpt`
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

- Accuracy: **0.8905**
- F1-macro: **0.8510**
- F1-weighted: 0.8912
- Precision-macro: 0.8458
- Recall-macro: 0.8592

### Per-class Performance

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| SUV | 0.8284 | 0.8724 | 0.8498 | 670 |
| VAN | 0.9495 | 0.9495 | 0.9495 | 455 |
| STATION WAGON | 0.7121 | 0.7121 | 0.7121 | 66 |
| MICRO | 0.8636 | 0.8636 | 0.8636 | 22 |
| OPEN WHEEL / F1 | 0.9863 | 0.9863 | 0.9863 | 586 |
| SEDAN | 0.8811 | 0.9043 | 0.8926 | 836 |
| HATCHBACK | 0.7050 | 0.7368 | 0.7206 | 266 |
| PICK UP | 0.8633 | 0.8922 | 0.8775 | 269 |

## Artifact sizes

- `model.safetensors`: 32.517 MB
- `model.onnx`: 32.12 MB
- Project limit: 95 MB

## Files

- `model.safetensors` — weights-only model artifact
- `metadata.json` — architecture, classes, preprocessing, source run, and artifact metadata
- `model.onnx` — ONNX Runtime inference artifact
- `model.simplified.onnx` — optional simplified ONNX graph when available
- `calibration.json` — temperature scaling metadata
- `size_report.json` / `size_check.json` — artifact size and ONNX smoke evidence

## Intended use

Educational demo and report evidence for classifying uploaded vehicle images into the 8 AutoLens body-type classes.

## Limitations

The dataset is assembled from public/open sources and may contain domain bias. Similar body styles such as hatchback, station wagon, and sedan can be ambiguous. The merged raw dataset is not redistributed in this model repository.
