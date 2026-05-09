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

# AutoLens ResNet18 Baseline

Baseline CNN model for 8-class vehicle body type classification.

## Model

- Architecture: ResNet18
- Classes: 8 vehicle body types
- Source run: `baseline_0_resnet18_20260509_092916`
- Source checkpoint: `checkpoints/baseline_0_resnet18_20260509_092916/best-12-0.8414.ckpt`
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

- Accuracy: **0.8968**
- F1-macro: **0.8590**
- F1-weighted: 0.8963
- Precision-macro: 0.8887
- Recall-macro: 0.8361

### Per-class Performance

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| SUV | 0.9104 | 0.9104 | 0.9104 | 670 |
| VAN | 0.9165 | 0.9165 | 0.9165 | 455 |
| STATION WAGON | 0.6212 | 0.6212 | 0.6212 | 66 |
| MICRO | 0.7727 | 0.7727 | 0.7727 | 22 |
| OPEN WHEEL / F1 | 0.9949 | 0.9949 | 0.9949 | 586 |
| SEDAN | 0.8800 | 0.9031 | 0.8914 | 836 |
| HATCHBACK | 0.7306 | 0.7444 | 0.7374 | 266 |
| PICK UP | 0.9328 | 0.8253 | 0.8757 | 269 |

## Artifact sizes

- `model.safetensors`: 42.698 MB
- `model.onnx`: 42.642 MB

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
