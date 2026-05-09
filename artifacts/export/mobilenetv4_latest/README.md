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

Lightweight CNN baseline for 8-class vehicle body type classification. This repository is part of the AutoLens AI Yazlab 2 project.

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

## Calibration

Post-hoc temperature scaling was fitted on the validation split only.

- Temperature: `2.669509`
- Validation samples: 3157
- NLL: 0.54661 -> 0.32764
- ECE: 0.07737 -> 0.01263
- Accuracy: 0.88882

No internal test or instructor final test images were used for calibration.

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
