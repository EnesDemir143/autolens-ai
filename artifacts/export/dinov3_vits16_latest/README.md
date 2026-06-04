---
library_name: onnxruntime
pipeline_tag: image-classification
tags:
  - computer-vision
  - image-classification
  - cars
  - onnx
  - safetensors
  - dinov3
  - vit
license: other
---

# AutoLens DINOv3 ViT-S/16

Best-performing model in the AutoLens AI baseline comparison (F1-macro: 0.9187).

## Model

- Architecture: DINOv3 ViT-Small/16 (timm: `vit_small_patch16_dinov3`)
- Classes: 8 vehicle body types
- Source run: `baseline_0_dinov3_vits16_20260509_235120`
- Source checkpoint: `checkpoints/baseline_0_dinov3_vits16_20260509_235120/best-09-0.9238.ckpt`
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
- Center crop / model input: 256 x 256 (no crop, crop_pct=1.0)
- Mean: `[0.4429, 0.4354, 0.437]`
- Std: `[0.2456, 0.2421, 0.2449]`

## Internal Test Results

Evaluated on held-out internal test set (3170 samples, no data leakage).

- Accuracy: **0.9438**
- F1-macro: **0.9187**
- F1-weighted: **0.9442**
- Precision-macro: 0.9279
- Recall-macro: 0.9121

### Per-class Performance

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------| 
| SUV | 0.9666 | 0.9060 | 0.9353 | 670 |
| VAN | 0.9759 | 0.9780 | 0.9769 | 455 |
| STATION WAGON | 0.9474 | 0.8182 | 0.8780 | 66 |
| MICRO | 0.9000 | 0.8182 | 0.8571 | 22 |
| OPEN WHEEL / F1 | 0.9983 | 1.0000 | 0.9991 | 586 |
| SEDAN | 0.9385 | 0.9486 | 0.9435 | 836 |
| HATCHBACK | 0.7945 | 0.8722 | 0.8315 | 266 |
| PICK UP | 0.9018 | 0.9554 | 0.9278 | 269 |

## Temperature Scaling (Calibration)

Post-training calibration uses temperature scaling. Temperature was fit on the validation set only; the before/after values below are measured on the held-out internal test set.

- Temperature: **1.7211**
- Internal test ECE before: 0.0415 → after: **0.0136**
- Internal test NLL before: 0.2372 → after: **0.1784**
- Internal test mean confidence before: 0.9849 → after: **0.9555**
- Internal test accuracy: **0.9438** (3170 samples)

Calibration parameters are stored in `calibration.json`. The ONNX model outputs raw logits; apply temperature scaling at inference time.

## Artifact sizes

- `model.safetensors`: 82.373 MB
- `model.onnx`: 82.572 MB

## Files

- `model.safetensors` — weights-only model artifact
- `metadata.json` — architecture, classes, preprocessing, source run, and artifact metadata
- `model.onnx` — ONNX Runtime inference artifact
- `calibration.json` — temperature scaling parameters
- `size_check.json` — artifact size evidence

## Intended use

Educational demo and report evidence for classifying uploaded vehicle images into the 8 AutoLens body-type classes.

## Limitations

The dataset is assembled from public/open sources and may contain domain bias. Similar body styles such as hatchback, station wagon, and sedan can be ambiguous. The merged raw dataset is not redistributed in this model repository.
