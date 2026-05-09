# Phase 4 Model Selection Notes

## Current deployable candidate

The immediate Phase 4 demo lane uses the current best EfficientNet-B2 validation-F1 checkpoint:

- Run: `checkpoints/baseline_0_efficientnet_b2_20260509_135313`
- Checkpoint: `checkpoints/baseline_0_efficientnet_b2_20260509_135313/best-04-0.8994.ckpt`
- Source config: `configs/experiments/baseline_0_efficientnet_b2.yaml`
- Validation split used for calibration: `artifacts/dataset/splits/val.csv`
- Reserved split not used for calibration: `artifacts/dataset/splits/internal_test.csv`

This is not the final Phase 4 comparison winner. It is the current working candidate that unblocks the Phase 5 Gradio demo while DINOv3/LoRA/overnight comparison work remains deferred.

## Artifact flow

```text
Lightning .ckpt -> model.safetensors + metadata.json -> model.onnx -> calibration.json -> artifacts/demo/active_model.json
```

`model.safetensors` stores only model weights. `metadata.json` stores the architecture key, class mapping, preprocessing, source checkpoint, source config, validation split, and artifact pointers needed to rebuild/export the model.

## Replacement rule

When a later model wins by macro F1 and passes size/latency checks, repeat the same export/calibration sequence for that run and update only `artifacts/demo/active_model.json`. Phase 5 UI code should load the active artifact pointer rather than hardcoding a model path.
