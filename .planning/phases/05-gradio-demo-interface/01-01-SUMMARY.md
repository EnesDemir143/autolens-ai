---
phase: 5
plan: 01
subsystem: inference
requirements-completed: [UI-03, UI-04, UI-07]
duration: 10 min
completed: 2026-05-09
---

# Phase 5 Plan 01: Create swappable calibrated model loading and prediction adapter — Summary

**Objective:** Create swappable calibrated model loading and prediction adapter

## What was built

- `src/autolens_ai/inference/onnx_predictor.py` — ONNX Runtime predictor that loads the active artifact JSON config and runs inference on PIL images.
- The predictor is **model-agnostic**: it reads `model_path`, `metadata_path`, `class_labels`, `display_labels`, `preprocessing`, and `temperature` from the artifact config. Swapping the model only requires updating `artifacts/demo/active_model.json`.
- `_preprocess()` replicates the validation preprocessing pipeline (resize, center crop, normalize) using the metadata-driven mean/std.
- `predict()` returns a structured dict with predicted class, confidence, probability distribution, latency, and temperature.

## Key decisions

- Used ONNX Runtime (CPU) for maximum compatibility across MPS/CPU and for HF Spaces deployment.
- Temperature scaling is applied at inference time via softmax division, preserving the calibration fitted in Phase 4.

## Files created/modified

- `src/autolens_ai/inference/onnx_predictor.py` (created)
- `src/autolens_ai/inference/__init__.py` (modified — exports `ONNXPredictor`)

## Verification

- `make demo-smoke` loads the predictor from `artifacts/demo/active_model.json` and runs inference successfully.
- Smoke test reports ~15 ms average latency on M2 Pro.

## Deviations from Plan

None — plan executed exactly as written.

## Next

Ready for Plan 02 (Gradio Blocks layout).
