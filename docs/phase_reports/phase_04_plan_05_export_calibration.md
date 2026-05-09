# Phase 4 Plan 05 — Export and Calibration Report

**Date:** 2026-05-09  
**Scope:** Current EfficientNet-B2 deployability lane only. DINOv3, LoRA, overnight benchmarks, final selection, Docker, API work, and instructor/internal final-test data are out of scope.

## Outcome

Converted the current best EfficientNet-B2 Lightning checkpoint into a clean deployment/share artifact set and active demo pointer:

- Source checkpoint: `checkpoints/baseline_0_efficientnet_b2_20260509_135313/best-04-0.8994.ckpt`
- Weights-only artifact: `artifacts/export/efficientnet_b2_current/model.safetensors`
- Metadata: `artifacts/export/efficientnet_b2_current/metadata.json`
- ONNX export: `artifacts/export/efficientnet_b2_current/model.onnx`
- Optional simplified ONNX: `artifacts/export/efficientnet_b2_current/model.simplified.onnx`
- Calibration: `artifacts/export/efficientnet_b2_current/calibration.json`
- Active demo pointer: `artifacts/demo/active_model.json`

## Validation evidence

| Check | Result |
|---|---|
| Checkpoint locator | Selected `best-04-0.8994.ckpt` from `baseline_0_efficientnet_b2_20260509_135313` |
| ONNX Runtime smoke | Passed with `CPUExecutionProvider`, input `image`, output shape `[1, 8]` |
| Size limit | `model.safetensors` 29.72 MB, `model.onnx` 29.37 MB, both under 95 MB |
| Calibration split | Used `artifacts/dataset/splits/val.csv` only; `internal_test` is reserved |
| Calibration samples | 3,157 validation images |
| Temperature | `1.779860258102417` |
| NLL | 0.29796 → 0.24160 |
| ECE | 0.03823 → 0.01043 |
| Accuracy | 0.92398 before/after |
| Make help | Lists `checkpoint-to-safetensors`, `export-model`, `size-check`, `calibrate-model`, `prepare-demo-artifact`, `deploy-run-to-demo` |

## Commands run

```bash
make graphify-update
make checkpoint-to-safetensors
make export-model
make size-check
make calibrate-model
make prepare-demo-artifact
make help
uv run ruff format src/autolens_ai/inference scripts/checkpoint_to_safetensors.py scripts/export_model.py scripts/check_artifact_size.py scripts/calibrate_model.py scripts/prepare_demo_artifact.py tests/test_phase4_artifact_scripts.py
uv run ruff check src/autolens_ai/inference scripts/checkpoint_to_safetensors.py scripts/export_model.py scripts/check_artifact_size.py scripts/calibrate_model.py scripts/prepare_demo_artifact.py tests/test_phase4_artifact_scripts.py
uv run pytest tests/test_phase4_artifact_scripts.py tests/test_package_import.py
uv run mypy src/autolens_ai/inference
```

## Notes

- `model.safetensors` contains only model weights, not optimizer/callback/datamodule state.
- `metadata.json` contains class labels, preprocessing, source config/checkpoint, split paths, and artifact pointers.
- `calibrate_model.py` rejects calibration paths containing `internal_test`, `final`, or `instructor`.
- The final model swap should update `artifacts/demo/active_model.json`; Phase 5 UI code should not hardcode the model path.
