# Deployment Artifact Runbook

## Current Phase 4 sequence

Use these targets for the current EfficientNet-B2 candidate:

```bash
make checkpoint-to-safetensors
make export-model
make size-check
make calibrate-model
make prepare-demo-artifact
```

Or run the full safe sequence:

```bash
make deploy-run-to-demo
```

The default checkpoint locator chooses the highest-score `best-*.ckpt` under `checkpoints/baseline_0_efficientnet_b2_*`. To deploy a specific run:

```bash
make deploy-run-to-demo CHECKPOINT=checkpoints/.../best-XX-0.XXXX.ckpt EXPORT_DIR=artifacts/export/my_run
```

## Inputs and outputs

- Input checkpoint: Lightning `.ckpt`; retained for resume/debugging.
- Export directory: `artifacts/export/efficientnet_b2_current` by default.
- Clean weights: `model.safetensors`.
- Rebuild context: `metadata.json`.
- Deploy model: `model.onnx` and optional `model.simplified.onnx`.
- Size evidence: `size_report.json` and `size_check.json`.
- Calibration: `calibration.json`, fit only on `val.csv`.
- Active demo pointer: `artifacts/demo/active_model.json`.

## Data leakage guard

`calibrate-model` reads the validation split from metadata. It rejects paths containing `internal_test`, `final`, or `instructor` so instructor/final data is not used for tuning.
