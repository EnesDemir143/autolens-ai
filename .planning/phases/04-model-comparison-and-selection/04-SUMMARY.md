---
phase: 4
phase_name: Export, Calibration, Overnight Model Selection
status: complete
completed_at: 2026-05-10
final_model: DINOv3 ViT-S/16 non-LoRA
source_run: baseline_0_dinov3_vits16_20260509_235120
final_artifact_dir: artifacts/export/dinov3_vits16_latest
---

# Phase 4 Summary — Final Model Selection Complete

## Outcome

Phase 4 is now complete. The project selected **DINOv3 ViT-S/16 non-LoRA** as the final model candidate.

## Completed Plan Status

| Plan | Status | Evidence |
|---|---|---|
| 04-01 Evaluation suite | Complete | `docs/model_comparison.md`, `artifacts/export/*/internal_test_results.json`, calibration `test_eval` fields |
| 04-02 DINOv3 path | Complete | Non-LoRA and LoRA run metrics exist; non-LoRA selected; LoRA rejected with rationale |
| 04-03 Model benchmark | Complete | Split-explicit benchmark table with Accuracy, Balanced Accuracy, MCC, Precision, Recall, F1, size |
| 04-04 Final selection | Complete | `docs/dinov3_non_lora_selection.md` and `docs/model_comparison.md` |
| 04-05 Export/calibration lane | Complete | EfficientNet deploy lane completed earlier; DINOv3 final export/calibration now exists in `artifacts/export/dinov3_vits16_latest/` |

## Final Model Evidence

| Item | Value |
|---|---|
| Selected model | DINOv3 ViT-S/16 non-LoRA |
| Source run | `baseline_0_dinov3_vits16_20260509_235120` |
| Checkpoint | `checkpoints/baseline_0_dinov3_vits16_20260509_235120/best-09-0.9238.ckpt` |
| Export folder | `artifacts/export/dinov3_vits16_latest/` |
| Internal-test Accuracy | 0.9438 |
| Internal-test F1-macro | 0.9187 |
| Internal-test F1-weighted | 0.9442 |
| Safetensors size | 82.4 MB |
| ONNX size | 82.6 MB |
| Calibration | Temperature fit on `val`; ECE/NLL evaluated on `internal_test` |

## LoRA Decision

LoRA was not selected because it did not beat the non-LoRA run on validation F1 and its completed internal-test evidence was much weaker. The non-LoRA model also has the complete deployment evidence chain already available.

See: `docs/dinov3_non_lora_selection.md`.

## Handoff

Current project phase should move to **Phase 6 — Hugging Face Spaces Demo Deploy and Model Publishing**.

Immediate next work:

1. Repoint `artifacts/demo/active_model.json` from EfficientNet-B2 to `artifacts/export/dinov3_vits16_latest/`.
2. Smoke-test the local demo with the DINOv3 active artifact.
3. Package/update the Hugging Face Space with the final DINOv3 ONNX artifact.
4. Record final Space/model URLs for Phase 7.
5. Freeze evidence and generate the IEEE report.
