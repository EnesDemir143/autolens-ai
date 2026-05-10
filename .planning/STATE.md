# Project State: AutoLens AI

## Project Reference

See: `.planning/PROJECT.md` (updated 2026-05-06)

**Core value:** Generalize to unseen presentation/test images and return the correct 8-class car body type with fast, explainable prediction.
**Current focus:** Phase 6 — deploy/publish the selected final DINOv3 ViT-S/16 non-LoRA artifact. Phase 4 final selection is complete; Phase 5 UI is model-agnostic and active demo pointer now uses DINOv3 with passing smoke/latency evidence.

## Workflow Preferences

- Mode: YOLO for planning docs; execution remains sequential by user preference.
- Granularity: Standard.
- Parallelization: false.
- Commit docs: true.
- Research before planning: true.
- Plan check: true.
- Verifier: true.

## Phase Status

| Phase | Branch | Status |
|---|---|---|
| Phase 1 — Project Foundation | `main` | Complete — verified 2026-05-06 (3/3 plans) |
| Phase 2 — Dataset Research and Curation | `feat/dataset-curation-eda` | Complete — verified 2026-05-07 (4/4 plans, local datasets) |
| Phase 3 — Baseline Training Pipeline | `feat/baseline-training-pipeline` | Complete — verified 2026-05-07 (4/4 plans) |
| Phase 4 — Export, Calibration, Overnight Model Selection | `main` | Complete — DINOv3 ViT-S/16 non-LoRA selected 2026-05-10; LoRA rejected by validation/internal-test evidence; final export/calibration exists |
| Phase 5 — Gradio Demo Interface | `feat/gradio-demo-interface` | Complete — verified 2026-05-09 (4/4 plans, ONNX predictor + Gradio layout + probability chart + smoke test/docs) |
| Phase 6 — Hugging Face Spaces Demo Deploy | `main` | In progress — DINOv3/EfficientNet model repos published; active demo repointed to DINOv3 and smoke-tested; remaining: deploy/update HF Space and record URL |
| Phase 7 — Final Evidence and IEEE Report | `main` | Pending — start after Phase 6 Space URL and final evidence are frozen |

---
*Initialized: 2026-05-06*
*Last activity: 2026-05-06 — Phase 1 planned*

*Last activity: 2026-05-06 — All 6 phases planned*

*Last activity: 2026-05-06 — Phase 1 complete and verified*

*Last activity: 2026-05-07 — Phase 2 complete and verified against local datasets*

*Last activity: 2026-05-07 — Phase 3 complete and verified (4 commits, MPS device working)*

*Last activity: 2026-05-09 — GSD planning pivot: current EfficientNet-B2 deployable demo first; DINOv3/LoRA/EfficientNet EMA experiments overnight; final winner export/calibration repeated after results.*

*Last activity: 2026-05-09 — Added Makefile orchestration requirement for chosen-run export, size check, calibration, Gradio pointer update, and demo smoke order.*

*Last activity: 2026-05-09 — Added Gradio loading/progress state requirement and new Phase 6 HF Spaces deploy before Phase 7 report.*

*Last activity: 2026-05-09 — Phase 4 Plan 05 executed: EfficientNet-B2 checkpoint converted to safetensors, exported to ONNX, size-checked under 95 MB, validation-calibrated, and active demo artifact config written.*

*Last activity: 2026-05-09 — Phase 5 complete and verified (4 plans, ONNX predictor + Gradio Blocks UI + probability chart + smoke test + docs).*
*Last activity: 2026-05-10 — Final model selected: DINOv3 ViT-S/16 non-LoRA. Phase 4 plans 01-04 marked complete; `docs/dinov3_non_lora_selection.md` and split-explicit `docs/model_comparison.md` are the selection evidence.*

*Last activity: 2026-05-10 — Current phase is Phase 6. `artifacts/demo/active_model.json` now points to `artifacts/export/dinov3_vits16_latest/`; DINOv3 local smoke test passed with 44.21 ms single inference and 39.64 ms average over 5 runs. Next action: deploy/update Hugging Face Space and record URL for Phase 7.*
