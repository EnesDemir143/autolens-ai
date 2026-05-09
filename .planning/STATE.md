# Project State: AutoLens AI

## Project Reference

See: `.planning/PROJECT.md` (updated 2026-05-06)

**Core value:** Generalize to unseen presentation/test images and return the correct 8-class car body type with fast, explainable prediction.
**Current focus:** Phase 4 Plan 05 execution complete for the current EfficientNet-B2 deployability lane. Next execution should start only the deferred DINOv3/LoRA/benchmark/final-selection plans when explicitly requested.

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
| Phase 4 — Export, Calibration, Overnight Model Selection | `feat/model-comparison-and-selection` | In progress — Plan 05 current EfficientNet-B2 export/calibration/demo artifact complete 2026-05-09; overnight model queue and final selection still deferred |
| Phase 5 — Gradio Demo Interface | `feat/gradio-demo-interface` | Revised 2026-05-09 — build model-agnostic demo against current calibrated/exported checkpoint, then repoint to final winner |
| Phase 6 — Hugging Face Spaces Demo Deploy | `feat/hugging-face-spaces-deploy` | Planned 2026-05-09 — package/publish local Gradio ONNX demo without Docker |
| Phase 7 — Final Evidence and IEEE Report | `feat/final-evidence-and-ieee-report` | Ready after deploy evidence — existing report plans shifted from Phase 6 |

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

*Last activity: 2026-05-09 — Phase 5 execution started: Gradio Demo Interface (4 plans).*
