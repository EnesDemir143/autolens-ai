# Phase 5 Validation Strategy

**Phase:** 5 — Gradio Demo Interface
**Created:** 2026-05-06
**Branch:** `feat/gradio-demo-interface`

## Required Evidence

- Gradio app launches.
- upload component exists.
- preview and result are visible.
- probability chart has 8 classes.
- latency smoke result is recorded.


## 2026-05-09 Pivot Evidence

- UI loads a swappable model artifact contract rather than hardcoding EfficientNet-B2 or DINOv3.
- Initial smoke test may use the current calibrated/exported EfficientNet-B2 artifact from Phase 4 Plan 05.
- Repointing to the final selected model after overnight experiments should require updating artifact/config paths, not rewriting Gradio layout or prediction display code.

- Makefile demo targets launch Gradio with the active artifact config and provide a smoke-check command before presentation.

- Loading/progress/status state is visible while inference runs; slow CPU inference does not leave the user with a frozen-looking UI.
