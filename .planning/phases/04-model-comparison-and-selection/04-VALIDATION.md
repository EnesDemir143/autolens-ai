# Phase 4 Validation Strategy

**Phase:** 4 — Main DINOv3 Model and Selection
**Created:** 2026-05-06
**Branch:** `feat/model-comparison-and-selection`

## Required Evidence

- metric suite computes required metrics.
- plots are generated to artifacts directory.
- DINOv3 main-path access/fallback is documented.
- benchmark table contains DINOv3 vs baseline F1/size/latency.
- selected final model is DINOv3 unless blocked/inferior on F1, size, or latency; selected artifact is under 95 MB or mitigation documented.
