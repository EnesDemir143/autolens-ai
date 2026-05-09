---
phase: 5
plan: 04
subsystem: ops
requirements-completed: [UI-06, UI-07]
duration: 15 min
completed: 2026-05-09
---

# Phase 5 Plan 04: Add UI smoke test/run docs and latency evidence — Summary

**Objective:** Add UI smoke test/run docs and latency evidence

## What was built

- `Makefile` targets:
  - `make demo` — launches Gradio server on port 7860
  - `make demo-smoke` — runs non-interactive smoke test
- `scripts/smoke_test_demo.py` — comprehensive smoke test that:
  - Verifies artifact config exists
  - Loads predictor and checks 8 class labels
  - Runs inference on synthetic image
  - Validates result structure, probability sum, latency, and confidence range
  - Benchmarks 5 iterations and records average latency
  - Saves evidence to `artifacts/demo/smoke_test_evidence.json`
- `docs/ui.md` — runbook covering quick start, model swapping, UI features, requirements traceability, smoke test evidence, and troubleshooting

## Key decisions

- Smoke test uses synthetic random image (no dependency on dataset or instructor test images).
- Makefile targets use `uv run python` for consistency with other project commands.
- Evidence JSON is reproducible and can be consumed by the final IEEE report.

## Files created/modified

- `Makefile` (modified — added `demo`, `demo-smoke` targets)
- `scripts/smoke_test_demo.py` (created)
- `docs/ui.md` (created)
- `artifacts/demo/smoke_test_evidence.json` (created by smoke test)

## Verification

- `make demo-smoke` passes with exit code 0.
- `make help` shows both demo commands.
- Smoke test evidence shows ~15 ms average inference latency on M2 Pro.

## Deviations from Plan

None — plan executed exactly as written.

## Next

Phase 5 complete. Ready for Phase 6 (Hugging Face Spaces deploy) or Phase 7 (IEEE report).
