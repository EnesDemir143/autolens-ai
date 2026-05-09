# Phase 4: Export, Calibration, Overnight Model Selection - Context
**Branch:** `feat/model-comparison-and-selection`

**Gathered:** 2026-05-06
**Revised:** 2026-05-09
**Status:** Ready for execution with demo-first pivot
**Source:** Original `$gsd-discuss-phase 4`, revised from user direction to finish current-checkpoint deployability before overnight experiments.

<domain>
## Phase Boundary

Phase 4 now has two lanes:

1. **Immediate deployability lane:** make the current best EfficientNet-B2 checkpoint usable for demo by exporting to ONNX, checking artifact size, fitting validation-only temperature scaling, and producing inference metadata.
2. **Overnight comparison lane:** run DINOv3 ViT-S non-LoRA, DINOv3 LoRA, and EfficientNet-B2 augmentation+EMA+focal-loss experiments, then select the best model by macro F1 with size/latency constraints.

The immediate lane may unblock Phase 5 before all overnight experiments finish. Final selection still belongs to Phase 4, and after the overnight winner is known, export/calibration are repeated only for that winning artifact.

</domain>

<decisions>
## Implementation Decisions

### D-01
- Use the current best EfficientNet-B2 checkpoint as the first deployable candidate so the project can reach a working ONNX/calibrated/Gradio demo quickly.

### D-02
- Fit temperature scaling on validation predictions only. Do not use internal_test or instructor final test images for calibration or tuning.

### D-03
- Queue long runs overnight: DINOv3 ViT-S non-LoRA first, DINOv3 LoRA second, and EfficientNet-B2 with augmentation + EMA + focal loss as an enhanced CNN candidate.

### D-04
- DINOv3 remains a required comparison family, but it is not allowed to block the first working demo if EfficientNet-B2 is already exportable and under the size limit.

### D-05
- Rank completed candidates by macro F1 first, then balanced accuracy, MCC, per-class behavior, artifact size, and latency. Baseline/current EfficientNet-B2 can win if it is better or more deployable.

### D-06
- Keep Gradio model-agnostic: model path, class mapping, preprocessing metadata, and calibration temperature must be artifact/config driven so the final winner can replace the initial checkpoint without UI rewrites.

### D-07
- Final model must be under 95 MB or have a documented compression/export mitigation.

### the agent's Discretion
- Exact filenames and helper function names may be chosen during execution if they remain simple, testable, and consistent with prior phases.
- Keep implementation small and explainable; avoid speculative abstraction.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project requirements and roadmap
- `.planning/PROJECT.md` — project context, constraints, and stack decisions.
- `.planning/REQUIREMENTS.md` — requirement IDs mapped to this phase.
- `.planning/ROADMAP.md` — revised phase goal and success criteria.
- `docs/baseline_training_runbook.md` — existing CNN experiment runbook.
- `docs/augmentation_testing_plan.md` — augmentation/EMA/focal-loss intent if present.
- `docs/plan.md` and `docs/dependencies.md` — approved library list and dependency rationale.
- `docs/Yazlab 2- Proje 3.md` — assignment constraints.
- `.agents/skills/andrej-karpathy-skills/skills/karpathy-guidelines/SKILL.md` — simplicity and verifiability guidance.

</canonical_refs>

<specifics>
## Specific Ideas

- Immediate sequence: current EfficientNet-B2 checkpoint → ONNX export → optional onnxsim simplification → artifact size check → validation-only temperature scaling → inference metadata for Phase 5.
- Overnight sequence: DINOv3 ViT-S non-LoRA → DINOv3 LoRA → EfficientNet-B2 augmentation+EMA+focal-loss.
- Final sequence after overnight: compare all finished runs → pick winner → repeat ONNX export/calibration for the winner → update model pointer used by Gradio.
- Required ranking metrics include macro F1, weighted F1, balanced accuracy, MCC, per-class F1, normalized confusion matrix, artifact size, and latency.
- Required calibration evidence should include NLL/ECE or equivalent confidence diagnostics before and after temperature scaling when available.

</specifics>

<deferred>
## Deferred Ideas

- IEEE report generation remains Phase 6.
- Grad-CAM, Optuna, or extra model families remain optional unless the core demo and required comparison evidence are already safe.

</deferred>

---
*Phase: 04-model-comparison-and-selection*
*Context gathered: 2026-05-06 via `$gsd-discuss-phase 4`*
*Context revised: 2026-05-09 via GSD planning pivot*
