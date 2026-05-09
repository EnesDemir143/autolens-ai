# Phase 5: Gradio Demo Interface - Context
**Branch:** `feat/gradio-demo-interface`

    **Gathered:** 2026-05-06
    **Status:** Ready for planning
    **Source:** `$gsd-discuss-phase 5` from user-approved six-phase roadmap

    <domain>
    ## Phase Boundary

    Gradio Demo Interface delivers only the capabilities mapped to requirement IDs: UI-01, UI-02, UI-03, UI-04, UI-05, UI-06, UI-07. It should consume the swappable inference artifact contract from Phase 4 Plan 05: first the current calibrated/exported EfficientNet-B2 artifact, then the final winner after overnight comparison. Work outside these IDs is deferred to the appropriate roadmap phase.

    </domain>

    <decisions>
    ## Implementation Decisions

    ### D-01
- Use Gradio Blocks for a clean modern single-page demo.

### D-02
- The UI must show upload/drag-drop, preview, classify action, predicted class, confidence, and 8-class probability chart.

### D-03
- Inference preprocessing must match validation preprocessing and use the selected final model artifact.

### D-04
- Layout should be presentation-ready: hero/header, two-column image/result area, clear probability chart.

### D-05
- Latency must be measured with a local smoke check.

    ### the agent's Discretion
    - Exact filenames and helper function names may be chosen during execution if they remain simple, testable, and consistent with prior phases.
    - Keep implementation small and explainable; avoid speculative abstraction.
- Do not hardcode EfficientNet-B2 in UI code; load backend/model path/class mapping/preprocessing/calibration from explicit metadata so the final selected model can replace it later.

    </decisions>

    <canonical_refs>
    ## Canonical References

    **Downstream agents MUST read these before planning or implementing.**

    ### Project requirements and roadmap
- `.planning/PROJECT.md` — project context, constraints, and stack decisions.
- `.planning/REQUIREMENTS.md` — requirement IDs mapped to this phase.
- `.planning/ROADMAP.md` — phase goal and success criteria.
- `docs/plan.md` — approved library list and dependency rationale.
- `docs/Yazlab 2- Proje 3.md` — assignment constraints.
- `.agents/skills/andrej-karpathy-skills/skills/karpathy-guidelines/SKILL.md` — simplicity and verifiability guidance.

    </canonical_refs>

    <specifics>
    ## Specific Ideas

    - Requirement IDs for this phase: UI-01, UI-02, UI-03, UI-04, UI-05, UI-06, UI-07.
    - Phase 5 may begin after Phase 4 Plan 05 produces either a current EfficientNet-B2 demo artifact or a documented blocker/fallback stub; it does not need to wait for overnight DINOv3/EMA runs.
    - Execute sequentially, preserving outputs from prior phases.
    - Write evidence files that later phases and the final report can consume.

    </specifics>

    <deferred>
    ## Deferred Ideas

    - Capabilities assigned to later roadmap phases remain deferred and must not be implemented here.

    </deferred>

    ---
    *Phase: 05-gradio-demo-interface*
    *Context gathered: 2026-05-06 via `$gsd-discuss-phase 5`*

*Context revised: 2026-05-09 — UI is model-agnostic and initially targets the current EfficientNet-B2 export/calibration artifact.*
