# Phase 4: Main DINOv3 Model and Selection - Context
**Branch:** `feat/model-comparison-and-selection`

    **Gathered:** 2026-05-06
    **Status:** Ready for planning
    **Source:** `$gsd-discuss-phase 4` from user-approved six-phase roadmap

    <domain>
    ## Phase Boundary

    Main DINOv3 Model and Selection delivers only the capabilities mapped to requirement IDs: TRN-04, TRN-05, EVAL-01, EVAL-02, EVAL-03, EVAL-04, EVAL-05, EVAL-06, EVAL-07. Work outside these IDs is deferred to the appropriate roadmap phase.

    </domain>

    <decisions>
    ## Implementation Decisions

    ### D-01
- Treat DINOv3 ViT-S/16 as the intended main model and validate gated Hugging Face access for facebook/dinov3-vits16-pretrain-lvd1689m before planning training assumptions.

### D-02
- Evaluate DINOv3 without LoRA first; LoRA is optional only after non-LoRA works.

### D-03
- Rank DINOv3 against baselines by macro F1 first, then artifact size and latency; baselines are fallback/comparison evidence, not the default target.

### D-04
- Generate all required assignment plots as reproducible artifacts.

### D-05
- Final model must be under 95 MB or have documented compression/export path.

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
- `.planning/ROADMAP.md` — phase goal and success criteria.
- `docs/plan.md` — approved library list and dependency rationale.
- `docs/Yazlab 2- Proje 3.md` — assignment constraints.
- `.agents/skills/andrej-karpathy-skills/skills/karpathy-guidelines/SKILL.md` — simplicity and verifiability guidance.

    </canonical_refs>

    <specifics>
    ## Specific Ideas

    - Requirement IDs for this phase: TRN-04, TRN-05, EVAL-01, EVAL-02, EVAL-03, EVAL-04, EVAL-05, EVAL-06, EVAL-07.
    - Execute sequentially, preserving outputs from prior phases.
    - Write evidence files that later phases and the final report can consume.

    </specifics>

    <deferred>
    ## Deferred Ideas

    - Capabilities assigned to later roadmap phases remain deferred and must not be implemented here.

    </deferred>

    ---
    *Phase: 04-model-comparison-and-selection*
    *Context gathered: 2026-05-06 via `$gsd-discuss-phase 4`*
