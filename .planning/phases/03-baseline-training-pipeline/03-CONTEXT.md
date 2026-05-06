# Phase 3: Baseline Training Pipeline - Context
**Branch:** `feat/baseline-training-pipeline`

    **Gathered:** 2026-05-06
    **Status:** Ready for planning
    **Source:** `$gsd-discuss-phase 3` from user-approved six-phase roadmap

    <domain>
    ## Phase Boundary

    Baseline Training Pipeline delivers only the capabilities mapped to requirement IDs: TRN-01, TRN-02, TRN-03, TRN-06, TRN-07. Work outside these IDs is deferred to the appropriate roadmap phase.

    </domain>

    <decisions>
    ## Implementation Decisions

    ### D-01
- Use PyTorch Lightning for training loops, checkpointing, early stopping, and device selection.

### D-02
- Use timm for MobileNetV4 Conv Medium and EfficientNet-B2 when available; document fallback names if timm lacks the exact model.

### D-03
- Use a ResNet baseline from torchvision or timm as the classic control model.

### D-04
- Use shared preprocessing/augmentation configuration to keep train/validation behavior explainable.

### D-05
- Do not add DINOv3 in Phase 3; DINOv3 is the intended main model and belongs to Phase 4. Phase 3 CNNs are baselines/comparison only.

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

    - Requirement IDs for this phase: TRN-01, TRN-02, TRN-03, TRN-06, TRN-07. TRN-01..TRN-03 are baseline/comparison requirements, not the main model path.
    - Execute sequentially, preserving outputs from prior phases.
    - Write evidence files that later phases and the final report can consume.

    </specifics>

    <deferred>
    ## Deferred Ideas

    - Capabilities assigned to later roadmap phases remain deferred and must not be implemented here.

    </deferred>

    ---
    *Phase: 03-baseline-training-pipeline*
    *Context gathered: 2026-05-06 via `$gsd-discuss-phase 3`*
