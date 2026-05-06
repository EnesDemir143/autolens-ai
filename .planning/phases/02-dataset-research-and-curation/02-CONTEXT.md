# Phase 2: Dataset Research and Curation - Context
**Branch:** `feat/dataset-research-and-curation`

    **Gathered:** 2026-05-06
    **Status:** Ready for planning
    **Source:** `$gsd-discuss-phase 2` from user-approved six-phase roadmap

    <domain>
    ## Phase Boundary

    Dataset Research and Curation delivers only the capabilities mapped to requirement IDs: DATA-01, DATA-02, DATA-03, DATA-04, DATA-05, DATA-06. Work outside these IDs is deferred to the appropriate roadmap phase.

    </domain>

    <decisions>
    ## Implementation Decisions

    ### D-01
- Use Kaggle + Hugging Face as primary source families and record exact IDs/URLs.

### D-02
- Normalize all labels into exactly 8 assignment classes.

### D-03
- Create source manifest, class mapping, counts, known gaps, license notes, and target-vs-actual counts before training. Target 36k–40k raw candidates and about 32k clean near-balanced images (~4k/class).

### D-04
- Prevent leakage with hash/dedup and train/validation split scripts.

### D-05
- Treat MICRO, STATION WAGON, OPEN WHEEL/F1, and clean PICK_UP as likely gap/risk classes requiring targeted collection and audit.

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

    - Requirement IDs for this phase: DATA-01, DATA-02, DATA-03, DATA-04, DATA-05, DATA-06. Dataset scale target: 36k–40k raw candidates -> ~32k clean images, about 4k/class.
    - Execute sequentially, preserving outputs from prior phases.
    - Write evidence files that later phases and the final report can consume.

    </specifics>

    <deferred>
    ## Deferred Ideas

    - Capabilities assigned to later roadmap phases remain deferred and must not be implemented here.

    </deferred>

    ---
    *Phase: 02-dataset-research-and-curation*
    *Context gathered: 2026-05-06 via `$gsd-discuss-phase 2`*
