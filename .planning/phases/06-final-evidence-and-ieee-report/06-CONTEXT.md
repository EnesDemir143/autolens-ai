# Phase 6: Final Evidence and IEEE Report - Context

    **Gathered:** 2026-05-06
    **Status:** Ready for planning
    **Source:** `$gsd-discuss-phase 6` from user-approved six-phase roadmap

    <domain>
    ## Phase Boundary

    Final Evidence and IEEE Report delivers only the capabilities mapped to requirement IDs: RPT-01, RPT-02, RPT-03. Work outside these IDs is deferred to the appropriate roadmap phase.

    </domain>

    <decisions>
    ## Implementation Decisions

    ### D-01
- Generate the IEEE report only after implementation evidence exists.

### D-02
- Use `.agents/skills/latex-report-creator/SKILL.md` / awesome-ieee-report rules.

### D-03
- Report must include dataset sources, preprocessing, model rationale, metrics, plots, normalized confusion matrix, UI evidence, and final model size.

### D-04
- Report body must use academic prose and pseudocode, not raw source code dumps.

### D-05
- Final package must include LaTeX sources, BibTeX, build README, reproducibility README, and submission artifacts.

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
- `.agents/skills/latex-report-creator/SKILL.md` — final IEEE report generator.

    </canonical_refs>

    <specifics>
    ## Specific Ideas

    - Requirement IDs for this phase: RPT-01, RPT-02, RPT-03.
    - Execute sequentially, preserving outputs from prior phases.
    - Write evidence files that later phases and the final report can consume.

    </specifics>

    <deferred>
    ## Deferred Ideas

    - Capabilities assigned to later roadmap phases remain deferred and must not be implemented here.

    </deferred>

    ---
    *Phase: 06-final-evidence-and-ieee-report*
    *Context gathered: 2026-05-06 via `$gsd-discuss-phase 6`*
