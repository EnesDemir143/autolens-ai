# Phase 1: Project Foundation - Context

**Gathered:** 2026-05-06
**Status:** Ready for planning
**Source:** `$gsd-discuss-phase 1` inline/auto decisions from user-approved roadmap

<domain>
## Phase Boundary

Phase 1 delivers only the reproducible project foundation: uv/Python 3.12 initialization, dependency declaration, minimal source/test/config structure, and quality gate commands. It does not implement dataset ingestion, model training, evaluation logic, Gradio UI, or report generation.

</domain>

<decisions>
## Implementation Decisions

### D-01 — Python Runtime and Package Manager
- Use Python 3.12 and uv as the canonical runtime/package workflow.
- The initialization command must be `uv init --python 3.12`.
- Future install commands must be fish-compatible and documented as `uv add ...` groups.

### D-02 — Dependency Groups
- Runtime ML dependencies are grouped by purpose: core vision/training, metrics/reporting, deployment/UI, MLOps/config, Hugging Face/transformers.
- Developer dependencies are installed with `uv add --dev ruff mypy pre-commit pytest`.
- Do not introduce new dependencies in Phase 1 beyond the project-approved stack unless documented in `docs/dependencies.md` with rationale.

### D-03 — Minimal Explainable Structure
- Create a small, explainable Python package skeleton only.
- Avoid speculative abstractions; Phase 1 may create placeholder modules but must not implement training/data/UI behavior.
- Structure must be understandable for student demo questions.

### D-04 — Quality Gates
- Provide commands for linting, formatting, type checking, and testing.
- Tests should initially validate project importability/config assumptions rather than ML behavior.
- Quality gates must be runnable with uv, e.g. `uv run pytest`, `uv run ruff check .`, `uv run mypy src`.

### D-05 — Documentation First
- Add README/setup docs that explain environment creation, dependency install groups, and next-phase expectations.
- Mention that the IEEE report is intentionally Phase 6 and will use the local LaTeX report creator skill.

### the agent's Discretion
- Exact package/module names may be chosen by the executor as long as they are simple and consistent.
- Exact ruff/mypy strictness may start moderate to avoid blocking early ML iteration, but commands must exist.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project requirements and roadmap
- `.planning/PROJECT.md` — project context, constraints, and stack decisions.
- `.planning/REQUIREMENTS.md` — ENV-01, ENV-02, ENV-03 requirements.
- `.planning/ROADMAP.md` — Phase 1 goal and success criteria.
- `docs/plan.md` — approved library list and dependency rationale.
- `docs/Yazlab 2- Proje 3.md` — assignment constraints that drive package/UI/report choices.

### Skills and coding guidance
- `.agents/skills/andrej-karpathy-skills/skills/karpathy-guidelines/SKILL.md` — simplicity, surgical changes, verifiable goals.
- `.agents/skills/latex-report-creator/SKILL.md` — final report skill; Phase 1 should only reference it in docs, not generate report.

</canonical_refs>

<specifics>
## Specific Ideas

- Use `autolens_ai` as the Python package name.
- Keep first tests lightweight: package import, class label constants/config loading if introduced.
- Document dependency commands in `docs/dependencies.md` so Phase 1 can pass even before all large ML packages are installed.

</specifics>

<deferred>
## Deferred Ideas

- Dataset downloading and curation — Phase 2.
- Training loop/model factories — Phase 3.
- DINOv3/LoRA evaluation — Phase 4.
- Gradio UI — Phase 5.
- IEEE LaTeX report — Phase 6.

</deferred>

---
*Phase: 01-project-foundation*
*Context gathered: 2026-05-06 via `$gsd-discuss-phase 1`*
