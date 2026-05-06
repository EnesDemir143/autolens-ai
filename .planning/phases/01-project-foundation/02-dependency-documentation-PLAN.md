---
phase: 1
phase_name: Project Foundation
plan_id: 01-PLAN-02
objective: Document the approved uv dependency installation groups for the full ML/UI/reporting stack.
wave: 1
depends_on: []
requirements_addressed: [ENV-02]
files_modified:
  - docs/dependencies.md
  - pyproject.toml
autonomous: true
---

# Plan 02 — Dependency Documentation

<objective>
Create a concrete dependency plan that executors can apply with fish-compatible `uv add` commands.
</objective>

<must_haves>
- Covers D-02 (dependency groups) and D-05 (documentation first).
- `docs/dependencies.md` lists every approved dependency group from `docs/plan.md`.
- The document includes `transformers`, `huggingface_hub`, and `datasets` for Hugging Face/DINOv3 work.
- The document marks DINOv3 gated access as a later validation task, not a Phase 1 blocker.
</must_haves>

<threat_model>
Dependency installation can execute package build scripts and download external wheels. Mitigation: use official PyPI packages only, document dependencies before execution, and do not add unapproved packages without rationale.
</threat_model>

<tasks>

<task id="P1-02-T1" type="execute">
  <title>Create dependency install guide</title>
  <read_first>
    - `docs/plan.md`
    - `.planning/research/STACK.md`
    - `.planning/phases/01-project-foundation/01-RESEARCH.md`
  </read_first>
  <action>
    Create `docs/dependencies.md` with sections: Core Vision and Training, Metrics and Reporting, Deployment and UI, MLOps and Config, Hugging Face and Transformers, Developer Tools. Under each section include the exact fish-compatible `uv add` command from Phase 1 research.
  </action>
  <acceptance_criteria>
    - `test -f docs/dependencies.md` exits 0.
    - `grep 'uv add torch torchvision pytorch-lightning timm albumentations opencv-python pillow' docs/dependencies.md` prints a match.
    - `grep 'uv add huggingface_hub datasets transformers' docs/dependencies.md` prints a match.
    - `grep 'uv add --dev ruff mypy pre-commit pytest' docs/dependencies.md` prints a match.
  </acceptance_criteria>
</task>

<task id="P1-02-T2" type="execute">
  <title>Record package purpose and risk notes</title>
  <read_first>
    - `docs/dependencies.md`
    - `docs/Yazlab 2- Proje 3.md`
  </read_first>
  <action>
    In `docs/dependencies.md`, add a table with columns `Group`, `Packages`, `Purpose`, `Used starting phase`. Include rows for Phase 2 dataset tools, Phase 3 training tools, Phase 4 evaluation/model comparison tools, Phase 5 Gradio UI tools, and Phase 6 report evidence tools. Add a note: `DINOv3 Hugging Face access is gated and validated in Phase 4.`
  </action>
  <acceptance_criteria>
    - `grep 'Used starting phase' docs/dependencies.md` prints a match.
    - `grep 'DINOv3 Hugging Face access is gated and validated in Phase 4' docs/dependencies.md` prints a match.
    - `grep 'Gradio' docs/dependencies.md` prints a match.
  </acceptance_criteria>
</task>

<task id="P1-02-T3" type="execute">
  <title>Optionally install dependencies through uv</title>
  <read_first>
    - `docs/dependencies.md`
    - `pyproject.toml`
  </read_first>
  <action>
    If network and wheel resolution are available, run the documented `uv add` commands exactly as listed in `docs/dependencies.md`. If any command fails due to network or platform wheel availability, capture the failing command and package in `docs/dependencies.md` under `## Install Notes`.
  </action>
  <acceptance_criteria>
    - If install succeeds: `grep 'torch' pyproject.toml` prints a match and `grep 'pytest' pyproject.toml` prints a match.
    - If install fails: `grep '## Install Notes' docs/dependencies.md` prints a match and includes the failing `uv add` command.
  </acceptance_criteria>
</task>

</tasks>

<verification>
Run:

```fish
grep 'uv add huggingface_hub datasets transformers' docs/dependencies.md
grep 'uv add --dev ruff mypy pre-commit pytest' docs/dependencies.md
```
</verification>
