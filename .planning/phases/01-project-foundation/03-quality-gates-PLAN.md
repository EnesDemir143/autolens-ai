---
phase: 1
phase_name: Project Foundation
plan_id: 01-PLAN-03
objective: Configure practical lint, type, test, and pre-commit gates for the project foundation.
wave: 2
depends_on: [01-PLAN-01, 01-PLAN-02]
requirements_addressed: [ENV-03]
files_modified:
  - pyproject.toml
  - .pre-commit-config.yaml
  - README.md
autonomous: true
---

# Plan 03 — Quality Gates

<objective>
Add runnable development quality commands without over-constraining early ML experimentation.
</objective>

<must_haves>
- Covers D-04 (quality gates).
- `uv run pytest` succeeds.
- `uv run ruff check .` succeeds or has concrete remediation tasks before completion.
- `uv run mypy src` succeeds or documents an explicit temporary exception.
- `.pre-commit-config.yaml` exists with ruff hooks if `pre-commit` is installed.
</must_haves>

<threat_model>
Quality tooling should not read credentials or mutate external systems. Mitigation: commands are local-only; pre-commit hooks are limited to ruff and lightweight checks.
</threat_model>

<tasks>

<task id="P1-03-T1" type="execute">
  <title>Configure ruff, pytest, and mypy in pyproject</title>
  <read_first>
    - `pyproject.toml`
    - `.planning/phases/01-project-foundation/01-CONTEXT.md`
  </read_first>
  <action>
    Add practical tool configuration to `pyproject.toml`: ruff line length `100`, ruff target version `py312`, pytest testpaths `tests`, and mypy Python version `3.12` for `src`. Do not enable strict mypy settings that block early ML imports unless they already pass.
  </action>
  <acceptance_criteria>
    - `grep 'line-length = 100' pyproject.toml` prints a match.
    - `grep 'py312' pyproject.toml` prints a match.
    - `grep 'testpaths' pyproject.toml` prints a match.
    - `grep 'python_version = "3.12"' pyproject.toml` or `grep "python_version = '3.12'" pyproject.toml` prints a match.
  </acceptance_criteria>
</task>

<task id="P1-03-T2" type="execute">
  <title>Add pre-commit configuration</title>
  <read_first>
    - `docs/dependencies.md`
    - `pyproject.toml`
  </read_first>
  <action>
    Create `.pre-commit-config.yaml` with ruff check and ruff format hooks from `https://github.com/astral-sh/ruff-pre-commit`. Pin the hook revision to a concrete version matching the installed or planned ruff version where practical.
  </action>
  <acceptance_criteria>
    - `test -f .pre-commit-config.yaml` exits 0.
    - `grep 'ruff-pre-commit' .pre-commit-config.yaml` prints a match.
    - `grep 'ruff-check' .pre-commit-config.yaml` or `grep 'ruff' .pre-commit-config.yaml` prints a match.
    - `grep 'ruff-format' .pre-commit-config.yaml` or `grep 'format' .pre-commit-config.yaml` prints a match.
  </acceptance_criteria>
</task>

<task id="P1-03-T3" type="execute">
  <title>Document quality gate commands</title>
  <read_first>
    - `README.md`
    - `.planning/phases/01-project-foundation/01-VALIDATION.md`
  </read_first>
  <action>
    Add a `## Quality Gates` section to `README.md` containing exactly these commands: `uv run pytest`, `uv run ruff check .`, `uv run ruff format .`, and `uv run mypy src`.
  </action>
  <acceptance_criteria>
    - `grep '## Quality Gates' README.md` prints a match.
    - `grep 'uv run pytest' README.md` prints a match.
    - `grep 'uv run ruff check .' README.md` prints a match.
    - `grep 'uv run ruff format .' README.md` prints a match.
    - `grep 'uv run mypy src' README.md` prints a match.
  </acceptance_criteria>
</task>

<task id="P1-03-T4" type="execute">
  <title>Run foundation verification</title>
  <read_first>
    - `src/autolens_ai/__init__.py`
    - `tests/test_package_import.py`
    - `pyproject.toml`
  </read_first>
  <action>
    Run the Phase 1 validation commands. If a command fails, fix the foundation issue if local and non-destructive; otherwise document the blocker and exact output in a Phase 1 execution summary.
  </action>
  <acceptance_criteria>
    - `uv run pytest` exits 0.
    - `uv run ruff check .` exits 0.
    - `uv run mypy src` exits 0 or an explicit documented temporary exception exists with the failing output.
  </acceptance_criteria>
</task>

</tasks>

<verification>
Run:

```fish
uv run pytest
uv run ruff check .
uv run mypy src
```
</verification>
