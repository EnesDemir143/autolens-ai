---
phase: 1
phase_name: Project Foundation
plan_id: 01-PLAN-01
objective: Initialize the uv/Python 3.12 project skeleton and minimal importable package.
wave: 1
depends_on: []
requirements_addressed: [ENV-01]
files_modified:
  - pyproject.toml
  - README.md
  - src/autolens_ai/__init__.py
  - tests/test_package_import.py
autonomous: true
---

# Plan 01 — Scaffold uv Project

<objective>
Create the minimal project skeleton required for a reproducible Python 3.12 package named `autolens_ai`.
</objective>

<must_haves>
- Covers D-01 (Python 3.12 + uv) and D-03 (minimal explainable structure).
- `pyproject.toml` exists and declares Python 3.12 compatibility.
- `src/autolens_ai/__init__.py` exists and defines `__version__`.
- `tests/test_package_import.py` imports `autolens_ai`.
- README contains a Phase 1 setup section.
</must_haves>

<threat_model>
No external service or credential flow is introduced in this plan. Main risk is accidental inclusion of local files or credentials in generated scaffolding. Mitigation: do not create `.env` with secrets; if an example is needed, use `.env.example` with placeholder values only.
</threat_model>

<tasks>

<task id="P1-01-T1" type="execute">
  <title>Initialize uv project metadata</title>
  <read_first>
    - `.planning/PROJECT.md`
    - `.planning/phases/01-project-foundation/01-CONTEXT.md`
    - `docs/plan.md`
  </read_first>
  <action>
    Run `uv init --python 3.12` if `pyproject.toml` does not already exist. Ensure `pyproject.toml` project name is `autolens-ai` or `autolens_ai`, and ensure Python requirement is compatible with Python 3.12.
  </action>
  <acceptance_criteria>
    - `test -f pyproject.toml` exits 0.
    - `grep -E "requires-python|python" pyproject.toml` prints a Python version declaration including `3.12` or `>=3.12`.
  </acceptance_criteria>
</task>

<task id="P1-01-T2" type="execute">
  <title>Create importable package skeleton</title>
  <read_first>
    - `pyproject.toml`
  </read_first>
  <action>
    Create `src/autolens_ai/__init__.py` with a simple `__version__ = "0.1.0"`. Do not implement dataset, training, model, evaluation, or UI modules in this plan.
  </action>
  <acceptance_criteria>
    - `test -f src/autolens_ai/__init__.py` exits 0.
    - `grep '__version__' src/autolens_ai/__init__.py` prints a version line.
    - `uv run python -c "import autolens_ai; print(autolens_ai.__version__)"` exits 0.
  </acceptance_criteria>
</task>

<task id="P1-01-T3" type="execute">
  <title>Add package import test</title>
  <read_first>
    - `src/autolens_ai/__init__.py`
  </read_first>
  <action>
    Create `tests/test_package_import.py` containing a test that imports `autolens_ai` and asserts `autolens_ai.__version__` is a non-empty string.
  </action>
  <acceptance_criteria>
    - `test -f tests/test_package_import.py` exits 0.
    - `grep 'import autolens_ai' tests/test_package_import.py` prints a match.
    - `grep '__version__' tests/test_package_import.py` prints a match.
  </acceptance_criteria>
</task>

<task id="P1-01-T4" type="execute">
  <title>Document initial setup in README</title>
  <read_first>
    - `.planning/ROADMAP.md`
    - `docs/Yazlab 2- Proje 3.md`
  </read_first>
  <action>
    Create or update `README.md` with a `## Setup` section containing `uv sync`, `uv run pytest`, `uv run ruff check .`, and `uv run mypy src`. Also include one sentence that this project targets the Yazlab 2 Project III 8-class car body type classifier.
  </action>
  <acceptance_criteria>
    - `grep '## Setup' README.md` prints a match.
    - `grep 'uv sync' README.md` prints a match.
    - `grep 'uv run pytest' README.md` prints a match.
    - `grep '8-class car body type' README.md` or `grep '8 sınıf' README.md` prints a match.
  </acceptance_criteria>
</task>

</tasks>

<verification>
Run:

```fish
uv run python -c "import autolens_ai; print(autolens_ai.__version__)"
uv run pytest
```
</verification>
