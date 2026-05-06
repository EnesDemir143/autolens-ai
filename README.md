# AutoLens AI

AutoLens AI targets the Yazlab 2 Project III 8-class car body type classifier assignment using Python 3.12 and uv.

## Setup

```fish
uv sync
uv run python -c "import autolens_ai; print(autolens_ai.__version__)"
uv run pytest
uv run ruff check .
uv run mypy src
```

Phase 1 only establishes the reproducible project foundation. Dataset curation, training, model comparison, Gradio UI, and the final IEEE report are intentionally deferred to later GSD phases.


## Make Commands

```fish
make help
make check
make format
make graphify-update
```

`make check` runs the package import check, pytest, ruff lint, and mypy type check.

## Quality Gates

```fish
uv run pytest
uv run ruff check .
uv run ruff format .
uv run mypy src
```
