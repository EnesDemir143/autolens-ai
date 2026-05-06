.PHONY: help sync import test lint format typecheck check pre-commit graphify-update graphify-report clean

help:
	@printf '%s\n' \
		'AutoLens AI development commands:' \
		'  make sync            Install/sync the uv environment' \
		'  make import          Verify package import/version' \
		'  make test            Run pytest' \
		'  make lint            Run ruff lint checks' \
		'  make format          Format with ruff' \
		'  make typecheck       Run mypy on src' \
		'  make check           Run import, tests, lint, and typecheck' \
		'  make pre-commit      Run all pre-commit hooks' \
		'  make graphify-update Refresh the project knowledge graph' \
		'  make graphify-report Show the current graph report' \
		'  make clean           Remove local test/cache artifacts'

sync:
	uv sync

import:
	uv run python -c "import autolens_ai; print(autolens_ai.__version__)"

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy src

check: import test lint typecheck

pre-commit:
	uv run pre-commit run --all-files

graphify-update:
	node "$(HOME)/.codex/get-shit-done/bin/gsd-tools.cjs" graphify build .

graphify-report:
	cat .planning/graphs/GRAPH_REPORT.md

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache
