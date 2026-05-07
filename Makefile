.PHONY: help sync import test lint format typecheck check pre-commit graphify-update graphify-report clean
.PHONY: train-baseline-0 train-baseline-1 train-baseline-2 train-all-baselines

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
		'  make clean           Remove local test/cache artifacts' \
		'' \
		'Training commands (Phase 3):' \
		'  make train-baseline-0       Train all 3 models with Baseline 0 (no augmentation)' \
		'  make train-baseline-1       Train ResNet18 with Baseline 1 (light augmentation)' \
		'  make train-baseline-2       Train ResNet18 with Baseline 2 (weighted sampler)' \
		'  make train-all-baselines    Run all baseline experiments sequentially' \
		'' \
		'Add --wandb flag for W&B logging:' \
		'  make train-baseline-0 WANDB=--wandb'

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

# Training targets
WANDB ?=

train-baseline-0:
	@echo "Training Baseline 0: ResNet18 (no augmentation)..."
	uv run python scripts/train.py --config configs/experiments/baseline_0_resnet18.yaml $(WANDB)
	@echo ""
	@echo "Training Baseline 0: MobileNetV4 Conv Medium..."
	uv run python scripts/train.py --config configs/experiments/baseline_0_mobilenetv4.yaml $(WANDB)
	@echo ""
	@echo "Training Baseline 0: EfficientNet-B2..."
	uv run python scripts/train.py --config configs/experiments/baseline_0_efficientnet_b2.yaml $(WANDB)

train-baseline-1:
	@echo "Training Baseline 1: ResNet18 (light augmentation)..."
	uv run python scripts/train.py --config configs/experiments/baseline_1_resnet18_augmented.yaml $(WANDB)

train-baseline-2:
	@echo "Training Baseline 2: ResNet18 (weighted sampler)..."
	uv run python scripts/train.py --config configs/experiments/baseline_2_resnet18_weighted_sampler.yaml $(WANDB)

train-all-baselines: train-baseline-0 train-baseline-1 train-baseline-2
	@echo ""
	@echo "All baseline experiments complete!"
