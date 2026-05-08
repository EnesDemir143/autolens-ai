.PHONY: help sync import test lint format typecheck check pre-commit graphify-update graphify-report clean
.PHONY: compute-stats train-baseline-0 train-baseline-1 train-baseline-2 train-all-baselines resume-training

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
		'Dataset preparation:' \
		'  make create-splits   Create stratified train/val/test splits (seed=42, 80/10/10)' \
		'  make compute-stats   Compute mean/std from train split for normalization' \
		'' \
		'Custom split ratios:' \
		'  make create-splits TRAIN_RATIO=0.7 VAL_RATIO=0.15 TEST_RATIO=0.15 SEED=123' \
		'' \
		'Model analysis:' \
		'  make check-model-size   Check if model checkpoints are under 95 MB limit' \
		'' \
		'Training commands (Phase 3):' \
		'  make train-resnet-mobilenet    Train ResNet18 then MobileNetV4 sequentially (with 30s pause)' \
		'  make train-baseline-0       Train all 3 models with Baseline 0 (no augmentation)' \
		'  make train-baseline-1       Train ResNet18 with Baseline 1 (light augmentation)' \
		'  make train-baseline-2       Train ResNet18 with Baseline 2 (weighted sampler)' \
		'  make train-all-baselines    Run all baseline experiments sequentially' \
		'' \
		'Resume training:' \
		'  make resume-training CONFIG=path/to/config.yaml RUN_ID=20260507_234800' \
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

# Dataset preparation
TRAIN_RATIO ?= 0.8
VAL_RATIO ?= 0.1
TEST_RATIO ?= 0.1
SEED ?= 42

create-splits:
	@echo "Creating stratified train/val/test splits with seed..."
	uv run python scripts/create_splits.py \
		--train $(TRAIN_RATIO) \
		--val $(VAL_RATIO) \
		--test $(TEST_RATIO) \
		--seed $(SEED)

compute-stats:
	@if [ -f artifacts/dataset/stats.json ]; then \
		echo "✓ Dataset stats already exist (artifacts/dataset/stats.json)"; \
		echo "  Delete the file to recompute"; \
	else \
		echo "Computing mean and std from training split..."; \
		uv run python scripts/compute_dataset_stats.py --output artifacts/dataset/stats.json; \
		echo "Stats saved to artifacts/dataset/stats.json"; \
	fi

# Model analysis
check-model-size:
	@echo "Checking model checkpoint sizes..."
	uv run python scripts/check_model_size.py checkpoints/

# Training targets
WANDB ?=

train-resnet-mobilenet: compute-stats
	@echo "Training Baseline 0: ResNet18..."
	uv run python scripts/train.py --config configs/experiments/baseline_0_resnet18.yaml $(WANDB)
	@echo "Waiting 30s for memory to clear..."
	sleep 30
	@echo "Training Baseline 0: MobileNetV4..."
	uv run python scripts/train.py --config configs/experiments/baseline_0_mobilenetv4.yaml $(WANDB)

train-baseline-0: compute-stats
	@echo "Training Baseline 0: ResNet18 (no augmentation)..."
	uv run python scripts/train.py --config configs/experiments/baseline_0_resnet18.yaml $(WANDB)
	@echo ""
	@echo "Training Baseline 0: MobileNetV4 Conv Medium..."
	uv run python scripts/train.py --config configs/experiments/baseline_0_mobilenetv4.yaml $(WANDB)
	@echo ""
	@echo "Training Baseline 0: EfficientNet-B2..."
	uv run python scripts/train.py --config configs/experiments/baseline_0_efficientnet_b2.yaml $(WANDB)

train-baseline-1: compute-stats
	@echo "Training Baseline 1: ResNet18 (light augmentation)..."
	uv run python scripts/train.py --config configs/experiments/baseline_1_resnet18_augmented.yaml $(WANDB)

train-baseline-2: compute-stats
	@echo "Training Baseline 2: ResNet18 (weighted sampler)..."
	uv run python scripts/train.py --config configs/experiments/baseline_2_resnet18_weighted_sampler.yaml $(WANDB)

train-all-baselines: train-baseline-0 train-baseline-1 train-baseline-2
	@echo ""
	@echo "All baseline experiments complete!"

# Resume training from checkpoint
resume-training:
	@if [ -z "$(CONFIG)" ] || [ -z "$(RUN_ID)" ]; then \
		echo "ERROR: Both CONFIG and RUN_ID are required"; \
		echo "Usage: make resume-training CONFIG=configs/experiments/baseline_0_resnet18.yaml RUN_ID=20260507_234800"; \
		exit 1; \
	fi
	@echo "Resuming training from run: $(RUN_ID)"
	uv run python scripts/train.py --config $(CONFIG) --run-id $(RUN_ID) --resume $(WANDB)
