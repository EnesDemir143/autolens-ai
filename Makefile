.PHONY: help sync import test lint format typecheck check pre-commit graphify-update graphify-report clean
.PHONY: compute-stats train-baseline-0 train-baseline-1 train-baseline-2 train-all-baselines resume-training
.PHONY: train-all-wandb
.PHONY: checkpoint-to-safetensors export-model size-check calibrate-model prepare-demo-artifact deploy-run-to-demo
.PHONY: demo demo-smoke ui-smoke-check

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
		'Phase 4 demo artifact sequence:' \
		'  make checkpoint-to-safetensors [CHECKPOINT=path.ckpt] [EXPORT_DIR=artifacts/export/efficientnet_b2_current]' \
		'  make export-model [SAFETENSORS=path] [METADATA=path] [ONNX=path]' \
		'  make size-check [SAFETENSORS=path] [ONNX=path]' \
		'  make calibrate-model [ONNX=path] [METADATA=path]  # validation split only' \
		'  make prepare-demo-artifact [ARTIFACT_CONFIG=artifacts/demo/active_model.json]' \
		'  make deploy-run-to-demo [CHECKPOINT=path.ckpt]  # convert -> export -> size -> calibrate -> pointer' \
		'' \
		'Phase 5 Gradio demo commands:' \
		'  make demo               Launch Gradio demo server' \
		'  make demo-smoke         Run UI/inference smoke test (non-interactive)' \
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
		'  make train-baseline-0 WANDB=--wandb' \
		'' \
		'Full W&B run (all 5 models sequentially):' \
		'  make train-all-wandb'

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

# Phase 4 current-candidate export/calibration targets.
EXPORT_DIR ?= artifacts/export/efficientnet_b2_current
SAFETENSORS ?= $(EXPORT_DIR)/model.safetensors
METADATA ?= $(EXPORT_DIR)/metadata.json
ONNX ?= $(EXPORT_DIR)/model.onnx
CALIBRATION ?= $(EXPORT_DIR)/calibration.json
ARTIFACT_CONFIG ?= artifacts/demo/active_model.json
EXPORT_CONFIG ?= configs/experiments/baseline_0_efficientnet_b2.yaml

checkpoint-to-safetensors:
	@echo "Converting Lightning checkpoint to safetensors + metadata..."
	uv run python scripts/checkpoint_to_safetensors.py \
		$(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),) \
		--config $(EXPORT_CONFIG) \
		--output-dir $(EXPORT_DIR)

export-model:
	@echo "Exporting safetensors artifact to ONNX..."
	uv run python scripts/export_model.py \
		--safetensors $(SAFETENSORS) \
		--metadata $(METADATA) \
		--output $(ONNX)

size-check:
	@echo "Checking deploy artifact sizes against 95 MB..."
	uv run python scripts/check_artifact_size.py \
		$(SAFETENSORS) \
		$(ONNX) \
		--json-output $(EXPORT_DIR)/size_check.json

calibrate-model:
	@echo "Fitting validation-only temperature scaling..."
	uv run python scripts/calibrate_model.py \
		--onnx $(ONNX) \
		--metadata $(METADATA) \
		--output $(CALIBRATION)

prepare-demo-artifact:
	@echo "Writing active demo artifact config..."
	uv run python scripts/prepare_demo_artifact.py \
		--metadata $(METADATA) \
		--onnx $(ONNX) \
		--calibration $(CALIBRATION) \
		--output $(ARTIFACT_CONFIG)

deploy-run-to-demo: checkpoint-to-safetensors export-model size-check calibrate-model prepare-demo-artifact
	@echo "Deployable demo artifact prepared at $(ARTIFACT_CONFIG)"

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

# Full W&B comparison run: 5 models sequentially, 30s sleep between each
# Early stopping (patience=10) is active in all configs.
# Even if you Ctrl+C mid-run, completed runs are already synced to W&B.
train-all-wandb: compute-stats
	@echo "=== [1/5] ResNet18 ==="
	uv run python scripts/train.py --config configs/experiments/baseline_0_resnet18.yaml --wandb
	@echo "Sleeping 30s..."; sleep 30
	@echo "=== [2/5] MobileNetV4 ==="
	uv run python scripts/train.py --config configs/experiments/baseline_0_mobilenetv4.yaml --wandb
	@echo "Sleeping 30s..."; sleep 30
	@echo "=== [3/5] EfficientNet-B2 ==="
	uv run python scripts/train.py --config configs/experiments/baseline_0_efficientnet_b2.yaml --wandb
	@echo "Sleeping 30s..."; sleep 30
	@echo "=== [4/5] DINOv3 ViT-S/16 + LoRA ==="
	uv run python scripts/train.py --config configs/experiments/baseline_0_dinov3_vits16_lora.yaml --wandb --lora
	@echo "Sleeping 30s..."; sleep 30
	@echo "=== [5/5] DINOv3 ViT-S/16 (no LoRA) ==="
	uv run python scripts/train.py --config configs/experiments/baseline_0_dinov3_vits16.yaml --wandb
	@echo ""
	@echo "All 5 models complete. Results synced to W&B project: autolens-ai"

# Phase 5: Gradio demo
demo:
	@echo "Launching Gradio demo on http://localhost:7860 ..."
	uv run python app.py

demo-smoke:
	@echo "Running UI/inference smoke test..."
	uv run python scripts/smoke_test_demo.py
