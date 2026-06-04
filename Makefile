.PHONY: help sync import test lint format typecheck check pre-commit graphify-update graphify-report clean
.PHONY: compute-stats train-baseline-0 train-baseline-1 train-baseline-2 train-all-baselines resume-training
.PHONY: train-all-wandb
.PHONY: train-dinov3 train-dinov3-safe-focal train-dinov3-safe-weighted dry-run-dinov3 dry-run-dinov3-safe-focal dry-run-dinov3-safe-weighted
.PHONY: checkpoint-to-safetensors export-model size-check
.PHONY: calibrate-model calibrate-temperature calibrate-vector calibrate-dirichlet calibrate-all calibrate-compare calibrate-sweep
.PHONY: calibrate-dinov3-weighted-temperature calibrate-dinov3-weighted-vector calibrate-dinov3-weighted-dirichlet calibrate-dinov3-weighted-all calibrate-dinov3-weighted-compare
.PHONY: calibrate-dinov3-focal-temperature calibrate-dinov3-focal-vector calibrate-dinov3-focal-dirichlet calibrate-dinov3-focal-all calibrate-dinov3-focal-compare
.PHONY: calibrate-dinov3-all calibrate-dinov3-compare
.PHONY: prepare-demo-artifact deploy-run-to-demo deploy-run-full
.PHONY: demo demo-smoke ui-smoke-check
.PHONY: frontend-install frontend-build demo-web demo-gradio
.PHONY: download-hf-model

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
		'' \
		'─── HuggingFace → Local ──────────────────────────────' \
		'  make download-hf-model [MODEL=dinov3-weighted]   Download model from HF Hub' \
		'' \
		'─── Run Demo ─────────────────────────────────────────' \
		'  make demo-gradio     Launch Gradio demo (port 7860)' \
		'  make demo-web        Launch FastAPI + React frontend (port 8000)' \
		'  make demo             Alias for make demo-gradio' \
		'' \
		'─── Quality Gates ────────────────────────────────────' \
		'' \
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
		'─── Training (Phase 3) ──────────────────────────────' \
		'  make train-baseline-0       Train all 3 models with Baseline 0 (no augmentation)' \
		'  make train-baseline-1       Train ResNet18 with Baseline 1 (light augmentation)' \
		'  make train-baseline-2       Train ResNet18 with Baseline 2 (weighted sampler)' \
		'  make train-all-baselines    Run all baseline experiments sequentially' \
		'  make resume-training CONFIG=path RUN_ID=20260507_234800' \
		'' \
		'  Add --wandb flag for W&B logging:' \
		'    make train-baseline-0 WANDB=--wandb' \
		'' \
		'─── DINOv3 Tuning ───────────────────────────────────' \
		'  make dry-run-dinov3-safe-focal WANDB=--wandb  # 1 epoch smoke test' \
		'  make train-dinov3-safe-focal WANDB=--wandb      # safe aug + focal + EMA' \
		'  make train-dinov3-safe-weighted WANDB=--wandb   # safe aug + weighted CE + EMA' \
		'' \
		'─── Export & Calibration (Phase 4) ──────────────────' \
		'  make checkpoint-to-safetensors' \
		'  make export-model' \
		'  make size-check' \
		'  make calibrate-dinov3-all       Run weighted + focal DINOv3 calibration' \
		'  make calibrate-dinov3-compare   Compare saved DINOv3 calibration outputs' \
		'  make prepare-demo-artifact' \
		'  make deploy-run-full            Full export pipeline' \
		'' \
		'─── Smoke Tests ─────────────────────────────────────' \
		'  make demo-smoke         Run UI/inference smoke test (non-interactive)' \

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

# Calibration comparison
CAL_EXP ?= efficientnet_b2_current
CAL_ONNX ?= artifacts/export/$(CAL_EXP)/model.onnx
CAL_META ?= artifacts/export/$(CAL_EXP)/metadata.json
CAL_BASE ?= artifacts/calibration
DINO_WEIGHTED_CONFIG ?= configs/experiments/dinov3_safe_weighted_aug.yaml
DINO_FOCAL_CONFIG ?= configs/experiments/dinov3_safe_focal_aug.yaml
DINO_CAL_OUTPUT_BASE ?= artifacts/export
DINO_CAL_BATCH_SIZE ?= 64
DINO_WEIGHTED_EXPORT_DIR ?= artifacts/export/dinov3_safe_weighted_latest
DINO_FOCAL_EXPORT_DIR ?= artifacts/export/dinov3_safe_focal_latest

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

# ── Calibration Methods ───────────────────────────────────────
# Three post-hoc calibration methods + grid search

calibrate-temperature:
	@echo "=== Temperature Scaling ==="
	uv run python scripts/calibrate_model.py \
		--onnx $(ONNX) \
		--metadata $(METADATA) \
		--method temperature \
		--output-base $(CAL_BASE)

calibrate-model: calibrate-temperature
	@echo "Alias complete: calibrate-model -> calibrate-temperature"

calibrate-vector:
	@echo "=== Vector Scaling (grid search over L2 λ) ==="
	uv run python scripts/calibrate_model.py \
		--onnx $(ONNX) \
		--metadata $(METADATA) \
		--method vector_scaling \
		--l2-lambda 0.001 \
		--output-base $(CAL_BASE)
	uv run python scripts/calibrate_model.py \
		--onnx $(ONNX) \
		--metadata $(METADATA) \
		--method vector_scaling \
		--l2-lambda 0.01 \
		--output-base $(CAL_BASE)
	uv run python scripts/calibrate_model.py \
		--onnx $(ONNX) \
		--metadata $(METADATA) \
		--method vector_scaling \
		--l2-lambda 0.1 \
		--output-base $(CAL_BASE)
	uv run python scripts/calibrate_model.py \
		--onnx $(ONNX) \
		--metadata $(METADATA) \
		--method vector_scaling \
		--l2-lambda 1.0 \
		--output-base $(CAL_BASE)

calibrate-dirichlet:
	@echo "=== Dirichlet Calibration (grid search over ODIR λ) ==="
	uv run python scripts/calibrate_model.py \
		--onnx $(ONNX) \
		--metadata $(METADATA) \
		--method dirichlet \
		--odir-lambda 0.001 \
		--output-base $(CAL_BASE)
	uv run python scripts/calibrate_model.py \
		--onnx $(ONNX) \
		--metadata $(METADATA) \
		--method dirichlet \
		--odir-lambda 0.01 \
		--output-base $(CAL_BASE)
	uv run python scripts/calibrate_model.py \
		--onnx $(ONNX) \
		--metadata $(METADATA) \
		--method dirichlet \
		--odir-lambda 0.1 \
		--output-base $(CAL_BASE)
	uv run python scripts/calibrate_model.py \
		--onnx $(ONNX) \
		--metadata $(METADATA) \
		--method dirichlet \
		--odir-lambda 1.0 \
		--output-base $(CAL_BASE)

# Run ALL calibration methods in one shot (recommended)
calibrate-all: calibrate-temperature calibrate-vector calibrate-dirichlet
	@echo "✅ All calibration methods complete. Results in $(CAL_BASE)/$(CAL_EXP)/"

# Compare calibration methods on internal test set
calibrate-compare:
	@echo "=== Calibration Comparison on Internal Test ==="
	uv run python scripts/evaluate_calibration.py \
		--experiment $(CAL_EXP)

# Full sweep: calibrate + compare (recommended entry point)
calibrate-sweep:
	@echo "=== FULL CALIBRATION SWEEP: all methods × all lambdas ==="
	uv run python scripts/calibrate_model.py \
		--onnx $(ONNX) \
		--metadata $(METADATA) \
		--method all \
		--grid-search \
		--output-base $(CAL_BASE)
	@echo ""
	@echo "=== Now comparing results ==="
	uv run python scripts/evaluate_calibration.py \
		--experiment $(CAL_EXP) \
		--onnx $(ONNX) \
		--metadata $(METADATA) \
		--internal-test artifacts/dataset/splits/internal_test.csv \
		--output-base $(CAL_BASE)

# DINOv3 calibration experiments.
# These targets read calibration grids from each DINOv3 config and save under:
#   artifacts/export/<experiment_name>/calibration/<method...>/
calibrate-dinov3-weighted-temperature:
	@echo "=== DINOv3 weighted: Temperature Scaling ==="
	uv run python scripts/run_calibration_temperature.py \
		--config $(DINO_WEIGHTED_CONFIG) \
		--output-base $(DINO_CAL_OUTPUT_BASE) \
		--batch-size $(DINO_CAL_BATCH_SIZE)

calibrate-dinov3-weighted-vector:
	@echo "=== DINOv3 weighted: Vector Scaling grid ==="
	uv run python scripts/run_calibration_vector.py \
		--config $(DINO_WEIGHTED_CONFIG) \
		--output-base $(DINO_CAL_OUTPUT_BASE) \
		--batch-size $(DINO_CAL_BATCH_SIZE)

calibrate-dinov3-weighted-dirichlet:
	@echo "=== DINOv3 weighted: Dirichlet Calibration grid ==="
	uv run python scripts/run_calibration_dirichlet.py \
		--config $(DINO_WEIGHTED_CONFIG) \
		--output-base $(DINO_CAL_OUTPUT_BASE) \
		--batch-size $(DINO_CAL_BATCH_SIZE)

calibrate-dinov3-weighted-all: calibrate-dinov3-weighted-temperature calibrate-dinov3-weighted-vector calibrate-dinov3-weighted-dirichlet
	@echo "✅ DINOv3 weighted calibration complete: $(DINO_WEIGHTED_EXPORT_DIR)/calibration/"

calibrate-dinov3-weighted-compare:
	@echo "=== DINOv3 weighted: compare saved calibration outputs ==="
	uv run python scripts/evaluate_calibration.py \
		--experiment dinov3_safe_weighted_aug \
		--results-dir $(DINO_WEIGHTED_EXPORT_DIR)/calibration

calibrate-dinov3-focal-temperature:
	@echo "=== DINOv3 focal: Temperature Scaling ==="
	uv run python scripts/run_calibration_temperature.py \
		--config $(DINO_FOCAL_CONFIG) \
		--output-base $(DINO_CAL_OUTPUT_BASE) \
		--batch-size $(DINO_CAL_BATCH_SIZE)

calibrate-dinov3-focal-vector:
	@echo "=== DINOv3 focal: Vector Scaling grid ==="
	uv run python scripts/run_calibration_vector.py \
		--config $(DINO_FOCAL_CONFIG) \
		--output-base $(DINO_CAL_OUTPUT_BASE) \
		--batch-size $(DINO_CAL_BATCH_SIZE)

calibrate-dinov3-focal-dirichlet:
	@echo "=== DINOv3 focal: Dirichlet Calibration grid ==="
	uv run python scripts/run_calibration_dirichlet.py \
		--config $(DINO_FOCAL_CONFIG) \
		--output-base $(DINO_CAL_OUTPUT_BASE) \
		--batch-size $(DINO_CAL_BATCH_SIZE)

calibrate-dinov3-focal-all: calibrate-dinov3-focal-temperature calibrate-dinov3-focal-vector calibrate-dinov3-focal-dirichlet
	@echo "✅ DINOv3 focal calibration complete: $(DINO_FOCAL_EXPORT_DIR)/calibration/"

calibrate-dinov3-focal-compare:
	@echo "=== DINOv3 focal: compare saved calibration outputs ==="
	uv run python scripts/evaluate_calibration.py \
		--experiment dinov3_safe_focal_aug \
		--results-dir $(DINO_FOCAL_EXPORT_DIR)/calibration

calibrate-dinov3-all: calibrate-dinov3-weighted-all calibrate-dinov3-focal-all
	@echo "✅ All DINOv3 calibration experiments complete."

calibrate-dinov3-compare: calibrate-dinov3-weighted-compare calibrate-dinov3-focal-compare
	@echo "✅ All DINOv3 calibration comparisons printed."

# ── Deploy pipeline ───────────────────────────────────────────
deploy-run-to-demo: checkpoint-to-safetensors export-model size-check calibrate-temperature prepare-demo-artifact
	@echo "Deployable demo artifact (with temperature scaling) prepared at $(ARTIFACT_CONFIG)"

deploy-run-full: checkpoint-to-safetensors export-model size-check calibrate-all calibrate-compare prepare-demo-artifact
	@echo "Full deployment pipeline complete — best calibration selected"

prepare-demo-artifact:
	@echo "Writing active demo artifact config..."
	uv run python scripts/prepare_demo_artifact.py \
		--metadata $(METADATA) \
		--onnx $(ONNX) \
		--calibration $(CALIBRATION) \
		--output $(ARTIFACT_CONFIG)

# Training targets
WANDB ?=
WANDB_PROJECT ?= autolens-ai

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

# DINOv3 ViT-S/16 — non-LoRA then LoRA (Phase 4 model comparison)
train-dinov3:
	@echo "[1/2] Training DINOv3 ViT-S/16 — non-LoRA baseline..."
	uv run python scripts/train.py \
		--config configs/experiments/baseline_0_dinov3_vits16.yaml \
		$(WANDB)
	@echo ""
	@echo "Waiting 30s for memory to clear before LoRA run..."
	sleep 30
	@echo "[2/2] Training DINOv3 ViT-S/16 — LoRA baseline..."
	uv run python scripts/train.py \
		--config configs/experiments/baseline_0_dinov3_vits16_lora.yaml \
		--lora \
		$(WANDB)
	@echo ""
	@echo "DINOv3 training complete (non-LoRA + LoRA)."

train-dinov3-safe-focal: compute-stats
	@echo "Training recommended DINOv3 run: safe augmentation + class-weighted focal loss + EMA..."
	uv run python scripts/train.py \
		--config configs/experiments/dinov3_safe_focal_aug.yaml \
		$(WANDB) \
		$(if $(WANDB),--wandb-project $(WANDB_PROJECT),)
	@echo ""
	@echo "Recommended DINOv3 safe-focal run complete."

dry-run-dinov3-safe-focal: compute-stats
	@echo "[DRY RUN] DINOv3 safe-focal — 1 epoch smoke test..."
	uv run python scripts/train.py \
		--config configs/experiments/dinov3_safe_focal_aug.yaml \
		--max-epochs 1 \
		$(WANDB) \
		$(if $(WANDB),--wandb-project $(WANDB_PROJECT),)
	@echo ""
	@echo "DINOv3 safe-focal dry run complete."

train-dinov3-safe-weighted: compute-stats
	@echo "Training DINOv3 run: safe augmentation + class-weighted loss + EMA..."
	uv run python scripts/train.py \
		--config configs/experiments/dinov3_safe_weighted_aug.yaml \
		--wandb \
		--wandb-project $(WANDB_PROJECT)
	@echo ""
	@echo "DINOv3 safe-weighted run complete."

dry-run-dinov3-safe-weighted: compute-stats
	@echo "[DRY RUN] DINOv3 safe-weighted — 1 epoch smoke test..."
	uv run python scripts/train.py \
		--config configs/experiments/dinov3_safe_weighted_aug.yaml \
		--max-epochs 1 \
		--wandb \
		--wandb-project $(WANDB_PROJECT)
	@echo ""
	@echo "DINOv3 safe-weighted dry run complete."

# DINOv3 dry-run — 1 epoch only, tests data pipeline / logger / model init
dry-run-dinov3:
	@echo "[DRY RUN 1/2] DINOv3 ViT-S/16 non-LoRA — 1 epoch smoke test..."
	uv run python scripts/train.py \
		--config configs/experiments/baseline_0_dinov3_vits16.yaml \
		--max-epochs 1
	@echo ""
	@echo "Waiting 15s..."
	sleep 15
	@echo "[DRY RUN 2/2] DINOv3 ViT-S/16 LoRA — 1 epoch smoke test..."
	uv run python scripts/train.py \
		--config configs/experiments/baseline_0_dinov3_vits16_lora.yaml \
		--lora \
		--max-epochs 1
	@echo ""
	@echo "Dry-run complete. Check logs for errors before overnight run."

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

# Phase 05-05: FastAPI + Vite frontend
frontend-install:
	@echo "Installing frontend npm dependencies..."
	cd frontend && npm install

frontend-build:
	@echo "Building production frontend..."
	cd frontend && npm run build

demo-web: frontend-build
	@echo "Launching FastAPI + Vite demo on http://localhost:8080 ..."
	uv run python -m uvicorn api:app --host 0.0.0.0 --port 8080

demo-gradio:
	@echo "Launching Gradio demo on http://localhost:7860 ..."
	uv run python app.py

# ── Download from HuggingFace Hub ────────────────────────
MODEL ?= dinov3-weighted

download-hf-model:
	@echo "Downloading model '$(MODEL)' from HuggingFace Hub ..."
	uv run python scripts/download_from_hf.py --model $(MODEL)
