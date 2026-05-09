# Roadmap: AutoLens AI — Yazlab 2 Proje 3

**Created:** 2026-05-06
**Granularity:** Standard
**Execution:** Sequential
**Target:** 7 phases

## Overview

| # | Phase | Branch | Goal | Requirements | Success Criteria |
|---|-------|--------|------|--------------|------------------|
| 1 | Project Foundation | `main` | Create reproducible uv/Python 3.12 project skeleton and quality gates | ENV-01, ENV-02, ENV-03 | 4 |
| 2 | Dataset Research and Curation | `feat/dataset-curation-eda` | Build documented near-balanced 8-class dataset pipeline from public sources | DATA-01..DATA-06 | 5 |
| 3 | Baseline Training Pipeline | `feat/baseline-training-pipeline` | Train explainable CNN baseline/comparison models with Lightning and MPS/CPU support | TRN-01, TRN-02, TRN-03, TRN-06, TRN-07 | 5 |
| 4 | Export, Calibration, Overnight Model Selection | `feat/model-comparison-and-selection` | First make the current best EfficientNet-B2 checkpoint deployable, then run overnight DINOv3/EfficientNet variants and select by F1/size/speed | TRN-04, TRN-05, EVAL-01..EVAL-07, OPT-01, OPT-02 | 7 |
| 5 | Gradio Demo Interface | `feat/gradio-demo-interface` | Deliver a model-agnostic, presentation-ready web UI that initially uses the current calibrated/exported checkpoint and can be repointed to the final winner | UI-01..UI-07 | 5 |
| 6 | Hugging Face Spaces Demo Deploy and Model Publishing | `feat/hugging-face-spaces-deploy` | Package/publish the Gradio + ONNX Runtime demo on HF Spaces and publish completed model artifacts to HF model repos without redistributing raw datasets | DEPLOY-01..DEPLOY-05, PUBLISH-01..PUBLISH-03 | 5 |
| 7 | Final Evidence and IEEE Report | `feat/final-evidence-and-ieee-report` | Package plots, metrics, final model, deploy evidence/link, and LaTeX IEEE report | RPT-01..RPT-03 | 5 |

## Phase Details

### Phase 1: Project Foundation

**Status:** Complete — verified 2026-05-06

**Branch:** `main`

**Goal:** Create a clean, reproducible Python 3.12 project using uv and the agreed ML stack.

**Requirements:** ENV-01, ENV-02, ENV-03

**Success criteria:**
1. `uv init --python 3.12` creates the project environment.
2. Planned dependencies are documented and installable with `uv add` commands.
3. Dev tools are configured for lint/type/test execution.
4. Repository contains a minimal source/test structure that future phases can extend without speculative abstractions.

**Implementation notes:**
- Keep config minimal: pyproject, src package, tests, configs.
- Do not add optional ONNX/Optuna workflows until baseline training works, but include dependencies as planned.

### Phase 2: Dataset Research and Curation

**Branch:** `feat/dataset-curation-eda`

**Goal:** Convert public Kaggle/Hugging Face/metadata-backed sources into a documented near-balanced 8-class dataset: 36k–40k raw candidates, approximately 32k clean images, about 4k/class.

**Requirements:** DATA-01, DATA-02, DATA-03, DATA-04, DATA-05, DATA-06

**Success criteria:**
1. Candidate sources are downloaded or documented with exact IDs/URLs.
2. Raw labels are mapped into the 8 required assignment classes.
3. A manifest records source, class mapping, counts, known gaps, license notes, and target-vs-actual counts.
4. Per-dataset CSV manifests and EDA summaries are produced before merging; merged EDA reports class distributions, source-by-class coverage, image quality statistics, and source/domain bias.
5. Missing-data, missing-file, outlier, duplicate, corrupt-image, and ambiguous-label review lists are created and used to decide accepted/filtered/excluded/deferred data.
6. Train/validation/internal-test split is created without final-test leakage after the EDA decision gate.
7. Class imbalance report checks the approximate 4k/class target and identifies weak classes such as MICRO, STATION WAGON, OPEN WHEEL/F1, and PICK_UP.

**Implementation notes:**
- Start with assignment-referenced Kaggle datasets.
- Use metadata/CSV-backed sources such as Car Model Variants, CompCars, Stanford/VMMR/BoxCars lookup, and F1 datasets to reach 30–40k scale.
- Do not blindly map ambiguous labels: `City Car`, generic `Truck`, `Crossover`, `MPV/Minibus`, and `Coupe` require filtering or exclusion.
- Use EDA evidence to decide the usable datasets/classes first; detailed preprocessing/augmentation choices are finalized in Phase 3 using the EDA findings.
- Keep manual review hooks for mislabeled/ambiguous images.

### Phase 3: Baseline Training Pipeline

**Branch:** `feat/baseline-training-pipeline`

**Goal:** Build the reusable training/evaluation path and train MobileNetV4 Conv Medium, EfficientNet-B2, and ResNet18 strictly as baseline/comparison models. These three Phase 3 baselines feed the final four-model-family comparison once DINOv3 is added in Phase 4.

**Requirements:** TRN-01, TRN-02, TRN-03, TRN-06, TRN-07

**Success criteria:**
1. Lightning training loop runs on MPS or CPU fallback.
2. Augmentation and normalization are consistent between training and validation.
3. Early stopping and checkpointing work.
4. Three CNN-family baselines produce checkpoints and validation metrics.
5. Code remains small enough for line-by-line demo explanation.

**Implementation notes:**
- Use timm for MobileNetV4/EfficientNet where possible.
- Use torchvision/timm for ResNet18 baseline to keep the classic control model safely below the 95 MB artifact limit.
- If MobileNetV4 Conv Medium name is unavailable in timm, phase must document the closest supported timm model and rationale before substitution.

### Phase 4: Export, Calibration, Overnight Model Selection

**Branch:** `feat/model-comparison-and-selection`

**Goal:** De-risk the project by first making the current best EfficientNet-B2 checkpoint deployable, calibrated, and size-checked, while the slower DINOv3 and enhanced EfficientNet experiments run overnight. After overnight runs finish, select the winner and repeat export/calibration only for that final model.

**Requirements:** TRN-04, TRN-05, EVAL-01, EVAL-02, EVAL-03, EVAL-04, EVAL-05, EVAL-06, EVAL-07, OPT-01, OPT-02

**Success criteria:**
1. Current best EfficientNet-B2 Lightning checkpoint is converted to `model.safetensors` plus metadata, then exported to ONNX, simplified when possible, and checked against the 95 MB artifact limit.
2. Temperature scaling calibration is fit on validation predictions only; internal test and instructor final test images remain untouched.
3. Calibration evidence records before/after confidence behavior with NLL/ECE or equivalent confidence diagnostics where available.
4. DINOv3 ViT-S non-LoRA, DINOv3 LoRA, and EfficientNet-B2 augmentation+EMA+focal-loss runs are queued for overnight execution, not blockers for the first working demo.
5. Every completed model has Accuracy, balanced accuracy, MCC, Precision, Recall, F1, macro/weighted averages, per-class metrics, required plots, size, and latency evidence.
6. Final model candidate is under 95 MB or has a clear compression/export path.
7. Final selection rationale is written in report-ready language; if a later overnight model wins, only export/calibration and the model pointer are repeated.
8. Makefile targets document and execute the required order for a chosen run/checkpoint: ckpt → safetensors → ONNX export → size check → calibration → Gradio artifact pointer/update → smoke test.

**Implementation notes:**
- Immediate path: Lightning checkpoint → safetensors+metadata → ONNX export → size check → temperature scaling → Gradio adapter readiness using the current EfficientNet-B2 best checkpoint.
- Overnight path: DINOv3 ViT-S non-LoRA, DINOv3 LoRA, and EfficientNet-B2 augmentation+EMA+focal-loss.
- F1-score is the primary ranking metric; size and latency decide deployability.
- Keep the Gradio code model-agnostic so replacing the artifact/config is enough after final selection.
- Add Makefile orchestration targets so the student can say which run/checkpoint to use and execute the deploy sequence without remembering individual script commands.

### Phase 5: Gradio Demo Interface

**Branch:** `feat/gradio-demo-interface`

**Goal:** Provide a clean, modern web UI that satisfies every assignment interface requirement using a swappable calibrated model artifact; start with the current EfficientNet-B2 export, then repoint to the final selected model after overnight comparison.

**Requirements:** UI-01, UI-02, UI-03, UI-04, UI-05, UI-06, UI-07

**Success criteria:**
1. User can upload or drag-and-drop a vehicle image.
2. UI shows uploaded image preview.
3. UI shows predicted class clearly and prominently.
4. UI shows confidence plus probability distribution for all 8 classes.
5. UI is visually clean, modern, and demo-ready with acceptable live prediction latency.
6. UI shows a professional loading/progress state during inference, especially for slower DINOv3 CPU predictions, so the page never appears frozen.
7. Makefile has a documented demo/run target that starts Gradio with the active artifact config and a smoke/check target that verifies the pointer before presentation.

**Implementation notes:**
- Use Gradio Blocks for layout control.
- Keep inference preprocessing identical to validation preprocessing.
- Load model path, class mapping, calibration temperature, and preprocessing metadata from explicit artifact/config files so the final winner can replace the initial EfficientNet-B2 artifact without rewriting UI code.
- The final model swap should be a Makefile/config operation, not a manual code edit inside Gradio logic.
- Include sample images only if licensing/source is documented.
- Use Gradio loading/progress/status feedback such as “Analyzing vehicle image…” while prediction runs; this is a UX requirement, not a model-speed substitute.

### Phase 6: Hugging Face Spaces Demo Deploy and Model Publishing

**Branch:** `feat/hugging-face-spaces-deploy`

**Goal:** Package the local Gradio + ONNX Runtime demo for Hugging Face Spaces CPU Basic using the Gradio SDK, and publish completed model artifacts to the student Hugging Face account as model repositories. Do not redistribute the merged raw dataset unless every source license explicitly allows it.

**Requirements:** DEPLOY-01, DEPLOY-02, DEPLOY-03, DEPLOY-04, DEPLOY-05, PUBLISH-01, PUBLISH-02, PUBLISH-03

**Success criteria:**
1. Local Gradio demo passes before deploy packaging starts.
2. Deploy package contains only required demo files: `app.py`, `requirements.txt`, active model config, ONNX artifact, metadata/calibration/class labels, and minimal inference/UI source.
3. Deploy package excludes training datasets, final instructor test images, secrets, `.env`, and unnecessary checkpoints.
4. Manual HF Spaces runbook explains Gradio SDK, CPU Basic, build logs, public URL test, and no-Docker path.
5. Makefile includes package/deploy helper targets where safe; if CLI auth is missing, manual fallback steps are documented.
6. Completed model artifacts can be packaged for HF model repos with `model.safetensors`, ONNX export, README model cards, metrics, calibration metadata, source dataset links, and no raw merged dataset images.

**Implementation notes:**
- Use Hugging Face Spaces + Gradio + ONNX Runtime on CPU Basic for public demo link.
- Training and model selection remain local; Spaces runs inference only.
- Docker Spaces, FastAPI, paid GPU deploy, and production API are out of scope.
- Dataset publishing is avoided by default because mixed Kaggle/HF/public sources may not permit redistribution; model cards should link/document sources instead.

### Phase 7: Final Evidence and IEEE Report

**Branch:** `feat/final-evidence-and-ieee-report`

**Goal:** Prepare final submission evidence and generate the IEEE-format LaTeX report using the local report creator skill.

**Requirements:** RPT-01, RPT-02, RPT-03

**Success criteria:**
1. Final artifacts include model/checkpoint, class mapping, metrics, plots, UI evidence, and HF Spaces deploy evidence/link when available.
2. Report generation uses the `awesome-ieee-report` / LaTeX report creator skill.
3. Report includes dataset sources, preprocessing, architecture rationale, model comparison, metrics, plots, normalized confusion matrix, UI description, and deploy/demo notes.
4. Report is minimum 4 pages in IEEE format and includes required LaTeX sources, not only PDF.
5. Final README explains how to install, train/evaluate, run local UI, reproduce results, and optionally open/redeploy the HF Spaces demo.

**Implementation notes:**
- Report is intentionally last so it can cite actual implementation, local demo, optional HF Spaces deploy, and final model evidence.
- The report body must avoid raw code dumps and use academic prose/pseudocode only.

## Requirement Coverage Validation

All 33 v1 requirements plus added calibration, orchestration, loading UX, and deploy requirements are mapped to roadmap phases.

---
*Roadmap created: 2026-05-06*
*Revised: 2026-05-09 — prioritized current-checkpoint export/calibration/demo before overnight experiments*
*Revised: 2026-05-09 — added loading-state UX and Phase 6 Hugging Face Spaces deploy before Phase 7 report*
