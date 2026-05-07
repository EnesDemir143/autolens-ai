# Roadmap: AutoLens AI — Yazlab 2 Proje 3

**Created:** 2026-05-06
**Granularity:** Standard
**Execution:** Sequential
**Target:** 6 phases

## Overview

| # | Phase | Branch | Goal | Requirements | Success Criteria |
|---|-------|--------|------|--------------|------------------|
| 1 | Project Foundation | `main` | Create reproducible uv/Python 3.12 project skeleton and quality gates | ENV-01, ENV-02, ENV-03 | 4 |
| 2 | Dataset Research and Curation | `feat/dataset-curation-eda` | Build documented near-balanced 8-class dataset pipeline from public sources | DATA-01..DATA-06 | 5 |
| 3 | Baseline Training Pipeline | `feat/baseline-training-pipeline` | Train explainable CNN baseline/comparison models with Lightning and MPS/CPU support | TRN-01, TRN-02, TRN-03, TRN-06, TRN-07 | 5 |
| 4 | Main DINOv3 Model and Selection | `feat/model-comparison-and-selection` | Implement the intended main DINOv3 path, compare against baselines, and select by F1/size/speed | TRN-04, TRN-05, EVAL-01..EVAL-07 | 6 |
| 5 | Gradio Demo Interface | `feat/gradio-demo-interface` | Deliver modern presentation-ready web UI for live classification | UI-01..UI-07 | 5 |
| 6 | Final Evidence and IEEE Report | `feat/final-evidence-and-ieee-report` | Package plots, metrics, final model, and LaTeX IEEE report | RPT-01..RPT-03 | 5 |

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

### Phase 4: Main DINOv3 Model and Selection

**Branch:** `feat/model-comparison-and-selection`

**Goal:** Implement/evaluate DINOv3 ViT-S/16 as the intended main model, then compare the four model families — MobileNetV4 Conv Medium, EfficientNet-B2, ResNet18, and DINOv3 — by macro F1, per-class behavior, artifact size, and speed.

**Requirements:** TRN-04, TRN-05, EVAL-01, EVAL-02, EVAL-03, EVAL-04, EVAL-05, EVAL-06, EVAL-07

**Success criteria:**
1. DINOv3 access is validated early because it is the intended main model path; if gated access blocks use, the fallback is documented.
2. DINOv3 non-LoRA path is implemented/evaluated before the separate optional LoRA fine-tuning variant.
3. Every model has Accuracy, balanced accuracy, MCC, Precision, Recall, F1, macro/weighted averages, and per-class metrics.
4. Required plots are generated: loss, accuracy, normalized confusion matrix.
5. Final model candidate is under 95 MB or has a clear compression/export path.
6. Final selection rationale is written in report-ready language.

**Implementation notes:**
- DINOv3 is the intended main model; baselines exist to quantify benefit and provide fallback evidence.
- F1-score is the primary ranking metric.
- Use artifact-size and latency checks before declaring the final deployable model.

### Phase 5: Gradio Demo Interface

**Branch:** `feat/gradio-demo-interface`

**Goal:** Provide a clean, modern web UI that satisfies every assignment interface requirement.

**Requirements:** UI-01, UI-02, UI-03, UI-04, UI-05, UI-06, UI-07

**Success criteria:**
1. User can upload or drag-and-drop a vehicle image.
2. UI shows uploaded image preview.
3. UI shows predicted class clearly and prominently.
4. UI shows confidence plus probability distribution for all 8 classes.
5. UI is visually clean, modern, and demo-ready with acceptable live prediction latency.

**Implementation notes:**
- Use Gradio Blocks for layout control.
- Keep inference preprocessing identical to validation preprocessing.
- Include sample images only if licensing/source is documented.

### Phase 6: Final Evidence and IEEE Report

**Branch:** `feat/final-evidence-and-ieee-report`

**Goal:** Prepare final submission evidence and generate the IEEE-format LaTeX report using the local report creator skill.

**Requirements:** RPT-01, RPT-02, RPT-03

**Success criteria:**
1. Final artifacts include model/checkpoint, class mapping, metrics, plots, and UI evidence.
2. Report generation uses the `awesome-ieee-report` / LaTeX report creator skill.
3. Report includes dataset sources, preprocessing, architecture rationale, model comparison, metrics, plots, normalized confusion matrix, and UI description.
4. Report is minimum 4 pages in IEEE format and includes required LaTeX sources, not only PDF.
5. Final README explains how to install, train/evaluate, run UI, and reproduce results.

**Implementation notes:**
- Report is intentionally last so it can cite actual evidence.
- The report body must avoid raw code dumps and use academic prose/pseudocode only.

## Requirement Coverage Validation

All 33 v1 requirements are mapped to exactly one roadmap phase.

---
*Roadmap created: 2026-05-06*
