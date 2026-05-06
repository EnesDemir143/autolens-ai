# Roadmap: AutoLens AI — Yazlab 2 Proje 3

**Created:** 2026-05-06
**Granularity:** Standard
**Execution:** Sequential
**Target:** 6 phases

## Overview

| # | Phase | Goal | Requirements | Success Criteria |
|---|-------|------|--------------|------------------|
| 1 | Project Foundation | Create reproducible uv/Python 3.12 project skeleton and quality gates | ENV-01, ENV-02, ENV-03 | 4 |
| 2 | Dataset Research and Curation | Build documented 8-class dataset pipeline from Kaggle + Hugging Face sources | DATA-01..DATA-06 | 5 |
| 3 | Baseline Training Pipeline | Train explainable CNN baselines with Lightning and MPS/CPU support | TRN-01, TRN-02, TRN-03, TRN-06, TRN-07 | 5 |
| 4 | Model Comparison and Selection | Evaluate DINOv3 plus all baselines, select final model by F1/size/speed | TRN-04, TRN-05, EVAL-01..EVAL-07 | 6 |
| 5 | Gradio Demo Interface | Deliver modern presentation-ready web UI for live classification | UI-01..UI-07 | 5 |
| 6 | Final Evidence and IEEE Report | Package plots, metrics, final model, and LaTeX IEEE report | RPT-01..RPT-03 | 5 |

## Phase Details

### Phase 1: Project Foundation

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

**Goal:** Convert public Kaggle/Hugging Face sources into a documented 8-class training/validation dataset.

**Requirements:** DATA-01, DATA-02, DATA-03, DATA-04, DATA-05, DATA-06

**Success criteria:**
1. Candidate sources are downloaded or documented with exact IDs/URLs.
2. Raw labels are mapped into the 8 required assignment classes.
3. A manifest records source, class mapping, counts, known gaps, and license notes.
4. Train/validation split is created without final-test leakage.
5. Class imbalance report identifies weak classes such as MICRO, STATION WAGON, and OPEN WHEEL/F1.

**Implementation notes:**
- Start with assignment-referenced Kaggle datasets.
- Use Hugging Face candidates to fill body-style gaps where possible.
- Keep manual review hooks for mislabeled/ambiguous images.

### Phase 3: Baseline Training Pipeline

**Goal:** Build the reusable training/evaluation path and train MobileNetV4 Conv Medium, EfficientNet-B2, and ResNet baseline.

**Requirements:** TRN-01, TRN-02, TRN-03, TRN-06, TRN-07

**Success criteria:**
1. Lightning training loop runs on MPS or CPU fallback.
2. Augmentation and normalization are consistent between training and validation.
3. Early stopping and checkpointing work.
4. Three CNN-family baselines produce checkpoints and validation metrics.
5. Code remains small enough for line-by-line demo explanation.

**Implementation notes:**
- Use timm for MobileNetV4/EfficientNet where possible.
- Use torchvision/timm for ResNet baseline.
- If MobileNetV4 Conv Medium name is unavailable in timm, phase must document the closest supported timm model and rationale before substitution.

### Phase 4: Model Comparison and Selection

**Goal:** Evaluate DINOv3 ViT-S/16 and compare all models by macro F1, per-class behavior, artifact size, and speed.

**Requirements:** TRN-04, TRN-05, EVAL-01, EVAL-02, EVAL-03, EVAL-04, EVAL-05, EVAL-06, EVAL-07

**Success criteria:**
1. DINOv3 access is validated; if gated access blocks use, the fallback is documented.
2. DINOv3 non-LoRA path is evaluated before any optional LoRA extension.
3. Every model has Accuracy, Precision, Recall, F1, macro/weighted averages, and per-class metrics.
4. Required plots are generated: loss, accuracy, normalized confusion matrix.
5. Final model candidate is under 95 MB or has a clear compression/export path.
6. Final selection rationale is written in report-ready language.

**Implementation notes:**
- F1-score is the primary ranking metric.
- Use artifact-size and latency checks before declaring the winner.

### Phase 5: Gradio Demo Interface

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
