# Requirements: AutoLens AI — Yazlab 2 Proje 3

**Defined:** 2026-05-06
**Core Value:** The system must generalize to unseen presentation/test images and return the correct 8-class car body type with a fast, explainable prediction workflow.

## v1 Requirements

### Environment

- [x] **ENV-01**: Developer can initialize the project with uv and Python 3.12.
- [x] **ENV-02**: Developer can install all planned runtime and dev dependencies with documented `uv add` commands.
- [x] **ENV-03**: Developer can run lint, type checks, and tests through documented commands.

### Dataset

- [ ] **DATA-01**: Developer can download or ingest candidate Kaggle datasets for car body type classification.
- [ ] **DATA-02**: Developer can download or ingest candidate Hugging Face datasets for car/vehicle classification.
- [ ] **DATA-03**: Developer can map raw labels into exactly 8 target classes: SUV, VAN, STATION WAGON, MICRO, OPEN WHEEL, SEDAN, HATCHBACK, PICK UP.
- [ ] **DATA-04**: Developer can produce a dataset manifest listing source, license/URL, class mapping, image counts, known gaps, and target-vs-actual counts for the 36k–40k raw / ~32k clean dataset goal.
- [ ] **DATA-05**: Developer can create train/validation splits without using instructor final test images.
- [ ] **DATA-06**: Developer can audit class balance against an approximate 4k-clean-images-per-class target and identify classes requiring additional collection.

### Training

- [ ] **TRN-01**: Developer can train MobileNetV4 Conv Medium as a baseline/comparison model on the curated dataset.
- [ ] **TRN-02**: Developer can train EfficientNet-B2 as a baseline/comparison model on the curated dataset.
- [ ] **TRN-03**: Developer can train ResNet18 as a classic control baseline model on the curated dataset.
- [ ] **TRN-04**: Developer can run the intended main DINOv3 ViT-S/16 path using `facebook/dinov3-vits16-pretrain-lvd1689m` when gated access is available.
- [ ] **TRN-05**: Developer can run DINOv3 without LoRA first as the main-path implementation and then run a LoRA fine-tuning variant as a separate optional comparison when access/time allow.
- [ ] **TRN-06**: Training supports MPS when available and CPU fallback when not available.
- [ ] **TRN-07**: Training uses augmentation, normalization, early stopping, and checkpointing to reduce overfitting.

### Evaluation

- [ ] **EVAL-01**: Developer can compute Accuracy, balanced accuracy, MCC, Precision, Recall, and F1-score for all experiments.
- [ ] **EVAL-02**: Developer can report per-class, macro average, and weighted average metrics.
- [ ] **EVAL-03**: Developer can generate training/validation loss graph by epoch.
- [ ] **EVAL-04**: Developer can generate training/validation accuracy graph by epoch.
- [ ] **EVAL-05**: Developer can generate an 8x8 normalized confusion matrix heatmap.
- [ ] **EVAL-06**: Developer can compare model artifacts by macro F1, size, and inference speed.
- [ ] **EVAL-07**: Developer can verify the selected final model artifact is under 95 MB.

### Interface

- [ ] **UI-01**: User can upload or drag-and-drop a car image in a Gradio web interface.
- [ ] **UI-02**: User can preview the uploaded image before or during prediction.
- [ ] **UI-03**: User can trigger classification with a clear button or equivalent interaction.
- [ ] **UI-04**: User can see the predicted car body type prominently.
- [ ] **UI-05**: User can see confidence score and probability distribution for all 8 classes.
- [ ] **UI-06**: UI presents uploaded image and result in a clean, modern, presentation-ready layout.
- [ ] **UI-07**: UI prediction latency is acceptable for live demo usage.
- [ ] **UI-08**: UI shows a clear loading/progress/status state during inference so slower CPU predictions do not appear frozen.

### Deployment

- [ ] **PUBLISH-01**: Developer can publish completed model artifacts to the student Hugging Face account as model repositories with model cards.
- [ ] **PUBLISH-02**: Model publishing excludes raw merged datasets and documents dataset sources/links/license caveats instead of redistributing third-party images.
- [ ] **PUBLISH-03**: Optional Makefile/CLI publish path is documented for uploading a selected artifact directory to a Hugging Face model repo.

- [ ] **DEPLOY-01**: Developer can package the local Gradio + ONNX Runtime demo for Hugging Face Spaces without Docker.
- [ ] **DEPLOY-02**: Deploy package includes only required demo files and excludes training datasets, final instructor test images, secrets, and unnecessary checkpoints.
- [ ] **DEPLOY-03**: Deploy package uses the active artifact config/pointer so the final selected ONNX model can be swapped without Gradio code edits.
- [ ] **DEPLOY-04**: Developer has a manual HF Spaces runbook covering Gradio SDK, CPU Basic hardware, build logs, and public URL smoke test.
- [ ] **DEPLOY-05**: Optional Makefile/CLI deploy path is documented; missing Hugging Face auth falls back to manual steps.

### Reporting

- [ ] **RPT-01**: Developer can collect all dataset, model, metric, plot, and UI evidence needed for the final report.
- [ ] **RPT-02**: Developer can generate the final IEEE-format LaTeX report using the local LaTeX report creator skill.
- [ ] **RPT-03**: Report includes required metrics, graphs, normalized confusion matrix, model rationale, dataset sources, and UI evidence.

## v2 Requirements

### Optimization

- **OPT-01**: Developer can export the selected model to ONNX and run ONNX Runtime inference. For the 2026-05-09 pivot, this is first applied to the current best EfficientNet-B2 checkpoint so the demo can finish before overnight experiments complete.
- **OPT-02**: Developer can simplify ONNX graph with onnxsim and record artifact size against the 95 MB limit.
- **OPT-03**: Developer can run Optuna hyperparameter tuning after all baseline models work.
- **OPT-04**: Developer can add Grad-CAM visualizations for supported CNN models.
- **OPT-05**: Developer can add LoRA fine-tuning for the main DINOv3 path if time and access allow.


### Calibration / Demo-readiness

- **CAL-01**: Developer can fit temperature scaling on validation predictions only and save the calibration temperature/metadata beside the exported model.
- **CAL-02**: Developer can compare pre/post-calibration confidence behavior using NLL/ECE or equivalent confidence diagnostics without touching internal test or instructor final test data.
- **OPS-01**: Developer can queue long-running DINOv3 ViT-S, DINOv3 LoRA, and EfficientNet-B2 augmentation+EMA+focal-loss experiments for overnight execution while the current checkpoint demo path remains usable.
- **OPS-02**: Developer can run the post-training deployment sequence from Makefile targets: choose a run/checkpoint, export it to ONNX, check size, calibrate, update the Gradio model artifact pointer/config, and run a demo smoke check in the documented order.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Mobile app | Assignment requires web UI; Gradio is sufficient. |
| Production backend service | Demo scope does not require a deployed API. |
| Training on final instructor test data | Explicit assignment violation. |
| Full custom framework abstractions | Reduces explainability and increases deadline risk. |
| Report before implementation evidence | Final report should reflect actual results and artifacts. |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| ENV-01 | Phase 1 | Complete — Phase 1 verified 2026-05-06 |
| ENV-02 | Phase 1 | Complete — Phase 1 verified 2026-05-06 |
| ENV-03 | Phase 1 | Complete — Phase 1 verified 2026-05-06 |
| DATA-01 | Phase 2 | Pending |
| DATA-02 | Phase 2 | Pending |
| DATA-03 | Phase 2 | Pending |
| DATA-04 | Phase 2 | Pending |
| DATA-05 | Phase 2 | Pending |
| DATA-06 | Phase 2 | Pending |
| TRN-01 | Phase 3 | Pending |
| TRN-02 | Phase 3 | Pending |
| TRN-03 | Phase 3 | Pending |
| TRN-06 | Phase 3 | Pending |
| TRN-07 | Phase 3 | Pending |
| TRN-04 | Phase 4 | Pending |
| TRN-05 | Phase 4 | Pending |
| EVAL-01 | Phase 4 | Pending |
| EVAL-02 | Phase 4 | Pending |
| EVAL-03 | Phase 4 | Pending |
| EVAL-04 | Phase 4 | Pending |
| EVAL-05 | Phase 4 | Pending |
| EVAL-06 | Phase 4 | Pending |
| EVAL-07 | Phase 4 | Pending |
| OPT-01 | Phase 4 | Pending — immediate EfficientNet-B2 export first, repeat for final winner later |
| OPT-02 | Phase 4 | Pending — simplify/check artifact size where supported |
| CAL-01 | Phase 4 | Pending — validation-only temperature scaling |
| CAL-02 | Phase 4 | Pending — calibration evidence without test leakage |
| OPS-01 | Phase 4 | Pending — overnight experiment queue |
| OPS-02 | Phase 4/5 | Pending — Makefile-driven export/calibration/demo artifact orchestration |
| UI-01 | Phase 5 | Complete — 2026-05-09 |
| UI-02 | Phase 5 | Complete — 2026-05-09 |
| UI-03 | Phase 5 | Complete — 2026-05-09 |
| UI-04 | Phase 5 | Complete — 2026-05-09 |
| UI-05 | Phase 5 | Complete — 2026-05-09 |
| UI-06 | Phase 5 | Complete — 2026-05-09 |
| UI-07 | Phase 5 | Complete — 2026-05-09 |
| UI-08 | Phase 5 | Complete — 2026-05-09 |
| PUBLISH-01 | Phase 6 | Pending — publish completed model artifacts to HF model repos |
| PUBLISH-02 | Phase 6 | Pending — do not redistribute merged raw datasets; link/document sources instead |
| PUBLISH-03 | Phase 6 | Pending — optional Makefile/CLI model publish path |
| DEPLOY-01 | Phase 6 | Pending — HF Spaces package without Docker |
| DEPLOY-02 | Phase 6 | Pending — safe deploy package contents |
| DEPLOY-03 | Phase 6 | Pending — active artifact config reuse |
| DEPLOY-04 | Phase 6 | Pending — manual Spaces runbook |
| DEPLOY-05 | Phase 6 | Pending — optional Makefile/CLI deploy path |
| RPT-01 | Phase 7 | Pending |
| RPT-02 | Phase 7 | Pending |
| RPT-03 | Phase 7 | Pending |

**Coverage:**
- v1 requirements: 33 total
- v2/v3 execution requirements now mapped for export, calibration, and overnight run orchestration
- Mapped to phases: 49
- Unmapped: 0

---
*Requirements defined: 2026-05-06*
*Last updated: 2026-05-09 after loading UX and HF Spaces deploy phase planning update*
