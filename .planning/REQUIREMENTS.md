# Requirements: AutoLens AI — Yazlab 2 Proje 3

**Defined:** 2026-05-06
**Core Value:** The system must generalize to unseen presentation/test images and return the correct 8-class car body type with a fast, explainable prediction workflow.

## v1 Requirements

### Environment

- [ ] **ENV-01**: Developer can initialize the project with uv and Python 3.12.
- [ ] **ENV-02**: Developer can install all planned runtime and dev dependencies with documented `uv add` commands.
- [ ] **ENV-03**: Developer can run lint, type checks, and tests through documented commands.

### Dataset

- [ ] **DATA-01**: Developer can download or ingest candidate Kaggle datasets for car body type classification.
- [ ] **DATA-02**: Developer can download or ingest candidate Hugging Face datasets for car/vehicle classification.
- [ ] **DATA-03**: Developer can map raw labels into exactly 8 target classes: SUV, VAN, STATION WAGON, MICRO, OPEN WHEEL, SEDAN, HATCHBACK, PICK UP.
- [ ] **DATA-04**: Developer can produce a dataset manifest listing source, license/URL, class mapping, image counts, and known gaps.
- [ ] **DATA-05**: Developer can create train/validation splits without using instructor final test images.
- [ ] **DATA-06**: Developer can audit class balance and identify classes requiring additional collection.

### Training

- [ ] **TRN-01**: Developer can train MobileNetV4 Conv Medium on the curated dataset.
- [ ] **TRN-02**: Developer can train EfficientNet-B2 on the curated dataset.
- [ ] **TRN-03**: Developer can train one ResNet baseline on the curated dataset.
- [ ] **TRN-04**: Developer can evaluate DINOv3 ViT-S/16 using `facebook/dinov3-vits16-pretrain-lvd1689m` when gated access is available.
- [ ] **TRN-05**: Developer can run DINOv3 without LoRA first and keep LoRA as an optional extension.
- [ ] **TRN-06**: Training supports MPS when available and CPU fallback when not available.
- [ ] **TRN-07**: Training uses augmentation, normalization, early stopping, and checkpointing to reduce overfitting.

### Evaluation

- [ ] **EVAL-01**: Developer can compute Accuracy, Precision, Recall, and F1-score for all experiments.
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

### Reporting

- [ ] **RPT-01**: Developer can collect all dataset, model, metric, plot, and UI evidence needed for the final report.
- [ ] **RPT-02**: Developer can generate the final IEEE-format LaTeX report using the local LaTeX report creator skill.
- [ ] **RPT-03**: Report includes required metrics, graphs, normalized confusion matrix, model rationale, dataset sources, and UI evidence.

## v2 Requirements

### Optimization

- **OPT-01**: Developer can export the selected model to ONNX and run ONNX Runtime inference.
- **OPT-02**: Developer can simplify ONNX graph with onnxsim.
- **OPT-03**: Developer can run Optuna hyperparameter tuning after all baseline models work.
- **OPT-04**: Developer can add Grad-CAM visualizations for supported CNN models.
- **OPT-05**: Developer can add LoRA fine-tuning for DINOv3 if time and access allow.

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
| ENV-01 | Phase 1 | Pending |
| ENV-02 | Phase 1 | Pending |
| ENV-03 | Phase 1 | Pending |
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
| UI-01 | Phase 5 | Pending |
| UI-02 | Phase 5 | Pending |
| UI-03 | Phase 5 | Pending |
| UI-04 | Phase 5 | Pending |
| UI-05 | Phase 5 | Pending |
| UI-06 | Phase 5 | Pending |
| UI-07 | Phase 5 | Pending |
| RPT-01 | Phase 6 | Pending |
| RPT-02 | Phase 6 | Pending |
| RPT-03 | Phase 6 | Pending |

**Coverage:**
- v1 requirements: 33 total
- Mapped to phases: 33
- Unmapped: 0

---
*Requirements defined: 2026-05-06*
*Last updated: 2026-05-06 after roadmap initialization*
