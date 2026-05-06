# AutoLens AI — Yazlab 2 Proje 3

## What This Is

AutoLens AI is a Python 3.12 / uv based computer vision project for classifying car body type from uploaded vehicle images. It targets the Kocaeli University Yazılım Laboratuvarı-II Project III requirements: build a custom dataset from open sources, train and compare multiple image classifiers, report Accuracy/Precision/Recall/F1 and normalized confusion matrix, and deliver a live web demo.

## Core Value

The system must generalize to unseen presentation/test images and return the correct 8-class car body type with a fast, explainable prediction workflow.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Build a documented 8-class custom dataset using Kaggle and Hugging Face sources.
- [ ] Train and compare MobileNetV4 Conv Medium, DINOv3 ViT-S/16, EfficientNet-B2, and one ResNet baseline.
- [ ] Optimize for macro F1 first, then Accuracy, Precision, and Recall.
- [ ] Keep final deployable model artifact under the 95 MB submission limit.
- [ ] Provide a modern Gradio web UI with upload, preview, prediction, confidence score, and class probability chart.
- [ ] Produce required training/validation loss, accuracy curves, and normalized confusion matrix figures.
- [ ] Produce the IEEE LaTeX report at the end using the local LaTeX report creator skill.

### Out of Scope

- Mobile application — web UI is required and sufficient for the project demo.
- Heavy end-to-end transformer-only production path — DINOv3 will be evaluated, but a smaller model may be selected if size/speed/generalization are better.
- Test-set leakage — instructor-provided final test data must never be used during training or tuning.
- New dependencies beyond the planned stack unless a phase explicitly justifies them.

## Context

The assignment requires an 8-class car body type classifier: SUV, VAN, STATION WAGON, MİCRO, AÇIK TEKERLEKLİ / F1 vehicles, SEDAN, HATCHBACK, and PICK UP. The training data must be custom-created from public sources such as Kaggle and Hugging Face, with sources documented for the report and presentation. The final demo will be tested on unseen images, so generalization and anti-overfitting practices matter more than memorizing a single dataset.

The project will use uv with Python 3.12. The planned stack comes from `docs/plan.md`: PyTorch, torchvision, PyTorch Lightning, timm, albumentations, OpenCV, Pillow, torchmetrics, W&B, scikit-learn, NumPy, pandas, matplotlib, seaborn, grad-cam, Gradio, safetensors, ONNX/ONNX Runtime/onnxsim, torchinfo, Kaggle CLI, tqdm, Optuna, Hydra, dotenv, ruff, mypy, pre-commit, pytest, PyYAML, rich/loguru, Hugging Face Hub/datasets, and transformers.

Karpathy-guideline constraints apply: keep implementation small and explainable, avoid speculative abstractions, make every phase verifiable, and prefer code the student can explain during demo.

## Constraints

- **Runtime**: Python 3.12 with uv — user-selected environment and dependency manager.
- **Hardware**: Apple Silicon M2 Pro / MPS expected — training code must support MPS and CPU fallback.
- **Deadline**: 30.05.2026 — roadmap must prioritize a working baseline before deep optimization.
- **Model Size**: Final submitted model artifact must be under 95 MB — required by assignment.
- **Metrics**: F1-score is the first-priority performance metric; Accuracy, Precision, Recall also required.
- **Classes**: Exactly 8 target classes must be supported, matching the assignment labels.
- **UI**: Web interface must support image upload, preview, prediction output, confidence, and class probability visualization.
- **Report**: IEEE-format LaTeX report, minimum 4 pages, generated at the end using the local LaTeX report creator skill.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Use uv + Python 3.12 | Modern reproducible Python workflow; user requested it explicitly. | — Pending |
| Use Standard GSD granularity with sequential execution | User requested 5-8 phases and sequential planning. | — Pending |
| Commit `.planning/` docs | User requested planning docs in git. | — Pending |
| Compare four model families | MobileNetV4, DINOv3, EfficientNet-B2, and ResNet cover efficient CNN, transformer features, strong CNN baseline, and classic baseline. | — Pending |
| Use Gradio for UI | Fast demo delivery while satisfying web UI requirements. | — Pending |
| Generate report at the end with LaTeX skill | Final report should reflect actual implemented evidence, metrics, and UI screenshots/placeholders. | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition**:
1. Requirements invalidated? → Move to Out of Scope with reason.
2. Requirements validated? → Move to Validated with phase reference.
3. New requirements emerged? → Add to Active.
4. Decisions to log? → Add to Key Decisions.
5. "What This Is" still accurate? → Update if drifted.

**After each milestone**:
1. Full review of all sections.
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state.

---
*Last updated: 2026-05-06 after initialization*
