<!-- GSD:project-start source:PROJECT.md -->
## Project

**AutoLens AI — Yazlab 2 Proje 3**

AutoLens AI is a Python 3.12 / uv based computer vision project for classifying car body type from uploaded vehicle images. It targets the Kocaeli University Yazılım Laboratuvarı-II Project III requirements: build a custom dataset from open sources, train and compare multiple image classifiers, report Accuracy/Precision/Recall/F1 and normalized confusion matrix, and deliver a live web demo.

**Core Value:** The system must generalize to unseen presentation/test images and return the correct 8-class car body type with a fast, explainable prediction workflow.

### Constraints

- **Runtime**: Python 3.12 with uv — user-selected environment and dependency manager.
- **Hardware**: Apple Silicon M2 Pro / MPS expected — training code must support MPS and CPU fallback.
- **Deadline**: 30.05.2026 — roadmap must prioritize a working baseline before deep optimization.
- **Model Size**: Final submitted model artifact must be under 95 MB — required by assignment.
- **Metrics**: F1-score is the first-priority performance metric; Accuracy, Precision, Recall also required.
- **Classes**: Exactly 8 target classes must be supported, matching the assignment labels.
- **UI**: Web interface must support image upload, preview, prediction output, confidence, and class probability visualization.
- **Report**: IEEE-format LaTeX report, minimum 4 pages, generated at the end using the local LaTeX report creator skill.
<!-- GSD:project-end -->

<!-- GSD:stack-start source:research/STACK.md -->
## Technology Stack

## Runtime and Packaging
- Python 3.12 managed by uv.
- uv commands are the canonical install path:
## Verified Package Version Snapshot
| Package | Current PyPI version | Python requirement |
|---|---:|---|
| torch | 2.11.0 | >=3.10 |
| torchvision | 0.26.0 | >=3.10, !=3.14.1 |
| pytorch-lightning | 2.6.1 | >=3.10 |
| timm | 1.0.26 | >=3.8 |
| albumentations | 2.0.8 | >=3.9 |
| opencv-python | 4.13.0.92 | >=3.6 |
| pillow | 12.2.0 | >=3.10 |
| torchmetrics | 1.9.0 | >=3.10 |
| gradio | 6.14.0 | >=3.10 |
| onnxruntime | 1.25.1 | >=3.11 |
| transformers | 5.8.0 | >=3.10 |
| datasets | 4.8.5 | >=3.10 |
| huggingface_hub | 1.14.0 | >=3.10 |
## Model Stack
## Developer Practices
- Hydra/YAML for experiment configs.
- Lightning for clean train/validation loops, early stopping, checkpointing, and MPS/CPU device handling.
- torchmetrics + scikit-learn for required metrics and per-class reports.
- torchinfo and saved artifact size checks before selecting the final model.
- ruff, mypy, pytest, pre-commit for maintainable code.
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

Conventions not yet established. Will populate as patterns emerge during development.
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

Architecture not yet mapped. Follow existing patterns found in the codebase.
<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->
## Project Skills

| Skill | Description | Path |
|-------|-------------|------|
| awesome-ieee-report | Generates IEEE-format LaTeX technical reports from project requirements and codebase evidence, using academic prose, pseudocode-only explanations, BibTeX references, and IEEEtran formatting. | `.agents/skills/latex-report-creator/SKILL.md` |
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->



<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
