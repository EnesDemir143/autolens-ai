# Dependency Installation Guide

Phase 1 documents the approved dependency groups for AutoLens AI. Install commands are fish-compatible and use uv.

## Core Vision and Training

```fish
uv add torch torchvision pytorch-lightning timm albumentations opencv-python pillow
```

## Metrics and Reporting

```fish
uv add torchmetrics wandb scikit-learn numpy pandas matplotlib seaborn grad-cam
```

## Deployment and UI

```fish
uv add gradio safetensors onnx onnxruntime onnxsim torchinfo
```

## MLOps and Config

```fish
uv add kaggle tqdm optuna hydra-core python-dotenv pyyaml rich loguru
```

## Hugging Face and Transformers

```fish
uv add huggingface_hub datasets transformers
```

DINOv3 Hugging Face access is gated and validated in Phase 4.

## Developer Tools

```fish
uv add --dev ruff mypy pre-commit pytest
```

## Package Purpose Map

| Group | Packages | Purpose | Used starting phase |
|---|---|---|---|
| Dataset tools | kaggle, datasets, huggingface_hub, pandas, tqdm, opencv-python, pillow | Download, inspect, and manifest public image sources. | Phase 2 |
| Training tools | torch, torchvision, pytorch-lightning, timm, albumentations, torchmetrics, hydra-core, pyyaml | Train baseline CNN classifiers with repeatable configs and MPS/CPU fallback. | Phase 3 |
| Evaluation/model comparison tools | scikit-learn, numpy, matplotlib, seaborn, torchinfo, safetensors, onnx, onnxruntime, onnxsim, grad-cam, optuna | Compute metrics, generate plots, inspect artifact size, and support optional optimization/export work. | Phase 4 |
| Gradio UI tools | gradio, pillow, numpy, torch, torchvision | Run the demo upload, preview, prediction, confidence, and probability visualization interface. | Phase 5 |
| Report evidence tools | matplotlib, seaborn, pandas, scikit-learn, torchmetrics | Produce report-ready metrics, curves, and normalized confusion matrix evidence. | Phase 6 |
| Developer tools | ruff, mypy, pre-commit, pytest | Keep lint, format, type, and test gates runnable. | Phase 1 |


## Install Status

Installed on 2026-05-06 with the documented `uv add` commands. The runtime dependency list now lives in `pyproject.toml` and the resolved package set is locked in `uv.lock`.

Credential-dependent packages were installed, but credentials are not configured by installation:

- W&B: run `uv run wandb login` or set `WANDB_API_KEY` before experiment tracking.
- Kaggle: provide `~/.kaggle/kaggle.json` or the required Kaggle environment variables before dataset download.
- Hugging Face: run `uv run huggingface-cli login` or set `HF_TOKEN` for gated models/datasets such as DINOv3 access validation in Phase 4.

## Install Notes

- The full ML stack is documented here for reproducibility, but Phase 1 validation only requires the developer tools and import/test skeleton.
- If a future `uv add` command fails because of network or wheel availability, record the exact command, failing package, and platform details in this section before proceeding.
