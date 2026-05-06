# Phase 1 Research: Project Foundation

## Research Question

What needs to be known to plan a reproducible uv/Python 3.12 ML project foundation for AutoLens AI?

## Findings

### uv + Python 3.12

- `uv init --python 3.12` should be the first implementation command.
- Package additions should be documented and grouped before bulk installation to keep failures diagnosable.
- `pyproject.toml` should contain project metadata, Python requirement, and tool config for ruff/mypy/pytest where practical.

### Dependency Strategy

Use the approved `docs/plan.md` stack. Phase 1 should document install commands and may install packages during execution, but downstream phases should verify large framework behavior after installation.

Recommended fish-compatible commands:

```fish
uv add torch torchvision pytorch-lightning timm albumentations opencv-python pillow
uv add torchmetrics wandb scikit-learn numpy pandas matplotlib seaborn grad-cam
uv add gradio safetensors onnx onnxruntime onnxsim torchinfo
uv add kaggle tqdm optuna hydra-core python-dotenv pyyaml rich loguru
uv add huggingface_hub datasets transformers
uv add --dev ruff mypy pre-commit pytest
```

### Quality Gate Baseline

Start with commands that prove the foundation works:

```fish
uv run pytest
uv run ruff check .
uv run mypy src
```

Formatting can use:

```fish
uv run ruff format .
```

### Risks

- PyTorch/torchvision wheels can be platform-sensitive; install should be verified on the local Apple Silicon environment.
- DINOv3 is gated on Hugging Face, but access validation belongs in Phase 4, not Phase 1.
- Over-configuring mypy too strictly can slow ML experimentation; start with practical checks.

## Validation Architecture

Phase 1 validation proves the project foundation, not ML correctness:

1. `pyproject.toml` exists and declares Python `>=3.12` or equivalent.
2. `docs/dependencies.md` contains all approved `uv add` groups.
3. Source package imports with `uv run python -c "import autolens_ai"`.
4. `uv run pytest` exits successfully.
5. `uv run ruff check .` exits successfully.
6. `uv run mypy src` exits successfully or documents any intentionally deferred strictness.
