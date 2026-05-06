# Stack Research

## Runtime and Packaging

- Python 3.12 managed by uv.
- uv commands are the canonical install path:
  - `uv init --python 3.12`
  - `uv add torch torchvision pytorch-lightning timm albumentations opencv-python pillow`
  - `uv add torchmetrics wandb scikit-learn numpy pandas matplotlib seaborn grad-cam`
  - `uv add gradio safetensors onnx onnxruntime onnxsim torchinfo`
  - `uv add kaggle tqdm optuna hydra-core python-dotenv pyyaml rich loguru`
  - `uv add huggingface_hub datasets transformers`
  - `uv add --dev ruff mypy pre-commit pytest`

## Verified Package Version Snapshot

Checked against PyPI on 2026-05-06. Use these as planning anchors, but allow uv to resolve compatible transitive versions unless pinning becomes necessary.

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

1. **Main model:** DINOv3 ViT-S/16 through Hugging Face `facebook/dinov3-vits16-pretrain-lvd1689m` and transformers; model is gated, so access must be validated early.
2. **Baseline/comparison:** MobileNetV4 Conv Medium via timm when available in the installed timm build.
3. **Baseline/comparison:** EfficientNet-B2 via timm.
4. **Baseline/control:** ResNet via torchvision or timm.

DINOv3 metadata from Hugging Face: image feature extraction model, `dinov3_vit` architecture, about 21.6M parameters, gated repository. DINOv3 is the intended main architecture, but the project must include baseline fallback evidence if gated access, final artifact size, or demo latency blocks deployment.

## Developer Practices

- Hydra/YAML for experiment configs.
- Lightning for clean train/validation loops, early stopping, checkpointing, and MPS/CPU device handling.
- torchmetrics + scikit-learn for required metrics and per-class reports.
- torchinfo and saved artifact size checks before selecting the final model.
- ruff, mypy, pytest, pre-commit for maintainable code.
