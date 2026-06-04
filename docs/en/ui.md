# AutoLens AI — User Interface

AutoLens AI provides two user interfaces that share the same ONNX inference backend.

## Option A — Gradio Demo

The Gradio interface is the simplest deployment path and is suitable for HuggingFace Spaces.

```bash
make download-hf-model
make demo-gradio
```

Open: `http://localhost:7860`

Features:
- Upload or paste a vehicle image
- Preview the image
- Run ONNX Runtime inference
- Show predicted class, confidence, latency, and class probabilities
- Uses the same `ONNXPredictor` class as the full web UI

## Option B — FastAPI + React Web UI

The full local web app uses a FastAPI backend and a Vite/React frontend.

```bash
make download-hf-model
make frontend-build
make demo-web
```

Open: `http://localhost:8080`

Features:
- Desktop-focused 3-column layout
- Image upload and clipboard paste support
- Prediction card with confidence and latency
- Softmax probability chart
- Model information panel
- Optional GradCAM endpoint when safetensors artifacts are available locally

## Shared Inference Contract

Both interfaces use `artifacts/demo/active_model.json` as the active model pointer:

```json
{
  "backend": "onnxruntime",
  "model_path": "artifacts/export/dinov3_safe_weighted_latest/model.onnx",
  "metadata_path": "artifacts/export/dinov3_safe_weighted_latest/metadata.json",
  "calibration_path": "artifacts/export/dinov3_safe_weighted_latest/calibration/dirichlet/best_calibration.json",
  "calibration_method": "dirichlet"
}
```

`make download-hf-model` downloads model files from HuggingFace Hub and writes this JSON automatically.

## HuggingFace Hub Models

All model binaries are hosted on HuggingFace Hub instead of being committed to GitHub:

- [DINOv3 weighted](https://huggingface.co/morpeN1/autolens-dinov3-safe-weighted) — selected model
- [DINOv3 focal](https://huggingface.co/morpeN1/autolens-dinov3-safe-focal)
- [DINOv3 ViT-S/16 baseline](https://huggingface.co/morpeN1/autolens-dinov3-vits16)
- [EfficientNet-B2](https://huggingface.co/morpeN1/autolens-efficientnet-b2)
- [ResNet18](https://huggingface.co/morpeN1/autolens-resnet18)
- [MobileNetV4](https://huggingface.co/morpeN1/autolens-mobilenetv4)

Full collection: [AutoLens Vehicle Body Classification](https://huggingface.co/collections/morpeN1/autolens-vehicle-body-classification)

## Screenshots

Screenshots from the HuggingFace Space / local web UI can be placed under `assets/screenshots/` and linked from the README.

Recommended screenshots:
1. Upload state
2. Successful prediction
3. Probability chart
4. Full React web UI
