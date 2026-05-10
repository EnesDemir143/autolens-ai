"""FastAPI backend for AutoLens AI — Plan 05-05.

Serves:
  GET  /api/health           — liveness check
  GET  /api/models           — all export artifacts with metadata, metrics, is_active
  POST /api/models/active    — switch active model (body: {"id": "..."}), asyncio.Lock serialised
  POST /api/predict          — multipart/form-data image → prediction JSON
  POST /api/gradcam          — multipart/form-data image → GradCAM overlay base64 PNG
  GET  /                     — serve frontend/dist/index.html (production)

app.py (Gradio) is NOT imported or modified.
ONNXPredictor is NOT modified.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from autolens_ai.inference.artifact import load_artifact_metadata, load_model_from_artifact
from autolens_ai.inference.onnx_predictor import ONNXPredictor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACTIVE_MODEL_JSON = Path("artifacts/demo/active_model.json")
EXPORT_ROOT = Path("artifacts/export")
ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB

# Directory names that map to export subdirs (order = display order in UI)
_EXPORT_DIRS: list[str] = [
    "efficientnet_b2_current",
    "resnet18_latest",
    "mobilenetv4_latest",
]

# ---------------------------------------------------------------------------
# State shared across requests
# ---------------------------------------------------------------------------

_predictor_cache: dict[str, ONNXPredictor] = {}  # id → ONNXPredictor
_gradcam_cache: dict[str, torch.nn.Module] = {}   # id → PyTorch model (for GradCAM)
_active_model_id: str = ""
_model_switch_lock = asyncio.Lock()

# ---------------------------------------------------------------------------
# Helpers — model discovery
# ---------------------------------------------------------------------------


def _load_active_model_id() -> str:
    """Derive model id from active_model.json metadata_path."""
    cfg = json.loads(ACTIVE_MODEL_JSON.read_text(encoding="utf-8"))
    meta_path = Path(cfg["metadata_path"])
    # metadata_path is like artifacts/export/<id>/metadata.json
    return meta_path.parent.name


def _export_dirs() -> list[Path]:
    """Return existing export subdirectories in the order defined by _EXPORT_DIRS."""
    result: list[Path] = []
    for name in _EXPORT_DIRS:
        d = EXPORT_ROOT / name
        if d.is_dir() and (d / "metadata.json").exists():
            result.append(d)
    return result


def _build_model_response(export_dir: Path, is_active: bool) -> dict[str, Any]:
    """Build the JSON object for one model as returned by /api/models."""
    meta = json.loads((export_dir / "metadata.json").read_text(encoding="utf-8"))

    # per-class from internal_test_results.json if present
    per_class: dict[str, Any] = {}
    test_results_path = export_dir / "internal_test_results.json"
    if test_results_path.exists():
        tr = json.loads(test_results_path.read_text(encoding="utf-8"))
        per_class = tr.get("per_class", {})

    # training metrics
    training_meta = meta.get("training", {})
    metrics_summary = training_meta.get("metrics_summary", {})

    # temperature from calibration.json (preferred) → metadata fallback → None
    temperature: float | None = None
    calibration_path = export_dir / "calibration.json"
    if calibration_path.exists():
        cal = json.loads(calibration_path.read_text(encoding="utf-8"))
        temperature = float(cal.get("temperature", 1.0))
    else:
        temperature = training_meta.get("temperature")

    return {
        "id": export_dir.name,
        "model_name": meta["model"]["model_name"],
        "accuracy": metrics_summary.get("test/acc"),
        "f1_macro": metrics_summary.get("test/f1_macro"),
        "f1_weighted": metrics_summary.get("test/f1_weighted"),
        "val_loss": metrics_summary.get("test/loss"),
        "temperature": temperature,
        "num_classes": meta["model"]["num_classes"],
        "input_size": meta["preprocessing"]["image_size"],
        "pretrained": meta["model"].get("pretrained_at_training", False),
        "source_run_id": meta["source"]["run_id"],
        "is_active": is_active,
        "per_class": per_class,
        "training": {
            "best_epoch": training_meta.get("epoch"),
            "total_epochs_run": metrics_summary.get("epoch"),
            "learning_rate": training_meta.get("learning_rate"),
            "weight_decay": training_meta.get("weight_decay"),
        },
        "preprocessing": meta.get("preprocessing", {}),
    }


def _active_model_config() -> dict[str, Any]:
    return json.loads(ACTIVE_MODEL_JSON.read_text(encoding="utf-8"))


def _predictor_for(model_id: str) -> ONNXPredictor:
    """Return cached ONNXPredictor for model_id, loading from its metadata if needed."""
    if model_id in _predictor_cache:
        return _predictor_cache[model_id]

    export_dir = EXPORT_ROOT / model_id
    if not export_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")

    meta = json.loads((export_dir / "metadata.json").read_text(encoding="utf-8"))

    # Resolve ONNX path (prefer simplified, fall back to regular)
    simplified = export_dir / "model.simplified.onnx"
    regular = export_dir / "model.onnx"
    onnx_path = simplified if simplified.exists() else regular

    # Build a temporary active_model-style config dict for ONNXPredictor
    # ONNXPredictor reads from a JSON file, so we write a temp one.
    # To avoid mutating active_model.json we call the constructor with a real path.
    # Easier: construct a minimal JSON and pass the path.
    calibration_path = export_dir / "calibration.json"
    temperature: float = 1.0
    if calibration_path.exists():
        cal = json.loads(calibration_path.read_text(encoding="utf-8"))
        temperature = float(cal.get("temperature", 1.0))

    tmp_cfg = {
        "backend": "onnxruntime",
        "model_path": str(onnx_path),
        "metadata_path": str(export_dir / "metadata.json"),
        "calibration_path": str(calibration_path) if calibration_path.exists() else None,
        "temperature": temperature,
        "class_labels": meta["classes"]["labels"],
        "display_labels": meta["classes"].get("display_labels", meta["classes"]["labels"]),
        "preprocessing": meta["preprocessing"],
        "source_run_id": meta["source"]["run_id"],
    }
    tmp_path = export_dir / "_api_cache_config.json"
    tmp_path.write_text(json.dumps(tmp_cfg), encoding="utf-8")

    predictor = ONNXPredictor(tmp_path)
    _predictor_cache[model_id] = predictor
    return predictor


def _pytorch_model_for(model_id: str) -> torch.nn.Module:
    """Return cached PyTorch model for GradCAM, loading from safetensors if needed."""
    if model_id in _gradcam_cache:
        return _gradcam_cache[model_id]

    export_dir = EXPORT_ROOT / model_id
    safetensors = export_dir / "model.safetensors"
    metadata = export_dir / "metadata.json"
    if not safetensors.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No safetensors found for model '{model_id}'.",
        )
    model = load_model_from_artifact(safetensors, metadata, map_location="cpu")
    model.eval()
    _gradcam_cache[model_id] = model
    return model


# ---------------------------------------------------------------------------
# Lifespan — preload active model
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    global _active_model_id
    _active_model_id = _load_active_model_id()
    # preload predictor so first request is not cold
    _predictor_for(_active_model_id)
    yield
    _predictor_cache.clear()
    _gradcam_cache.clear()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="AutoLens AI", version="1.0.0", lifespan=lifespan)

# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------


async def _read_validated_image(file: UploadFile) -> Image.Image:
    """Read, validate size+type, and return PIL Image."""
    content_type = file.content_type or ""
    if content_type not in ALLOWED_MIME:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type: '{content_type}'. Allowed: jpeg, png, webp.",
        )
    data = await file.read()
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {len(data)} bytes. Max: {MAX_FILE_BYTES} bytes.",
        )
    return Image.open(io.BytesIO(data)).convert("RGB")


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------


@app.get("/api/health")
async def health() -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "active_model": _active_model_id,
        "cached_models": list(_predictor_cache.keys()),
    })


@app.get("/api/models")
async def get_models() -> JSONResponse:
    models = []
    for d in _export_dirs():
        models.append(_build_model_response(d, is_active=(d.name == _active_model_id)))
    return JSONResponse(models)


class ActiveModelRequest(BaseModel):
    id: str


@app.post("/api/models/active")
async def set_active_model(body: ActiveModelRequest) -> JSONResponse:
    global _active_model_id

    async with _model_switch_lock:
        model_id = body.id
        export_dir = EXPORT_ROOT / model_id
        if not export_dir.is_dir():
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")

        # Preload into cache if not already there (don't evict others)
        _predictor_for(model_id)

        # Update active_model.json to maintain the contract
        meta = json.loads((export_dir / "metadata.json").read_text(encoding="utf-8"))
        calibration_path = export_dir / "calibration.json"
        simplified = export_dir / "model.simplified.onnx"
        regular = export_dir / "model.onnx"
        onnx_path = simplified if simplified.exists() else regular

        temperature: float = 1.0
        if calibration_path.exists():
            cal = json.loads(calibration_path.read_text(encoding="utf-8"))
            temperature = float(cal.get("temperature", 1.0))

        new_cfg = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
            "backend": "onnxruntime",
            "model_path": str(onnx_path),
            "metadata_path": str(export_dir / "metadata.json"),
            "calibration_path": str(calibration_path) if calibration_path.exists() else None,
            "temperature": temperature,
            "class_labels": meta["classes"]["labels"],
            "display_labels": meta["classes"].get("display_labels", meta["classes"]["labels"]),
            "preprocessing": meta["preprocessing"],
            "source_run_id": meta["source"]["run_id"],
            "contract": "Phase 5 should replace this JSON pointer, not edit UI prediction code, when final winner changes.",
        }
        ACTIVE_MODEL_JSON.write_text(json.dumps(new_cfg, indent=2), encoding="utf-8")
        _active_model_id = model_id

    return JSONResponse({"ok": True, "active_model": _active_model_id})


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    image = await _read_validated_image(file)
    predictor = _predictor_for(_active_model_id)
    result = predictor.predict(image)
    return JSONResponse(result)


@app.post("/api/gradcam")
async def gradcam(file: UploadFile = File(...)) -> JSONResponse:
    """Run GradCAM on the active model and return base64 PNG overlay."""
    from pytorch_grad_cam import GradCAM  # type: ignore[import-untyped]
    from pytorch_grad_cam.utils.image import show_cam_on_image  # type: ignore[import-untyped]
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget  # type: ignore[import-untyped]

    image = await _read_validated_image(file)

    model = _pytorch_model_for(_active_model_id)

    # Determine target layer based on model_name from metadata
    export_dir = EXPORT_ROOT / _active_model_id
    meta = json.loads((export_dir / "metadata.json").read_text(encoding="utf-8"))
    model_name: str = meta["model"]["model_name"]

    # Map model architecture to last conv layer
    # load_model_from_artifact returns the raw timm model (not wrapped in .model)
    def _last_conv(m: torch.nn.Module) -> torch.nn.Module:
        """Walk all modules; return last Conv2d found."""
        last: torch.nn.Module | None = None
        for mod in m.modules():
            if isinstance(mod, torch.nn.Conv2d):
                last = mod
        if last is None:
            raise ValueError("No Conv2d found in model.")
        return last

    target_layers: list[torch.nn.Module]
    if "efficientnet" in model_name:
        try:
            # timm EfficientNet: conv_head is the last conv before bn2/pool/classifier
            target_layers = [model.conv_head]  # type: ignore[attr-defined]
        except AttributeError:
            target_layers = [_last_conv(model)]
    elif "resnet" in model_name:
        try:
            target_layers = [model.layer4[-1]]  # type: ignore[attr-defined]
        except AttributeError:
            target_layers = [_last_conv(model)]
    elif "mobilenet" in model_name:
        try:
            target_layers = [model.blocks[-1][-1]]  # type: ignore[attr-defined]
        except AttributeError:
            target_layers = [_last_conv(model)]
    else:
        target_layers = [_last_conv(model)]

    # Preprocess to tensor (same pipeline as ONNXPredictor)
    pre = meta["preprocessing"]
    size = pre.get("image_size", 224)
    resize_size = pre.get("resize_size", size + 32)
    crop_size = pre.get("crop_size", size)
    mean = np.array(pre.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
    std = np.array(pre.get("std", [0.229, 0.224, 0.225]), dtype=np.float32)

    img_resized = image.resize((resize_size, resize_size), Image.BILINEAR)
    left = (resize_size - crop_size) // 2
    top = (resize_size - crop_size) // 2
    img_cropped = img_resized.crop((left, top, left + crop_size, top + crop_size))

    arr = np.array(img_cropped).astype(np.float32) / 255.0
    arr_normalized = (arr - mean) / std
    input_tensor = torch.from_numpy(np.transpose(arr_normalized, (2, 0, 1))).unsqueeze(0)

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]

    visualization = show_cam_on_image(arr, grayscale_cam, use_rgb=True)
    pil_out = Image.fromarray(visualization)

    # Side-by-side: original | heatmap overlay
    w, h = img_cropped.size
    combined = Image.new("RGB", (w * 2, h))
    combined.paste(img_cropped, (0, 0))
    combined.paste(pil_out, (w, 0))

    buf = io.BytesIO()
    combined.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return JSONResponse({"image": f"data:image/png;base64,{b64}"})


# ---------------------------------------------------------------------------
# Frontend static files (production) — mount LAST, /api/* routes take priority
# StaticFiles(html=True) handles SPA fallback (serves index.html for unknown paths)
# ---------------------------------------------------------------------------

_dist = Path("frontend/dist")
if _dist.is_dir():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="frontend")
