"""ONNX Runtime predictor for AutoLens AI.

Loads a model artifact via the active demo config and runs inference.
Model-agnostic: any ONNX export that follows the Phase 4 artifact contract works.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]
from PIL import Image


def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Stable softmax with optional temperature scaling."""
    x = x / temperature
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


class ONNXPredictor:
    """Load an ONNX model artifact and predict car body type from images."""

    def __init__(self, artifact_config_path: str | Path) -> None:
        """Initialize predictor from the active artifact JSON config.

        Args:
            artifact_config_path: Path to the active artifact config JSON
                (e.g., artifacts/demo/active_model.json).
        """
        self.config_path = Path(artifact_config_path)
        self.config: dict[str, Any] = json.loads(
            self.config_path.read_text(encoding="utf-8")
        )

        self.model_path = Path(self.config["model_path"])
        self.metadata_path = Path(self.config["metadata_path"])
        self.temperature = float(self.config.get("temperature", 1.0))
        self.class_labels: list[str] = self.config["class_labels"]
        self.display_labels: list[str] = self.config.get(
            "display_labels", self.class_labels
        )
        self.preprocessing: dict[str, Any] = self.config["preprocessing"]

        # Build ONNX Runtime session (CPU by default for maximum compatibility)
        self.session = ort.InferenceSession(str(self.model_path))
        self.input_name = self.session.get_inputs()[0].name

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        """Convert PIL image to model input tensor (NCHW float32)."""
        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        size = self.preprocessing.get("image_size", 224)
        resize_size = self.preprocessing.get("resize_size", size + 32)
        crop_size = self.preprocessing.get("crop_size", size)
        mean = self.preprocessing.get("mean", [0.485, 0.456, 0.406])
        std = self.preprocessing.get("std", [0.229, 0.224, 0.225])

        # Resize and center crop
        image = image.resize((resize_size, resize_size), Image.BICUBIC)
        left = (resize_size - crop_size) // 2
        top = (resize_size - crop_size) // 2
        image = image.crop((left, top, left + crop_size, top + crop_size))

        # To numpy and normalize
        arr = np.array(image).astype(np.float32) / 255.0
        mean_np = np.array(mean, dtype=np.float32)
        std_np = np.array(std, dtype=np.float32)
        arr = (arr - mean_np) / std_np

        # HWC -> CHW -> NCHW
        arr = np.transpose(arr, (2, 0, 1))
        arr = np.expand_dims(arr, axis=0)
        return arr.astype(np.float32)

    def predict(self, image: Image.Image) -> dict[str, Any]:
        """Run inference on a PIL image and return structured results.

        Returns:
            Dict with keys:
                - predicted_class: str
                - predicted_display: str
                - confidence: float
                - probabilities: dict[str, float]  # label -> prob
                - latency_ms: float
                - temperature: float
        """
        import time

        start = time.perf_counter()
        input_tensor = self._preprocess(image)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        logits = outputs[0][0]  # batch dim removed
        probs = _softmax(logits, temperature=self.temperature)
        latency_ms = (time.perf_counter() - start) * 1000

        # Build probability dict
        prob_dict = {
            label: float(prob) for label, prob in zip(self.class_labels, probs)
        }
        sorted_idx = int(np.argmax(probs))
        predicted_class = self.class_labels[sorted_idx]
        predicted_display = self.display_labels[sorted_idx]
        confidence = float(probs[sorted_idx])

        return {
            "predicted_class": predicted_class,
            "predicted_display": predicted_display,
            "confidence": confidence,
            "probabilities": prob_dict,
            "latency_ms": round(latency_ms, 2),
            "temperature": self.temperature,
        }

    @classmethod
    def from_default_config(cls) -> "ONNXPredictor":
        """Load predictor using the default active artifact config path."""
        default_path = Path("artifacts/demo/active_model.json")
        if not default_path.exists():
            raise FileNotFoundError(
                f"Default artifact config not found: {default_path}\n"
                "Run: make prepare-demo-artifact"
            )
        return cls(default_path)
