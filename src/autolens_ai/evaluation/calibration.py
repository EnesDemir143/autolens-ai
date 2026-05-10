"""Validation-only temperature scaling utilities."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]
import torch
import torch.nn.functional as F
from tqdm import tqdm  # type: ignore[import-untyped]

from autolens_ai.training.dataset import AutoLensDataset
from autolens_ai.training.preprocessing import PreprocessConfig


def expected_calibration_split(path: Path) -> None:
    """Reject held-out/final splits for calibration fitting."""
    lowered = str(path).lower()
    if "internal_test" in lowered or "final" in lowered or "instructor" in lowered:
        raise ValueError(f"Calibration split must be validation-only, got forbidden path: {path}")


def collect_logits(
    onnx_path: Path,
    metadata: dict[str, Any],
    batch_size: int,
    limit: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run ONNX inference over the validation split and return logits/labels."""
    data = metadata["data"]
    val_csv = Path(data["val_csv"])
    expected_calibration_split(val_csv)
    preprocess = metadata["preprocessing"]
    dataset = AutoLensDataset(
        csv_path=val_csv,
        root_dir=Path(data["data_root"]),
        transform=PreprocessConfig(
            resize_size=int(preprocess["resize_size"]),
            crop_size=int(preprocess["crop_size"]),
            mean=tuple(preprocess["mean"]),
            std=tuple(preprocess["std"]),
        ).get_val_transform(),
    )
    if limit is not None:
        dataset.samples = dataset.samples[:limit]

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    logits_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    for images, labels in tqdm(loader, desc="validation logits"):
        output = session.run(None, {input_name: images.numpy().astype(np.float32)})[0]
        logits_batches.append(torch.from_numpy(output).float())
        label_batches.append(labels.long())
    return torch.cat(logits_batches), torch.cat(label_batches)


def ece_score(probs: torch.Tensor, labels: torch.Tensor, bins: int = 15) -> float:
    """Expected calibration error over equally spaced confidence bins."""
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels)
    ece = torch.zeros((), dtype=torch.float32)
    boundaries = torch.linspace(0, 1, bins + 1)
    for lower, upper in zip(boundaries[:-1], boundaries[1:], strict=True):
        in_bin = confidences.gt(lower) & confidences.le(upper)
        prop = in_bin.float().mean()
        if prop.item() > 0:
            ece += torch.abs(confidences[in_bin].mean() - accuracies[in_bin].float().mean()) * prop
    return float(ece.item())


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fit a single positive temperature on validation logits."""
    log_temperature = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.1, max_iter=50, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature).clamp(0.05, 10.0)
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temperature).clamp(0.05, 10.0).detach().item())


def diagnostics(logits: torch.Tensor, labels: torch.Tensor, temperature: float) -> dict[str, float]:
    """Return confidence diagnostics at a given temperature."""
    scaled = logits / temperature
    probs = torch.softmax(scaled, dim=1)
    preds = probs.argmax(dim=1)
    return {
        "nll": float(F.cross_entropy(scaled, labels).item()),
        "ece": ece_score(probs, labels),
        "accuracy": float(preds.eq(labels).float().mean().item()),
        "mean_confidence": float(probs.max(dim=1).values.mean().item()),
    }


def collect_logits_from_csv(
    onnx_path: Path,
    metadata: dict[str, Any],
    csv_path: Path,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run ONNX inference over an arbitrary CSV split and return logits/labels."""
    preprocess = metadata["preprocessing"]
    dataset = AutoLensDataset(
        csv_path=csv_path,
        root_dir=Path(metadata["data"]["data_root"]),
        transform=PreprocessConfig(
            resize_size=int(preprocess["resize_size"]),
            crop_size=int(preprocess["crop_size"]),
            mean=tuple(preprocess["mean"]),
            std=tuple(preprocess["std"]),
        ).get_val_transform(),
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    logits_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    for images, labels in tqdm(loader, desc=f"logits [{csv_path.name}]"):
        output = session.run(None, {input_name: images.numpy().astype(np.float32)})[0]
        logits_batches.append(torch.from_numpy(output).float())
        label_batches.append(labels.long())
    return torch.cat(logits_batches), torch.cat(label_batches)


def fit_temperature_scaling(
    onnx_path: Path,
    metadata: dict[str, Any],
    batch_size: int = 64,
    limit: int | None = None,
) -> dict[str, Any]:
    """Fit validation-only temperature scaling and return serializable metadata."""
    logits, labels = collect_logits(onnx_path, metadata, batch_size, limit)
    temperature = fit_temperature(logits, labels)

    result: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "method": "temperature_scaling",
        "temperature": temperature,
        "fit_split": metadata["data"]["val_csv"],
        "forbidden_splits": ["internal_test", "instructor_final_test"],
        "num_samples": int(labels.numel()),
        "before": diagnostics(logits, labels, 1.0),
        "after": diagnostics(logits, labels, temperature),
    }

    # Evaluate calibration quality on internal test set (read-only, not used for fitting)
    test_csv_key = "test_csv_reserved_not_used_for_calibration"
    test_csv = Path(metadata["data"].get(test_csv_key, "artifacts/dataset/splits/internal_test.csv"))
    if test_csv.exists():
        test_logits, test_labels = collect_logits_from_csv(onnx_path, metadata, test_csv, batch_size)
        result["test_eval"] = {
            "split": str(test_csv),
            "num_samples": int(test_labels.numel()),
            "before": diagnostics(test_logits, test_labels, 1.0),
            "after": diagnostics(test_logits, test_labels, temperature),
        }

    return result
