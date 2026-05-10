"""Validation-only calibration utilities: Temperature Scaling, Vector Scaling, Dirichlet Calibration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from autolens_ai.training.dataset import AutoLensDataset
from autolens_ai.training.preprocessing import PreprocessConfig

CALIBRATION_METHODS = Literal["temperature", "vector_scaling", "dirichlet"]


# ──────────────────────────────────────────────────────────────
#  Utility functions (unchanged)
# ──────────────────────────────────────────────────────────────

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


def compute_classwise_ece(
    probs: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    bins: int = 15,
) -> float:
    """Classwise ECE: one-vs-rest calibration error averaged across classes.

    For each class k, compare p(y=k) with the binary event labels==k over all
    samples. This catches asymmetric per-class calibration errors that top-label
    ECE can hide.
    """
    total_ece = 0.0
    for k in range(num_classes):
        class_confidences = probs[:, k]
        class_accuracies = labels.eq(k).float()
        class_ece = torch.zeros((), dtype=torch.float32)
        boundaries = torch.linspace(0, 1, bins + 1)
        for lower, upper in zip(boundaries[:-1], boundaries[1:], strict=True):
            in_bin = class_confidences.gt(lower) & class_confidences.le(upper)
            prop = in_bin.float().mean()
            if prop.item() > 0:
                class_ece += (
                    torch.abs(
                        class_confidences[in_bin].mean()
                        - class_accuracies[in_bin].mean()
                    )
                    * prop
                )
        total_ece += float(class_ece.item())
    return total_ece / num_classes


def diagnostics(logits: torch.Tensor, labels: torch.Tensor, temperature: float) -> dict[str, float]:
    """Return confidence diagnostics at a given temperature."""
    scaled = logits / temperature
    probs = torch.softmax(scaled, dim=1)
    preds = probs.argmax(dim=1)
    num_classes = logits.shape[1]
    return {
        "nll": float(F.cross_entropy(scaled, labels).item()),
        "ece": ece_score(probs, labels),
        "cw_ece": compute_classwise_ece(probs, labels, num_classes),
        "accuracy": float(preds.eq(labels).float().mean().item()),
        "mean_confidence": float(probs.max(dim=1).values.mean().item()),
    }


def compute_classification_report(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_names: list[str],
) -> dict[str, dict[str, float]]:
    """Per-class precision, recall, F1 support dict."""
    preds = logits.argmax(dim=1)
    num_classes = logits.shape[1]
    report: dict[str, dict[str, float]] = {}
    for k in range(num_classes):
        name = class_names[k] if k < len(class_names) else str(k)
        tp = int(((preds == k) & (labels == k)).sum())
        fp = int(((preds == k) & (labels != k)).sum())
        fn = int(((preds != k) & (labels == k)).sum())
        support = int((labels == k).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        report[name] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": support,
        }
    return report


def compute_confusion_matrix(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> list[list[int]]:
    """Compute raw confusion matrix as nested list."""
    preds = logits.argmax(dim=1)
    matrix = [[0] * num_classes for _ in range(num_classes)]
    for p, t in zip(preds.tolist(), labels.tolist()):
        matrix[t][p] += 1
    return matrix


def compute_normalized_confusion_matrix(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> list[list[float]]:
    """Compute row-normalized confusion matrix."""
    raw = compute_confusion_matrix(logits, labels, num_classes)
    normalized = []
    for row in raw:
        total = sum(row)
        normalized.append([round(v / total, 4) if total > 0 else 0.0 for v in row])
    return normalized


def macro_metrics(report: dict[str, dict[str, float]]) -> dict[str, float]:
    """Compute macro-averaged metrics from per-class report."""
    n = len(report)
    if n == 0:
        return {"precision_macro": 0.0, "recall_macro": 0.0, "f1_macro": 0.0}
    prec = sum(r["precision"] for r in report.values()) / n
    rec = sum(r["recall"] for r in report.values()) / n
    f1 = sum(r["f1"] for r in report.values()) / n
    return {
        "precision_macro": round(prec, 6),
        "recall_macro": round(rec, 6),
        "f1_macro": round(f1, 6),
    }


def weighted_metrics(report: dict[str, dict[str, float]]) -> dict[str, float]:
    """Compute support-weighted averaged metrics from per-class report."""
    total_support = sum(r["support"] for r in report.values())
    if total_support == 0:
        return {"precision_weighted": 0.0, "recall_weighted": 0.0, "f1_weighted": 0.0}
    prec = sum(r["precision"] * r["support"] for r in report.values()) / total_support
    rec = sum(r["recall"] * r["support"] for r in report.values()) / total_support
    f1 = sum(r["f1"] * r["support"] for r in report.values()) / total_support
    return {
        "precision_weighted": round(prec, 6),
        "recall_weighted": round(rec, 6),
        "f1_weighted": round(f1, 6),
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


# ──────────────────────────────────────────────────────────────
#  Temperature Scaling
# ──────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────
#  Vector Scaling
# ──────────────────────────────────────────────────────────────

class VectorScaling(nn.Module):
    """Per-class affine logit transformation.

    For K classes, learns K weights and K biases:
        scaled_logits[i] = w[i] * logits[i] + b[i]

    Parameters: 2K = 16 for 8-class problem.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_classes))
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits * self.weight + self.bias


def fit_vector_scaling(
    logits: torch.Tensor,
    labels: torch.Tensor,
    l2_lambda: float = 0.01,
    max_iter: int = 200,
    lr: float = 0.05,
) -> tuple[VectorScaling, dict[str, Any]]:
    """Fit Vector Scaling with L2 regularization toward identity."""
    num_classes = logits.shape[1]
    model = VectorScaling(num_classes)
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=lr,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
    )

    identity_weight = torch.ones(num_classes)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        scaled = model(logits)
        ce = F.cross_entropy(scaled, labels)
        reg = l2_lambda * ((model.weight - identity_weight) ** 2).sum()
        reg += l2_lambda * (model.bias ** 2).sum()
        loss = ce + reg
        loss.backward()
        return loss

    optimizer.step(closure)

    info = {
        "weight": model.weight.detach().cpu().numpy().tolist(),
        "bias": model.bias.detach().cpu().numpy().tolist(),
        "l2_lambda": l2_lambda,
    }
    return model, info


# ──────────────────────────────────────────────────────────────
#  Dirichlet Calibration (with ODIR regularization)
# ──────────────────────────────────────────────────────────────

class DirichletCalibration(nn.Module):
    """Dirichlet Calibration with optional ODIR regularization.

    Operates on log-probabilities:
        calibrated = softmax( W * log(p) + b )

    For K classes: W ∈ R^{K×K}, b ∈ R^K  →  K² + K parameters.
    ODIR penalises off-diagonal elements of W and bias b.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(num_classes, num_classes, bias=True)
        # Initialize close to identity
        with torch.no_grad():
            self.fc.weight.copy_(torch.eye(num_classes).float())
            self.fc.bias.zero_()

    def forward(self, probs: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
        log_probs = torch.log(probs + eps)
        return torch.softmax(self.fc(log_probs), dim=1)


def fit_dirichlet(
    probs: torch.Tensor,
    labels: torch.Tensor,
    odir_lambda: float = 0.01,
    max_iter: int = 200,
    lr: float = 0.01,
) -> tuple[DirichletCalibration, dict[str, Any]]:
    """Fit Dirichlet Calibration with ODIR regularization."""
    num_classes = probs.shape[1]
    model = DirichletCalibration(num_classes)
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=lr,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        log_probs = torch.log(probs + 1e-9)
        calibrated_logits = model.fc(log_probs)
        ce = F.cross_entropy(calibrated_logits, labels)
        # ODIR: penalise off-diagonal elements and bias
        off_diag_mask = ~torch.eye(num_classes, dtype=torch.bool)
        reg = odir_lambda * (model.fc.weight[off_diag_mask] ** 2).sum()
        reg += odir_lambda * (model.fc.bias ** 2).sum()
        loss = ce + reg
        loss.backward()
        return loss

    optimizer.step(closure)

    info = {
        "W": model.fc.weight.detach().cpu().numpy().tolist(),
        "b": model.fc.bias.detach().cpu().numpy().tolist(),
        "odir_lambda": odir_lambda,
    }
    return model, info


# ──────────────────────────────────────────────────────────────
#  Main fitting wrappers
# ──────────────────────────────────────────────────────────────

def fit_temperature_scaling(
    onnx_path: Path,
    metadata: dict[str, Any],
    batch_size: int = 64,
    limit: int | None = None,
) -> dict[str, Any]:
    """Fit validation-only temperature scaling and return metadata."""
    logits, labels = collect_logits(onnx_path, metadata, batch_size, limit)
    temperature = fit_temperature(logits, labels)
    before = diagnostics(logits, labels, 1.0)
    after = diagnostics(logits, labels, temperature)

    result: dict[str, Any] = {
        "method": "temperature_scaling",
        "temperature": temperature,
        "vector_params": None,
        "dirichlet_params": None,
        "fit_split": str(metadata["data"]["val_csv"]),
        "num_samples": int(labels.numel()),
        "before": before,
        "after": after,
    }

    # Internal test evaluation (read-only)
    test_csv = Path(metadata["data"].get("test_csv", "artifacts/dataset/splits/internal_test.csv"))
    if test_csv.exists():
        test_logits, test_labels = collect_logits_from_csv(
            onnx_path, metadata, test_csv, batch_size
        )
        result["test_eval"] = {
            "split": str(test_csv),
            "num_samples": int(test_labels.numel()),
            "before": diagnostics(test_logits, test_labels, 1.0),
            "after": diagnostics(test_logits, test_labels, temperature),
        }
        # Add per-class metrics
        class_names = metadata.get("class_names", [str(i) for i in range(logits.shape[1])])
        result["test_eval"]["class_report_before"] = compute_classification_report(
            test_logits, test_labels, class_names
        )
        result["test_eval"]["class_report_after"] = compute_classification_report(
            test_logits / temperature, test_labels, class_names
        )
        result["test_eval"]["confusion_matrix_before"] = compute_normalized_confusion_matrix(
            test_logits, test_labels, logits.shape[1]
        )
        result["test_eval"]["confusion_matrix_after"] = compute_normalized_confusion_matrix(
            test_logits / temperature, test_labels, logits.shape[1]
        )
        result["test_eval"].update(macro_metrics(result["test_eval"]["class_report_after"]))
        result["test_eval"].update(weighted_metrics(result["test_eval"]["class_report_after"]))

    return result


def fit_vector_scaling_method(
    onnx_path: Path,
    metadata: dict[str, Any],
    batch_size: int = 64,
    limit: int | None = None,
    l2_lambda: float = 0.01,
) -> dict[str, Any]:
    """Fit validation-only Vector Scaling and return metadata."""
    logits, labels = collect_logits(onnx_path, metadata, batch_size, limit)
    model, info = fit_vector_scaling(logits, labels, l2_lambda=l2_lambda)

    weights = torch.tensor(info["weight"], dtype=torch.float32)
    biases = torch.tensor(info["bias"], dtype=torch.float32)
    scaled_logits = logits * weights + biases

    before = diagnostics(logits, labels, 1.0)
    after = diagnostics(scaled_logits, labels, 1.0)

    result: dict[str, Any] = {
        "method": "vector_scaling",
        "temperature": None,
        "vector_params": info,
        "dirichlet_params": None,
        "fit_split": str(metadata["data"]["val_csv"]),
        "num_samples": int(labels.numel()),
        "before": before,
        "after": after,
    }

    # Internal test evaluation
    test_csv = Path(metadata["data"].get("test_csv", "artifacts/dataset/splits/internal_test.csv"))
    if test_csv.exists():
        test_logits, test_labels = collect_logits_from_csv(
            onnx_path, metadata, test_csv, batch_size
        )
        test_scaled = test_logits * weights + biases
        result["test_eval"] = {
            "split": str(test_csv),
            "num_samples": int(test_labels.numel()),
            "before": diagnostics(test_logits, test_labels, 1.0),
            "after": diagnostics(test_scaled, test_labels, 1.0),
        }
        class_names = metadata.get("class_names", [str(i) for i in range(logits.shape[1])])
        result["test_eval"]["class_report_before"] = compute_classification_report(
            test_logits, test_labels, class_names
        )
        result["test_eval"]["class_report_after"] = compute_classification_report(
            test_scaled, test_labels, class_names
        )
        result["test_eval"]["confusion_matrix_before"] = compute_normalized_confusion_matrix(
            test_logits, test_labels, logits.shape[1]
        )
        result["test_eval"]["confusion_matrix_after"] = compute_normalized_confusion_matrix(
            test_scaled, test_labels, logits.shape[1]
        )
        result["test_eval"].update(macro_metrics(result["test_eval"]["class_report_after"]))
        result["test_eval"].update(weighted_metrics(result["test_eval"]["class_report_after"]))

    return result


def fit_dirichlet_method(
    onnx_path: Path,
    metadata: dict[str, Any],
    batch_size: int = 64,
    limit: int | None = None,
    odir_lambda: float = 0.01,
) -> dict[str, Any]:
    """Fit validation-only Dirichlet Calibration and return metadata."""
    logits, labels = collect_logits(onnx_path, metadata, batch_size, limit)
    probs = torch.softmax(logits, dim=1)
    model, info = fit_dirichlet(probs, labels, odir_lambda=odir_lambda)

    # Apply fitted calibration
    calibrated_probs = model(probs)
    calibrated_logits = torch.log(calibrated_probs + 1e-9)

    before = diagnostics(logits, labels, 1.0)
    # For Dirichlet, compute diagnostics from calibrated probs directly
    after_probs = calibrated_probs
    after_preds = after_probs.argmax(dim=1)
    after_nll = float(F.cross_entropy(calibrated_logits, labels).item())
    after_ece = ece_score(after_probs, labels)
    after_cw_ece = compute_classwise_ece(after_probs, labels, logits.shape[1])
    after_acc = float(after_preds.eq(labels).float().mean().item())
    after_mc = float(after_probs.max(dim=1).values.mean().item())
    after = {
        "nll": after_nll,
        "ece": after_ece,
        "cw_ece": after_cw_ece,
        "accuracy": after_acc,
        "mean_confidence": after_mc,
    }

    result: dict[str, Any] = {
        "method": "dirichlet",
        "temperature": None,
        "vector_params": None,
        "dirichlet_params": info,
        "fit_split": str(metadata["data"]["val_csv"]),
        "num_samples": int(labels.numel()),
        "before": before,
        "after": after,
    }

    # Internal test evaluation
    test_csv = Path(metadata["data"].get("test_csv", "artifacts/dataset/splits/internal_test.csv"))
    if test_csv.exists():
        test_logits, test_labels = collect_logits_from_csv(
            onnx_path, metadata, test_csv, batch_size
        )
        test_probs = torch.softmax(test_logits, dim=1)
        test_calibrated = model(test_probs)
        test_calib_logits = torch.log(test_calibrated + 1e-9)

        result["test_eval"] = {
            "split": str(test_csv),
            "num_samples": int(test_labels.numel()),
            "before": diagnostics(test_logits, test_labels, 1.0),
            "after": diagnostics(test_calib_logits, test_labels, 1.0),
        }
        class_names = metadata.get("class_names", [str(i) for i in range(logits.shape[1])])
        result["test_eval"]["class_report_before"] = compute_classification_report(
            test_logits, test_labels, class_names
        )
        result["test_eval"]["class_report_after"] = compute_classification_report(
            test_calib_logits, test_labels, class_names
        )
        result["test_eval"]["confusion_matrix_before"] = compute_normalized_confusion_matrix(
            test_logits, test_labels, logits.shape[1]
        )
        result["test_eval"]["confusion_matrix_after"] = compute_normalized_confusion_matrix(
            test_calib_logits, test_labels, logits.shape[1]
        )
        result["test_eval"].update(macro_metrics(result["test_eval"]["class_report_after"]))
        result["test_eval"].update(weighted_metrics(result["test_eval"]["class_report_after"]))

    return result
