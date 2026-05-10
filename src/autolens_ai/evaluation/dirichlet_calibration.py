"""Dirichlet calibration with ODIR-style regularization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from autolens_ai.evaluation.calibration import (
    collect_logits,
    collect_logits_from_csv,
    compute_classification_report,
    compute_classwise_ece,
    compute_normalized_confusion_matrix,
    diagnostics,
    ece_score,
    macro_metrics,
    weighted_metrics,
)


class DirichletCalibration(nn.Module):
    """Dirichlet calibration over log probabilities: softmax(W log(p) + b)."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(num_classes, num_classes, bias=True)
        with torch.no_grad():
            self.fc.weight.copy_(torch.eye(num_classes).float())
            self.fc.bias.zero_()

    def forward(self, probs: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
        return torch.softmax(self.fc(torch.log(probs + eps)), dim=1)


def fit_dirichlet(
    probs: torch.Tensor,
    labels: torch.Tensor,
    odir_lambda: float = 0.01,
    max_iter: int = 200,
    lr: float = 0.01,
) -> tuple[DirichletCalibration, dict[str, Any]]:
    """Fit Dirichlet calibration with off-diagonal and bias regularization."""
    num_classes = probs.shape[1]
    model = DirichletCalibration(num_classes)
    optimizer = torch.optim.LBFGS(
        model.parameters(), lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        log_probs = torch.log(probs + 1e-9)
        calibrated_logits = model.fc(log_probs)
        ce = F.cross_entropy(calibrated_logits, labels)
        off_diag_mask = ~torch.eye(num_classes, dtype=torch.bool)
        reg = odir_lambda * (model.fc.weight[off_diag_mask] ** 2).sum()
        reg += odir_lambda * (model.fc.bias**2).sum()
        loss = ce + reg
        loss.backward()
        return loss

    optimizer.step(closure)
    return model, {
        "W": model.fc.weight.detach().cpu().numpy().tolist(),
        "b": model.fc.bias.detach().cpu().numpy().tolist(),
        "odir_lambda": odir_lambda,
    }


def _diagnostics_from_probs(probs: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    logits = torch.log(probs + 1e-9)
    preds = probs.argmax(dim=1)
    return {
        "nll": float(F.cross_entropy(logits, labels).item()),
        "ece": ece_score(probs, labels),
        "cw_ece": compute_classwise_ece(probs, labels, probs.shape[1]),
        "accuracy": float(preds.eq(labels).float().mean().item()),
        "mean_confidence": float(probs.max(dim=1).values.mean().item()),
    }


def fit_dirichlet_method(
    onnx_path: Path,
    metadata: dict[str, Any],
    batch_size: int = 64,
    limit: int | None = None,
    odir_lambda: float = 0.01,
) -> dict[str, Any]:
    """Fit validation-only Dirichlet calibration and evaluate internal test when present."""
    logits, labels = collect_logits(onnx_path, metadata, batch_size, limit)
    probs = torch.softmax(logits, dim=1)
    model, info = fit_dirichlet(probs, labels, odir_lambda=odir_lambda)

    calibrated_probs = model(probs)
    result: dict[str, Any] = {
        "method": "dirichlet",
        "temperature": None,
        "vector_params": None,
        "dirichlet_params": info,
        "fit_split": str(metadata["data"]["val_csv"]),
        "num_samples": int(labels.numel()),
        "before": diagnostics(logits, labels, 1.0),
        "after": _diagnostics_from_probs(calibrated_probs, labels),
    }

    test_csv = Path(metadata["data"].get("test_csv", "artifacts/dataset/splits/internal_test.csv"))
    if test_csv.exists():
        test_logits, test_labels = collect_logits_from_csv(
            onnx_path, metadata, test_csv, batch_size
        )
        test_probs = torch.softmax(test_logits, dim=1)
        test_calibrated = model(test_probs)
        test_calib_logits = torch.log(test_calibrated + 1e-9)
        class_names = metadata.get("class_names", [str(i) for i in range(logits.shape[1])])
        class_report_after = compute_classification_report(
            test_calib_logits, test_labels, class_names
        )
        result["test_eval"] = {
            "split": str(test_csv),
            "num_samples": int(test_labels.numel()),
            "before": diagnostics(test_logits, test_labels, 1.0),
            "after": _diagnostics_from_probs(test_calibrated, test_labels),
            "class_report_before": compute_classification_report(
                test_logits, test_labels, class_names
            ),
            "class_report_after": class_report_after,
            "confusion_matrix_before": compute_normalized_confusion_matrix(
                test_logits, test_labels, logits.shape[1]
            ),
            "confusion_matrix_after": compute_normalized_confusion_matrix(
                test_calib_logits, test_labels, logits.shape[1]
            ),
        }
        result["test_eval"].update(macro_metrics(class_report_after))
        result["test_eval"].update(weighted_metrics(class_report_after))

    return result
