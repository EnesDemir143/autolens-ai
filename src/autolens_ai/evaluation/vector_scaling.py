"""Vector scaling calibration with per-class affine logit transforms."""

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
    compute_normalized_confusion_matrix,
    diagnostics,
    macro_metrics,
    weighted_metrics,
)


class VectorScaling(nn.Module):
    """Per-class affine logit transformation: scaled_logits[k] = w[k] * z[k] + b[k]."""

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
    """Fit vector scaling with L2 regularization toward identity."""
    num_classes = logits.shape[1]
    model = VectorScaling(num_classes)
    optimizer = torch.optim.LBFGS(
        model.parameters(), lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe"
    )
    identity_weight = torch.ones(num_classes)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        scaled = model(logits)
        ce = F.cross_entropy(scaled, labels)
        reg = l2_lambda * ((model.weight - identity_weight) ** 2).sum()
        reg += l2_lambda * (model.bias**2).sum()
        loss = ce + reg
        loss.backward()
        return loss

    optimizer.step(closure)
    return model, {
        "weight": model.weight.detach().cpu().numpy().tolist(),
        "bias": model.bias.detach().cpu().numpy().tolist(),
        "l2_lambda": l2_lambda,
    }


def fit_vector_scaling_method(
    onnx_path: Path,
    metadata: dict[str, Any],
    batch_size: int = 64,
    limit: int | None = None,
    l2_lambda: float = 0.01,
) -> dict[str, Any]:
    """Fit validation-only vector scaling and evaluate internal test when present."""
    logits, labels = collect_logits(onnx_path, metadata, batch_size, limit)
    _, info = fit_vector_scaling(logits, labels, l2_lambda=l2_lambda)

    weights = torch.tensor(info["weight"], dtype=torch.float32)
    biases = torch.tensor(info["bias"], dtype=torch.float32)
    scaled_logits = logits * weights + biases

    result: dict[str, Any] = {
        "method": "vector_scaling",
        "temperature": None,
        "vector_params": info,
        "dirichlet_params": None,
        "fit_split": str(metadata["data"]["val_csv"]),
        "num_samples": int(labels.numel()),
        "before": diagnostics(logits, labels, 1.0),
        "after": diagnostics(scaled_logits, labels, 1.0),
    }

    test_csv = Path(metadata["data"].get("test_csv", "artifacts/dataset/splits/internal_test.csv"))
    if test_csv.exists():
        test_logits, test_labels = collect_logits_from_csv(
            onnx_path, metadata, test_csv, batch_size
        )
        test_scaled = test_logits * weights + biases
        class_names = metadata.get("class_names", [str(i) for i in range(logits.shape[1])])
        class_report_after = compute_classification_report(test_scaled, test_labels, class_names)
        result["test_eval"] = {
            "split": str(test_csv),
            "num_samples": int(test_labels.numel()),
            "before": diagnostics(test_logits, test_labels, 1.0),
            "after": diagnostics(test_scaled, test_labels, 1.0),
            "class_report_before": compute_classification_report(
                test_logits, test_labels, class_names
            ),
            "class_report_after": class_report_after,
            "confusion_matrix_before": compute_normalized_confusion_matrix(
                test_logits, test_labels, logits.shape[1]
            ),
            "confusion_matrix_after": compute_normalized_confusion_matrix(
                test_scaled, test_labels, logits.shape[1]
            ),
        }
        result["test_eval"].update(macro_metrics(class_report_after))
        result["test_eval"].update(weighted_metrics(class_report_after))

    return result
