"""Temperature scaling calibration for multiclass image classifiers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
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


def fit_temperature_scaling(
    onnx_path: Path,
    metadata: dict[str, Any],
    batch_size: int = 64,
    limit: int | None = None,
) -> dict[str, Any]:
    """Fit validation-only temperature scaling and evaluate internal test when present."""
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
        class_names = metadata.get("class_names", [str(i) for i in range(logits.shape[1])])
        class_report_after = compute_classification_report(
            test_logits / temperature, test_labels, class_names
        )
        result["test_eval"]["class_report_before"] = compute_classification_report(
            test_logits, test_labels, class_names
        )
        result["test_eval"]["class_report_after"] = class_report_after
        result["test_eval"]["confusion_matrix_before"] = compute_normalized_confusion_matrix(
            test_logits, test_labels, logits.shape[1]
        )
        result["test_eval"]["confusion_matrix_after"] = compute_normalized_confusion_matrix(
            test_logits / temperature, test_labels, logits.shape[1]
        )
        result["test_eval"].update(macro_metrics(class_report_after))
        result["test_eval"].update(weighted_metrics(class_report_after))

    return result
