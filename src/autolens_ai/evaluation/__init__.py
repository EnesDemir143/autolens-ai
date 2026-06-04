"""Evaluation helpers for AutoLens AI."""

from autolens_ai.evaluation.calibration import (
    CALIBRATION_METHODS,
    collect_logits,
    compute_classification_report,
    compute_confusion_matrix,
    compute_normalized_confusion_matrix,
    diagnostics,
    ece_score,
    expected_calibration_split,
    fit_dirichlet,
    fit_dirichlet_method,
    fit_temperature,
    fit_temperature_scaling,
    fit_vector_scaling,
    fit_vector_scaling_method,
    macro_metrics,
    weighted_metrics,
)

__all__ = [
    "CALIBRATION_METHODS",
    "collect_logits",
    "compute_classification_report",
    "compute_confusion_matrix",
    "compute_normalized_confusion_matrix",
    "diagnostics",
    "ece_score",
    "expected_calibration_split",
    "fit_dirichlet",
    "fit_dirichlet_method",
    "fit_temperature",
    "fit_temperature_scaling",
    "fit_vector_scaling",
    "fit_vector_scaling_method",
    "macro_metrics",
    "weighted_metrics",
]
