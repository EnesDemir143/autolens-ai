#!/usr/bin/env python
"""Standalone Temperature Scaling calibration script for DINOv3 experiments.

Reads experiment config, finds exported ONNX artifact, fits temperature on
validation split, evaluates on internal test, and saves all metrics.

Usage:
    uv run python scripts/run_calibration_temperature.py \
        --config configs/experiments/dinov3_safe_weighted_aug.yaml

Output:
    artifacts/export/<experiment_name>/calibration/temperature/calibration.json
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from autolens_ai.evaluation.calibration import (
    collect_logits,
    collect_logits_from_csv,
    compute_classification_report,
    compute_normalized_confusion_matrix,
    diagnostics,
    macro_metrics,
    weighted_metrics,
)
from autolens_ai.evaluation.temperature_scaling import fit_temperature
from autolens_ai.inference import load_artifact_metadata


def load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML experiment config."""
    import yaml

    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def resolve_export_dir(config: dict[str, Any], output_base: Path) -> Path:
    """Resolve exported artifact directory for this calibration run."""
    calibration = config.get("calibration", {})
    configured = calibration.get("artifact_dir") or calibration.get("export_dir")
    if configured:
        return Path(configured)

    experiment_name = config["experiment_name"]
    direct = output_base / experiment_name
    if direct.exists():
        return direct

    # DINOv3 training experiment names include _aug, while submitted export
    # folders use stable *_latest names. Keep this fallback so Makefile targets
    # work with the current artifact layout without duplicating models.
    latest_name = experiment_name.replace("_aug", "_latest")
    return output_base / latest_name


def find_export_paths(config: dict[str, Any], output_base: Path) -> tuple[Path, Path, Path]:
    """Derive export dir, ONNX path, and metadata path from experiment config."""
    export_dir = resolve_export_dir(config, output_base)
    onnx_path = export_dir / "model.onnx"
    metadata_path = export_dir / "metadata.json"
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX not found: {onnx_path}. Export model first.")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}. Export model first.")
    return export_dir, onnx_path, metadata_path


def evaluate_on_test(
    test_logits: torch.Tensor,
    test_labels: torch.Tensor,
    temperature: float,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate calibrated logits on internal test set."""
    before = diagnostics(test_logits, test_labels, 1.0)
    after = diagnostics(test_logits, test_labels, temperature)

    class_names = metadata.get("class_names", [str(i) for i in range(test_logits.shape[1])])

    class_report_before = compute_classification_report(test_logits, test_labels, class_names)
    class_report_after = compute_classification_report(test_logits / temperature, test_labels, class_names)

    cm_before = compute_normalized_confusion_matrix(test_logits, test_labels, test_logits.shape[1])
    cm_after = compute_normalized_confusion_matrix(test_logits / temperature, test_labels, test_logits.shape[1])

    result = {
        "split": "internal_test",
        "num_samples": int(test_labels.numel()),
        "before": before,
        "after": after,
        "class_report_before": class_report_before,
        "class_report_after": class_report_after,
        "confusion_matrix_before": cm_before,
        "confusion_matrix_after": cm_after,
    }
    result.update(macro_metrics(class_report_after))
    result.update(weighted_metrics(class_report_after))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to experiment YAML config")
    parser.add_argument(
        "--output-base",
        type=Path,
        default=Path("artifacts/export"),
        help="Base output directory containing experiment export folders",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for ONNX inference")
    args = parser.parse_args()

    config = load_config(args.config)
    experiment_name = config["experiment_name"]
    export_dir, onnx_path, metadata_path = find_export_paths(config, args.output_base)
    metadata = load_artifact_metadata(metadata_path)

    print(f"\n{'='*60}")
    print(f"  Temperature Scaling | {experiment_name}")
    print(f"{'='*60}")
    print(f"  Export dir: {export_dir}")
    print(f"  ONNX:       {onnx_path}")

    # Fit on validation
    logits, labels = collect_logits(onnx_path, metadata, args.batch_size, limit=None)
    temperature = fit_temperature(logits, labels)
    print(f"  Fitted T = {temperature:.6f}")

    before = diagnostics(logits, labels, 1.0)
    after = diagnostics(logits, labels, temperature)

    result: dict[str, Any] = {
        "method": "temperature_scaling",
        "experiment_name": experiment_name,
        "temperature": temperature,
        "vector_params": None,
        "dirichlet_params": None,
        "fit_split": str(metadata["data"]["val_csv"]),
        "num_samples": int(labels.numel()),
        "before": before,
        "after": after,
    }

    # Evaluate on internal test
    test_csv = Path(metadata["data"].get("test_csv", "artifacts/dataset/splits/internal_test.csv"))
    if test_csv.exists():
        test_logits, test_labels = collect_logits_from_csv(onnx_path, metadata, test_csv, args.batch_size)
        result["test_eval"] = evaluate_on_test(test_logits, test_labels, temperature, metadata)
        print(f"  Test F1-macro:   {result['test_eval'].get('f1_macro', 0):.4f}")
        print(f"  Test NLL:        {result['test_eval']['before']['nll']:.4f} → {result['test_eval']['after']['nll']:.4f}")
        print(f"  Test ECE:        {result['test_eval']['before']['ece']:.4f} → {result['test_eval']['after']['ece']:.4f}")
    else:
        print(f"  Warning: test split not found at {test_csv}")

    # Save
    output_dir = export_dir / "calibration" / "temperature"
    output_dir.mkdir(parents=True, exist_ok=True)
    result["saved_at"] = datetime.now(UTC).isoformat()
    result["output_dir"] = str(output_dir)

    out_path = output_dir / "calibration.json"
    out_path.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")

    print(f"\n  Saved: {out_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
