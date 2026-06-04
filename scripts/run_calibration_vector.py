#!/usr/bin/env python
"""Standalone Vector Scaling calibration script with L2 grid search.

Reads experiment config, finds exported ONNX artifact, fits vector scaling
for each L2 lambda on validation, evaluates all on internal test, picks best,
and saves results.

Usage:
    uv run python scripts/run_calibration_vector.py \
        --config configs/experiments/dinov3_safe_weighted_aug.yaml

Output:
    artifacts/export/<experiment_name>/calibration/vector_scaling_lambda<l2>/calibration.json
    artifacts/export/<experiment_name>/calibration/vector_scaling/best_calibration.json
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
from autolens_ai.evaluation.vector_scaling import fit_vector_scaling
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
    weights: torch.Tensor,
    biases: torch.Tensor,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate calibrated logits on internal test set."""
    test_scaled = test_logits * weights + biases
    before = diagnostics(test_logits, test_labels, 1.0)
    after = diagnostics(test_scaled, test_labels, 1.0)

    class_names = metadata.get("class_names", [str(i) for i in range(test_logits.shape[1])])

    class_report_before = compute_classification_report(test_logits, test_labels, class_names)
    class_report_after = compute_classification_report(test_scaled, test_labels, class_names)

    cm_before = compute_normalized_confusion_matrix(test_logits, test_labels, test_logits.shape[1])
    cm_after = compute_normalized_confusion_matrix(test_scaled, test_labels, test_logits.shape[1])

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
    parser.add_argument(
        "--l2-lambdas",
        type=float,
        nargs="+",
        default=None,
        help="L2 lambda values to grid search (default: from config or [0.01, 0.1, 1.0])",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    experiment_name = config["experiment_name"]
    export_dir, onnx_path, metadata_path = find_export_paths(config, args.output_base)
    metadata = load_artifact_metadata(metadata_path)

    # Get lambda list from config or default
    l2_lambdas = args.l2_lambdas
    if l2_lambdas is None:
        calib_cfg = config.get("calibration", {})
        methods_cfg = calib_cfg.get("methods", {})
        vector_cfg = methods_cfg.get("vector_scaling", {})
        l2_lambdas = vector_cfg.get("l2_lambdas", [0.01, 0.1, 1.0])

    print(f"\n{'='*60}")
    print(f"  Vector Scaling Grid Search | {experiment_name}")
    print(f"  L2 λ values: {l2_lambdas}")
    print(f"{'='*60}")
    print(f"  Export dir: {export_dir}")
    print(f"  ONNX:       {onnx_path}")

    # Collect validation logits once
    logits, labels = collect_logits(onnx_path, metadata, args.batch_size, limit=None)

    # Collect test logits once
    test_csv = Path(metadata["data"].get("test_csv", "artifacts/dataset/splits/internal_test.csv"))
    test_logits, test_labels = None, None
    if test_csv.exists():
        test_logits, test_labels = collect_logits_from_csv(onnx_path, metadata, test_csv, args.batch_size)

    all_results: list[dict[str, Any]] = []
    best_result: dict[str, Any] | None = None
    best_val_nll = float("inf")
    best_val_ece = float("inf")

    for l2 in l2_lambdas:
        print(f"\n  >>> Fitting L2 λ = {l2} ...")
        model, info = fit_vector_scaling(logits, labels, l2_lambda=l2)

        weights = torch.tensor(info["weight"], dtype=torch.float32)
        biases = torch.tensor(info["bias"], dtype=torch.float32)
        scaled_logits = logits * weights + biases

        before = diagnostics(logits, labels, 1.0)
        after = diagnostics(scaled_logits, labels, 1.0)

        result: dict[str, Any] = {
            "method": "vector_scaling",
            "experiment_name": experiment_name,
            "temperature": None,
            "vector_params": info,
            "dirichlet_params": None,
            "fit_split": str(metadata["data"]["val_csv"]),
            "num_samples": int(labels.numel()),
            "before": before,
            "after": after,
        }

        if test_logits is not None and test_labels is not None:
            result["test_eval"] = evaluate_on_test(test_logits, test_labels, weights, biases, metadata)
            f1 = result["test_eval"].get("f1_macro", 0.0)
            print(f"      Val NLL: {before['nll']:.4f} → {after['nll']:.4f}")
            print(f"      Val ECE: {before['ece']:.4f} → {after['ece']:.4f}")
            print(f"      Test F1-macro: {f1:.4f}")
        else:
            print("      (No test split found)")

        val_nll = result["after"]["nll"]
        val_ece = result["after"]["ece"]
        if (val_nll, val_ece) < (best_val_nll, best_val_ece):
            best_val_nll = val_nll
            best_val_ece = val_ece
            best_result = result.copy()

        # Save individual
        output_dir = export_dir / "calibration" / f"vector_scaling_lambda{l2}"
        output_dir.mkdir(parents=True, exist_ok=True)
        result["saved_at"] = datetime.now(UTC).isoformat()
        result["output_dir"] = str(output_dir)

        out_path = output_dir / "calibration.json"
        out_path.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"      Saved: {out_path}")
        all_results.append(result)

    # Save best
    if best_result is not None:
        best_dir = export_dir / "calibration" / "vector_scaling"
        best_dir.mkdir(parents=True, exist_ok=True)
        best_result["saved_at"] = datetime.now(UTC).isoformat()
        best_result["output_dir"] = str(best_dir)
        best_result["selection_metric"] = "validation_nll_then_ece"
        best_result["note"] = f"Best validation NLL/ECE among L2 λ = {l2_lambdas}; internal test metrics are report-only."
        best_path = best_dir / "best_calibration.json"
        best_path.write_text(json.dumps(best_result, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"\n  Best result saved: {best_path}")
        print(
            f"  Best L2 λ = {best_result['vector_params']['l2_lambda']} "
            f"| Val NLL = {best_val_nll:.4f} | Val ECE = {best_val_ece:.4f} "
            f"| Test F1-macro = {best_result.get('test_eval', {}).get('f1_macro', 0):.4f}"
        )

    # Grid summary
    summary = {
        "experiment_name": experiment_name,
        "method": "vector_scaling",
        "l2_lambdas": l2_lambdas,
        "results": [
            {
                "l2_lambda": r["vector_params"]["l2_lambda"],
                "val_nll_after": r["after"]["nll"],
                "val_ece_after": r["after"]["ece"],
                "test_f1_macro": r.get("test_eval", {}).get("f1_macro", None),
            }
            for r in all_results
        ],
    }
    summary_path = export_dir / "calibration" / "vector_scaling" / "grid_search_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"  Grid summary: {summary_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
