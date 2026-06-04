#!/usr/bin/env python
"""Compare calibration methods on internal test set.

Loads all calibration results from artifacts/calibration/<experiment>/ folders
and produces a side-by-side comparison table with all metrics.

Usage:
    # Compare all methods for a specific experiment
    uv run python scripts/evaluate_calibration.py \\
        --experiment dinov3_safe_weighted_aug \\
        --internal-test artifacts/dataset/splits/internal_test.csv

    # Compare with custom ONNX path (for generating test logits)
    uv run python scripts/evaluate_calibration.py \\
        --experiment dinov3_safe_weighted_aug \\
        --onnx artifacts/export/dinov3_safe_weighted/model.onnx \\
        --metadata artifacts/export/dinov3_safe_weighted/metadata.json \\
        --internal-test artifacts/dataset/splits/internal_test.csv

    # Full sweep: run calibration + evaluation
    uv run python scripts/evaluate_calibration.py \\
        --full-sweep \\
        --onnx artifacts/export/dinov3_safe_weighted/model.onnx \\
        --metadata artifacts/export/dinov3_safe_weighted/metadata.json \\
        --internal-test artifacts/dataset/splits/internal_test.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any



def load_calibration_result(calibration_json: Path) -> dict[str, Any]:
    """Load a calibration.json file."""
    with calibration_json.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_metric(value: float | None, decimals: int = 4) -> str:
    """Format a metric value for table display."""
    if value is None:
        return "    N/A   "
    return f"{value:.{decimals}f}"


def print_comparison_table(results: list[dict[str, Any]]) -> None:
    """Print a formatted comparison table."""
    # Headers
    sep = "+" + "-" * 20 + "+" + "-" * 8 + "+" + "-" * 9 * 8 + "+"
    header = (
        f"| {'Method':<20} | {'Lambda':<8} | "
        f"{'NLL':>8} | {'ECE':>8} | {'cwECE':>8} | {'Acc':>8} | "
        f"{'F1-macro':>8} | {'F1-wt':>8} | {'Prec-m':>8} | {'Rec-m':>8} |"
    )

    print(sep)
    print(header)
    print(sep)

    for r in results:
        method = r["method"].upper()
        lam = r.get("lambda_label", "—")
        te = r.get("test_eval", {})
        after = r.get("after", {})

        # Prefer test_eval values, fall back to before/after
        nll = te.get("after", {}).get("nll", after.get("nll"))
        ece = te.get("after", {}).get("ece", after.get("ece"))
        cwece = te.get("after", {}).get("cw_ece", after.get("cw_ece"))
        acc = te.get("after", {}).get("accuracy", after.get("accuracy"))

        row = (
            f"| {method:<20} | {lam:<8} | "
            f"{format_metric(nll)} | {format_metric(ece)} | {format_metric(cwece)} | "
            f"{format_metric(acc)} | "
            f"{format_metric(te.get('f1_macro'))} | "
            f"{format_metric(te.get('f1_weighted'))} | "
            f"{format_metric(te.get('precision_macro'))} | "
            f"{format_metric(te.get('recall_macro'))} |"
        )
        print(row)

    print(sep)


def print_classwise_table(result: dict[str, Any], label: str) -> None:
    """Print per-class metrics for a single result."""
    te = result.get("test_eval", {})
    class_report = te.get("class_report_after", {})
    if not class_report:
        print(f"  No class-level report for {label}")
        return

    print(f"\n  Per-class metrics ({label}):")
    print(f"  {'Class':<22} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for cls_name, metrics in class_report.items():
        print(f"  {cls_name:<22} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
              f"{metrics['f1']:>10.4f} {metrics['support']:>10}")


def print_confusion_diff(before_cm: list, after_cm: list, class_names: list[str]) -> None:
    """Print confusion matrix improvement highlights."""
    print("\n  Biggest confusion matrix improvements (error reduction):")
    improvements = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i == j:
                continue
            before_val = before_cm[i][j] if i < len(before_cm) and j < len(before_cm[i]) else 0
            after_val = after_cm[i][j] if i < len(after_cm) and j < len(after_cm[i]) else 0
            diff = before_val - after_val
            improvements.append((diff, class_names[i], class_names[j], before_val, after_val))

    improvements.sort(reverse=True)
    for diff, true_cls, pred_cls, before_val, after_val in improvements[:5]:
        if diff > 0:
            print(f"    {true_cls:<20} → {pred_cls:<20}: {before_val:.4f} → {after_val:.4f} "
                  f"(Δ = {-diff:.4f})")


def collect_results_from_folder(base_dir: Path) -> list[dict[str, Any]]:
    """Collect all calibration.json files from subdirectories."""
    results = []
    if not base_dir.exists():
        print(f"WARNING: Base directory {base_dir} does not exist")
        return results

    for calib_json in sorted(base_dir.rglob("calibration.json")):
        # Skip grid_search_summary
        if "grid_search_summary" in str(calib_json):
            continue
        try:
            data = load_calibration_result(calib_json)
            # Build lambda label
            if data.get("vector_params"):
                lam_label = f"L2 λ={data['vector_params']['l2_lambda']}"
            elif data.get("dirichlet_params"):
                lam_label = f"ODIR λ={data['dirichlet_params']['odir_lambda']}"
            else:
                lam_label = f"T={data.get('temperature', 'N/A')}"
            data["lambda_label"] = lam_label
            data["_path"] = str(calib_json.relative_to(base_dir))
            results.append(data)
        except Exception as e:
            print(f"  ERROR loading {calib_json}: {e}")

    return results


def select_best(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Select best result by lowest validation NLL."""
    best = None
    min_nll = float("inf")
    for r in results:
        # Prefer validation NLL (from 'after')
        nll = r.get("after", {}).get("nll", float("inf"))
        if nll < min_nll:
            min_nll = nll
            best = r
    return best




def select_best_from_list(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Select best full-sweep result by lowest validation NLL."""
    if not results:
        raise ValueError("No calibration results available for selection")
    return min(results, key=lambda r: r.get("fit_after", {}).get("nll", float("inf")))

def run_comparison(args: argparse.Namespace) -> None:
    """Main comparison logic."""
    experiment = args.experiment

    if getattr(args, "results_dir", None):
        base_dir = Path(args.results_dir)
    # Determine base dir. New DINOv3 experiment targets save under:
    #   artifacts/export/<experiment>/calibration/
    # Keep the older artifacts/calibration/<experiment>/ layout as fallback.
    elif hasattr(args, "output_base") and args.output_base:
        output_base = Path(args.output_base)
        export_nested_dir = output_base / experiment / "calibration"
        legacy_dir = output_base / "calibration" / experiment
        base_dir = export_nested_dir if export_nested_dir.exists() else legacy_dir
    else:
        base_dir = Path("artifacts") / "calibration" / experiment

    # Collect existing results
    print(f"\n📊 Loading calibration results from: {base_dir}")
    results = collect_results_from_folder(base_dir)

    if not results:
        print(f"  No calibration results found at {base_dir}")
        return

    print(f"  Found {len(results)} calibration run(s)\n")

    # Print comparison
    print_comparison_table(results)

    # Print classwise for each method
    for r in results:
        label = f"{r['method']} ({r['lambda_label']})"
        print_classwise_table(r, label)

    # Print confusion matrix diffs
    for r in results:
        te = r.get("test_eval", {})
        cm_before = te.get("confusion_matrix_before")
        cm_after = te.get("confusion_matrix_after")
        if cm_before and cm_after and r.get("test_eval", {}).get("num_samples"):
            # Get class names from metadata if available
            class_names = []
            if hasattr(args, "metadata") and args.metadata:
                from autolens_ai.inference import load_artifact_metadata
                meta = load_artifact_metadata(args.metadata)
                class_names = meta.get("class_names", [])
            if not class_names:
                class_names = [f"Class {i}" for i in range(len(cm_before))]

            print(f"\n  Confusion matrix improvements for {r['method']} ({r['lambda_label']}):")
            print_confusion_diff(cm_before, cm_after, class_names)

    # Select best
    best = select_best(results)
    if best:
        print(f"\n{'='*60}")
        print(f"🏆 BEST METHOD: {best['method'].upper()} ({best['lambda_label']})")
        te = best.get("test_eval", {})
        print(f"   F1-macro:    {te.get('f1_macro', 'N/A'):.4f}")
        print(f"   F1-weighted: {te.get('f1_weighted', 'N/A'):.4f}")
        print(f"   Accuracy:    {te.get('after',{}).get('accuracy', 'N/A'):.4f}")
        print(f"   ECE:         {te.get('after',{}).get('ece', 'N/A'):.4f}")
        print(f"   Saved at:    {best.get('_path', 'N/A')}")
        print(f"{'='*60}")


def run_full_sweep(args: argparse.Namespace) -> None:
    """Run calibration (grid search) then evaluate comparison."""
    from autolens_ai.evaluation.calibration import (
        fit_dirichlet_method,
        fit_temperature_scaling,
        fit_vector_scaling_method,
    )
    from autolens_ai.inference import load_artifact_metadata

    metadata = load_artifact_metadata(args.metadata)
    onnx_path = args.onnx
    internal_test = Path(args.internal_test)
    output_base = Path(args.output_base) if hasattr(args, "output_base") and args.output_base else Path("artifacts")

    # Validate calibration split
    val_csv = Path(metadata["data"]["val_csv"])
    from autolens_ai.evaluation.calibration import expected_calibration_split
    expected_calibration_split(val_csv)

    experiment_name = metadata["experiment_name"]
    base_dir = output_base / "calibration" / experiment_name

    l2_lambdas = [0.001, 0.01, 0.1, 1.0]
    odir_lambdas = [0.001, 0.01, 0.1, 1.0]

    from autolens_ai.evaluation.calibration import collect_logits_from_csv

    # Collect internal test logits (read-only, used only for evaluation)
    print("📡 Collecting internal test logits...")
    test_logits, test_labels = collect_logits_from_csv(onnx_path, metadata, internal_test, batch_size=64)

    results = []

    # --- Temperature Scaling ---
    print("\n>>> Temperature Scaling...")
    temp = fit_temperature_scaling(onnx_path, metadata, batch_size=64)
    # Re-evaluate with our richer metrics
    temp_test = _evaluate_on_test(temp, test_logits, test_labels, metadata, "temperature", "auto", base_dir)
    results.append(temp_test)

    # --- Vector Scaling ---
    for l2 in l2_lambdas:
        print(f"\n>>> Vector Scaling (L2 λ={l2})...")
        vs_result = fit_vector_scaling_method(onnx_path, metadata, batch_size=64, l2_lambda=l2)
        vs_test = _evaluate_on_test(vs_result, test_logits, test_labels, metadata, "vector_scaling", l2, base_dir)
        results.append(vs_test)

    # --- Dirichlet Calibration ---
    for odir in odir_lambdas:
        print(f"\n>>> Dirichlet Calibration (ODIR λ={odir})...")
        dir_result = fit_dirichlet_method(onnx_path, metadata, batch_size=64, odir_lambda=odir)
        dir_test = _evaluate_on_test(dir_result, test_logits, test_labels, metadata, "dirichlet", odir, base_dir)
        results.append(dir_test)

    # Store results and run comparison
    all_results_dir = base_dir / "_all_results"
    all_results_dir.mkdir(parents=True, exist_ok=True)

    for r in results:
        out = all_results_dir / f"{r['method']}_{r.get('lambda_label', 'auto').replace(' ', '_').replace('=', '')}.json"
        with out.open("w") as f:
            json.dump(r, f, indent=2, default=str)

    # Print comparison
    print("\n" + "=" * 70)
    print("  COMPLETE COMPARISON — Internal Test Set")
    print("=" * 70)
    _print_results(results)

    # Best result
    best = select_best_from_list(results)
    val_nll = best.get("fit_after", {}).get("nll", float("inf"))
    print(f"\n🏆 BEST: {best['method'].upper()} ({best['lambda_label']}) — Val NLL: {val_nll:.4f} | Test F1-macro: {best['test_f1_macro']:.4f}")


def _evaluate_on_test(
    calib_result: dict,
    test_logits: Any,
    test_labels: Any,
    metadata: dict,
    method: str,
    lambda_val: str | float,
    base_dir: Path,
) -> dict:
    """Evaluate calibrated logits on internal test set with full metrics."""
    import torch
    from autolens_ai.evaluation.calibration import (
        diagnostics,
        compute_classification_report,
        compute_normalized_confusion_matrix,
        macro_metrics,
        weighted_metrics,
    )

    num_classes = test_logits.shape[1]

    if method == "temperature":
        T = calib_result["temperature"]
        scaled_logits = test_logits / T
        lambda_label = f"T={T:.6f}"
    elif method == "vector_scaling":
        params = calib_result["vector_params"]
        w = torch.tensor(params["weight"], dtype=torch.float32)
        b = torch.tensor(params["bias"], dtype=torch.float32)
        scaled_logits = test_logits * w + b
        lambda_label = f"L2 λ={lambda_val}"
    elif method == "dirichlet":
        params = calib_result["dirichlet_params"]
        W = torch.tensor(params["W"], dtype=torch.float32)
        b = torch.tensor(params["b"], dtype=torch.float32)
        test_probs = torch.softmax(test_logits, dim=1)
        test_log_probs = torch.log(test_probs + 1e-9)
        calib_probs = torch.softmax(W @ test_log_probs.T + b.unsqueeze(1), dim=0).T
        scaled_logits = torch.log(calib_probs + 1e-9)
        lambda_label = f"ODIR λ={lambda_val}"
    else:
        raise ValueError(method)

    before = diagnostics(test_logits, test_labels, 1.0)
    after = diagnostics(scaled_logits, test_labels, 1.0)

    class_names = metadata.get("class_names", [str(i) for i in range(num_classes)])

    # Per-class report and confusion matrices
    class_report_before = compute_classification_report(test_logits, test_labels, class_names)
    class_report_after = compute_classification_report(scaled_logits, test_labels, class_names)
    cm_before = compute_normalized_confusion_matrix(test_logits, test_labels, num_classes)
    cm_after = compute_normalized_confusion_matrix(scaled_logits, test_labels, num_classes)

    result = {
        "method": method,
        "lambda_label": lambda_label,
        "lambda_value": lambda_val,
        "fit_before": calib_result.get("before", {}),
        "fit_after": calib_result.get("after", {}),
        "test_eval": {
            "before": before,
            "after": after,
            "class_report_before": class_report_before,
            "class_report_after": class_report_after,
            "confusion_matrix_before": cm_before,
            "confusion_matrix_after": cm_after,
        },
    }
    result["test_eval"].update(macro_metrics(class_report_after))
    result["test_eval"].update(weighted_metrics(class_report_after))
    result["test_f1_macro"] = result["test_eval"].get("f1_macro", 0.0)

    # Save individual result
    method_dir = base_dir / method
    method_dir.mkdir(parents=True, exist_ok=True)
    out_json = method_dir / f"{lambda_label.replace(' ', '').replace('=', '')}.json"
    with out_json.open("w") as f:
        json.dump(result, f, indent=2, default=str)

    return result


def _print_results(results: list[dict]) -> None:
    sep = "+" + "-" * 22 + "+" + "-" * 14 + "+" + "-" * 9 * 8 + "+"
    header = (
        f"| {'Method':<22} | {'Lambda':<14} | "
        f"{'NLL':>8} | {'ECE':>8} | {'cwECE':>8} | {'Acc':>8} | "
        f"{'F1-macro':>10} | {'Prec-m':>8} | {'Rec-m':>8} |"
    )
    print(sep)
    print(header)
    print(sep)

    for r in sorted(results, key=lambda x: x.get("test_f1_macro", 0), reverse=True):
        method = r["method"].upper()
        lam = r["lambda_label"]
        te = r["test_eval"]
        after = te["after"]

        row = (
            f"| {method:<22} | {lam:<14} | "
            f"{after.get('nll', 999):>8.4f} "
            f"{after.get('ece', 999):>8.4f} "
            f"{after.get('cw_ece', 999):>8.4f} "
            f"{after.get('accuracy', 0):>8.4f} "
            f"{r.get('test_f1_macro', 0):>10.4f} "
            f"{r.get('precision_macro', 0):>8.4f} "
            f"{r.get('recall_macro', 0):>8.4f} |"
        )
        print(row)
    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=str, help="Experiment name for loading results")
    parser.add_argument("--onnx", type=Path, help="Path to ONNX model (for full sweep)")
    parser.add_argument("--metadata", type=Path, help="Path to metadata.json")
    parser.add_argument("--internal-test", type=Path, help="Path to internal_test.csv")
    parser.add_argument(
        "--full-sweep",
        action="store_true",
        help="Run full calibration + evaluation sweep (all methods + lambdas)",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=Path("artifacts"),
        help="Base output directory (default: artifacts/)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Explicit calibration results directory to compare",
    )

    args = parser.parse_args()

    if args.full_sweep:
        if not all([args.onnx, args.metadata, args.internal_test]):
            parser.error("--full-sweep requires --onnx, --metadata, and --internal-test")
        run_full_sweep(args)
    else:
        if not args.experiment:
            parser.error("For comparison, provide --experiment name (or use --full-sweep)")
        args.output_base = args.output_base  # ensure attribute exists
        run_comparison(args)


if __name__ == "__main__":
    main()
