#!/usr/bin/env python
"""Generalized calibration script supporting Temperature Scaling, Vector Scaling, and Dirichlet Calibration.

Each method saves its results to a separate folder:
    artifacts/calibration/<experiment_name>/<method>/calibration.json

Usage examples:
    # Temperature Scaling (default)
    uv run python scripts/calibrate_model.py \\
        --onnx artifacts/export/dinov3_safe_weighted/model.onnx \\
        --metadata artifacts/export/dinov3_safe_weighted/metadata.json \\
        --method temperature

    # Vector Scaling with L2 search
    uv run python scripts/calibrate_model.py \\
        --onnx artifacts/export/dinov3_safe_weighted/model.onnx \\
        --metadata artifacts/export/dinov3_safe_weighted/metadata.json \\
        --method vector_scaling --l2-lambda 0.01

    # Dirichlet Calibration with ODIR search
    uv run python scripts/calibrate_model.py \\
        --onnx artifacts/export/dinov3_safe_weighted/model.onnx \\
        --metadata artifacts/export/dinov3_safe_weighted/metadata.json \\
        --method dirichlet --odir-lambda 0.01

    # Grid search over all methods and lambdas
    uv run python scripts/calibrate_model.py \\
        --onnx artifacts/export/dinov3_safe_weighted/model.onnx \\
        --metadata artifacts/export/dinov3_safe_weighted/metadata.json \\
        --method all --grid-search
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


from autolens_ai.evaluation.calibration import CALIBRATION_METHODS
from autolens_ai.evaluation.dirichlet_calibration import fit_dirichlet_method
from autolens_ai.evaluation.temperature_scaling import fit_temperature_scaling
from autolens_ai.evaluation.vector_scaling import fit_vector_scaling_method
from autolens_ai.inference import load_artifact_metadata


def build_output_path(
    base_dir: Path,
    experiment_name: str,
    method: str,
    lambda_val: float | None,
) -> Path:
    """Build output folder path.

    artifacts/calibration/<experiment_name>/<method>[_lambda<X>]/
    """
    if lambda_val is not None:
        folder = f"{method}_lambda{lambda_val}"
    else:
        folder = method
    return base_dir / "calibration" / experiment_name / folder


def save_calibration_result(
    result: dict[str, Any],
    output_dir: Path,
    method: str,
    lambda_label: str,
) -> None:
    """Save calibration result JSON and print summary."""
    output_dir.mkdir(parents=True, exist_ok=True)

    result["saved_at"] = datetime.now(UTC).isoformat()
    result["output_dir"] = str(output_dir)

    out_path = output_dir / "calibration.json"
    out_path.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")

    # Print summary
    print(f"\n{'='*60}")
    print(f"  {method.upper()} | {lambda_label}")
    print(f"{'='*60}")
    print(f"  Output:     {out_path}")
    print(f"  Val NLL:    {result['before']['nll']:.4f} → {result['after']['nll']:.4f}")
    print(f"  Val ECE:    {result['before']['ece']:.4f} → {result['after']['ece']:.4f}")
    print(f"  Val cwECE:  {result['before'].get('cw_ece', 'N/A')} → {result['after'].get('cw_ece', 'N/A')}")
    print(f"  Val Acc:    {result['before']['accuracy']:.4f} → {result['after']['accuracy']:.4f}")

    if "test_eval" in result:
        te = result["test_eval"]
        print(f"\n  {'Internal Test:':<20}")
        print(f"    NLL:          {te['before']['nll']:.4f} → {te['after']['nll']:.4f}")
        print(f"    ECE:          {te['before']['ece']:.4f} → {te['after']['ece']:.4f}")
        print(f"    Accuracy:     {te['before']['accuracy']:.4f} → {te['after']['accuracy']:.4f}")
        if "f1_macro" in te:
            print(f"    F1-macro:     {te['f1_macro']:.4f}")
        if "f1_weighted" in te:
            print(f"    F1-weighted:  {te['f1_weighted']:.4f}")

    print(f"{'='*60}\n")


def run_single_method(
    onnx_path: Path,
    metadata: dict[str, Any],
    method: CALIBRATION_METHODS,
    output_base: Path,
    l2_lambda: float | None = None,
    odir_lambda: float | None = None,
    batch_size: int = 64,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run a single calibration method and save results."""
    experiment_name = metadata.get("experiment_name", "unknown")

    if method == "temperature":
        # Temperature has no lambda
        result = fit_temperature_scaling(onnx_path, metadata, batch_size, limit)
        output_dir = build_output_path(output_base, experiment_name, "temperature", None)
        save_calibration_result(result, output_dir, "temperature_scaling", "T=auto")
        return result

    elif method == "vector_scaling":
        assert l2_lambda is not None, "Vector Scaling requires --l2-lambda"
        result = fit_vector_scaling_method(onnx_path, metadata, batch_size, limit, l2_lambda=l2_lambda)
        output_dir = build_output_path(output_base, experiment_name, "vector_scaling", l2_lambda)
        save_calibration_result(result, output_dir, "vector_scaling", f"L2 λ={l2_lambda}")
        return result

    elif method == "dirichlet":
        assert odir_lambda is not None, "Dirichlet Calibration requires --odir-lambda"
        result = fit_dirichlet_method(onnx_path, metadata, batch_size, limit, odir_lambda=odir_lambda)
        output_dir = build_output_path(output_base, experiment_name, "dirichlet", odir_lambda)
        save_calibration_result(result, output_dir, "dirichlet", f"ODIR λ={odir_lambda}")
        return result

    else:
        raise ValueError(f"Unknown method: {method}")


def run_grid_search(
    onnx_path: Path,
    metadata: dict[str, Any],
    output_base: Path,
    batch_size: int = 64,
    limit: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Run grid search over all methods and lambda values."""
    l2_lambdas = [0.001, 0.01, 0.1, 1.0]
    odir_lambdas = [0.001, 0.01, 0.1, 1.0]
    results: dict[str, list[dict[str, Any]]] = {}

    # Temperature Scaling
    print("\n>>> Running Temperature Scaling...")
    temp_result = fit_temperature_scaling(onnx_path, metadata, batch_size, limit)
    output_dir = build_output_path(output_base, metadata["experiment_name"], "temperature", None)
    save_calibration_result(temp_result, output_dir, "temperature_scaling", "T=auto")
    results["temperature"] = [temp_result]

    # Search best Vector Scaling
    print("\n>>> Grid search: Vector Scaling (L2 λ ∈ {l2_lambdas})...")
    vector_results = []
    for l2 in l2_lambdas:
        r = fit_vector_scaling_method(onnx_path, metadata, batch_size, limit, l2_lambda=l2)
        output_dir = build_output_path(output_base, metadata["experiment_name"], "vector_scaling", l2)
        save_calibration_result(r, output_dir, "vector_scaling", f"L2 λ={l2}")
        vector_results.append(r)
    results["vector_scaling"] = vector_results

    # Search best Dirichlet
    print("\n>>> Grid search: Dirichlet Calibration (ODIR λ ∈ {odir_lambdas})...")
    dirich_results = []
    for odir in odir_lambdas:
        r = fit_dirichlet_method(onnx_path, metadata, batch_size, limit, odir_lambda=odir)
        output_dir = build_output_path(output_base, metadata["experiment_name"], "dirichlet", odir)
        save_calibration_result(r, output_dir, "dirichlet", f"ODIR λ={odir}")
        dirich_results.append(r)
    results["dirichlet"] = dirich_results

    # Print summary
    print("\n" + "=" * 70)
    print("  GRID SEARCH SUMMARY — Internal Test After Calibration")
    print("=" * 70)
    print(f"  {'Method':<25} {'Lambda':<10} {'NLL':>8} {'ECE':>8} {'Acc':>8} {'F1-macro':>10}")
    print("-" * 70)

    for method_name, method_results in results.items():
        for r in method_results:
            te = r.get("test_eval", {})
            lam = r.get("vector_params", r.get("dirichlet_params", ""))
            lam_str = f"λ={lam}" if lam else "auto"
            print(f"  {method_name:<25} {lam_str:<10} "
                  f"{te.get('after',{}).get('nll',999):>8.4f} "
                  f"{te.get('after',{}).get('ece',999):>8.4f} "
                  f"{te.get('after',{}).get('accuracy',0):>8.4f} "
                  f"{te.get('f1_macro',0):>10.4f}")

    print("=" * 70)

    # Save grid search summary
    summary_path = output_base / "calibration" / metadata["experiment_name"] / "grid_search_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment": metadata["experiment_name"],
        "onnx": str(onnx_path),
        "grid": {k: [
            {
                "lambda": r.get("vector_params", r.get("dirichlet_params", "auto")),
                "test_eval": r.get("test_eval", {}),
            }
            for r in v
        ] for k, v in results.items()},
    }
    
    if not best_calibration:
        logger.warning(f"No valid calibration found for {method_name}")
        return

    best_lambda = best_calibration.get("method_kwargs", {}).get("odir_lambda") or \
                  best_calibration.get("method_kwargs", {}).get("l2_lambda")
    
    logger.info(f"==> Seçilen en iyi {method_name.upper()} Modeli: {best_calibration['name']}")
    
    # Save a symlink or copy to the best one
    best_path = build_output_path(base_dir, method_name, best_lambda)
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\nGrid search summary saved: {summary_path}")

    # Select winner based on lowest NLL on validation set
    best_method, min_nll, best_result = None, float("inf"), None
    for method_name, method_results in results.items():
        for r in method_results:
            nll = r.get("after", {}).get("nll", float("inf"))
            if nll < min_nll:
                min_nll = nll
                best_method = method_name
                best_result = r

    print(f"\n🏆 BEST: {best_method} with Validation NLL = {min_nll:.4f}")
    lam = best_result.get("vector_params", best_result.get("dirichlet_params", "auto"))
    print(f"   Lambda: {lam}")
    print(f"   Results: {build_output_path(output_base, metadata['experiment_name'], best_method, lam if lam != 'auto' else None)}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True, help="Path to ONNX model")
    parser.add_argument("--metadata", type=Path, required=True, help="Path to metadata.json")
    parser.add_argument(
        "--method",
        type=str,
        choices=["temperature", "vector_scaling", "dirichlet", "all"],
        default="temperature",
        help="Calibration method (default: temperature)",
    )
    parser.add_argument("--l2-lambda", type=float, default=None, help="L2 lambda for Vector Scaling")
    parser.add_argument("--odir-lambda", type=float, default=None, help="ODIR lambda for Dirichlet")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for ONNX inference")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples (for smoke test)")
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Run grid search over all methods and lambdas",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=Path("artifacts"),
        help="Base directory for output (default: artifacts/)",
    )
    args = parser.parse_args()

    # Load and validate metadata
    metadata = load_artifact_metadata(args.metadata)

    # Check forbidden splits
    val_csv = Path(metadata["data"]["val_csv"])
    try:
        from autolens_ai.evaluation.calibration import expected_calibration_split
        expected_calibration_split(val_csv)
    except ValueError as e:
        print(f"ERROR: {e}")
        return

    if args.grid_search:
        run_grid_search(args.onnx, metadata, args.output_base, args.batch_size, args.limit)
        return

    if args.method == "all":
        print("Running ALL methods (temperature, vector_scaling, dirichlet)...")
        run_single_method(args.onnx, metadata, "temperature", args.output_base, batch_size=args.batch_size, limit=args.limit)

        # Vector Scaling with default lambda 0.01
        run_single_method(args.onnx, metadata, "vector_scaling", args.output_base, l2_lambda=0.01, batch_size=args.batch_size, limit=args.limit)

        # Dirichlet with default lambda 0.01
        run_single_method(args.onnx, metadata, "dirichlet", args.output_base, odir_lambda=0.01, batch_size=args.batch_size, limit=args.limit)

        print("\n✅ All methods completed.")
        return

    if args.method == "temperature":
        # Temperature scaling ignores lambda params
        run_single_method(args.onnx, metadata, "temperature", args.output_base, batch_size=args.batch_size, limit=args.limit)
    elif args.method == "vector_scaling":
        if args.l2_lambda is None:
            print("WARNING: No --l2-lambda specified, using default 0.01")
            args.l2_lambda = 0.01
        run_single_method(args.onnx, metadata, "vector_scaling", args.output_base, l2_lambda=args.l2_lambda, batch_size=args.batch_size, limit=args.limit)
    elif args.method == "dirichlet":
        if args.odir_lambda is None:
            print("WARNING: No --odir-lambda specified, using default 0.01")
            args.odir_lambda = 0.01
        run_single_method(args.onnx, metadata, "dirichlet", args.output_base, odir_lambda=args.odir_lambda, batch_size=args.batch_size, limit=args.limit)


if __name__ == "__main__":
    main()