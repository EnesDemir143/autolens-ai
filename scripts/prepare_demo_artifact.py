#!/usr/bin/env python
"""Write the active Gradio/demo artifact pointer.

Supports selecting the best calibration from a comparison folder,
or pointing directly to a specific calibration result.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from autolens_ai.inference import load_artifact_metadata


def select_best_calibration(calibration_dir: Path) -> tuple[Path, dict]:
    """Walk calibration results and pick the one with lowest validation NLL."""
    best_path = None
    best_data = None
    min_nll = float("inf")

    for calib_json in calibration_dir.rglob("calibration.json"):
        if "grid_search_summary" in str(calib_json):
            continue
        try:
            data = json.loads(calib_json.read_text(encoding="utf-8"))
            # Prefer validation NLL for academic correctness (zero test-leakage)
            nll = data.get("after", {}).get("nll", float("inf"))
            if nll < min_nll:
                min_nll = nll
                best_data = data
                best_path = calib_json
        except Exception:
            continue

    if best_path is None:
        raise FileNotFoundError(f"No valid calibration.json found in {calibration_dir}")

    return best_path, best_data


def extract_temperature(calibration_data: dict | None) -> float:
    """Get temperature from calibration result, or default 1.0."""
    if calibration_data is None:
        return 1.0
    if "temperature" in calibration_data and calibration_data["temperature"] is not None:
        return float(calibration_data["temperature"])

    # For vector_scaling / dirichlet, temperature is effectively 1.0
    # but we need to signal the method to ONNXPredictor via the calibration JSON
    return 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("artifacts/export/efficientnet_b2_current/metadata.json"),
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=Path("artifacts/export/efficientnet_b2_current/model.onnx"),
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="Specific calibration.json file (auto-detects if omitted)",
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=None,
        help="Directory containing calibration subfolders (auto-selects best)",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/demo/active_model.json"))
    args = parser.parse_args()

    metadata = load_artifact_metadata(args.metadata)

    # Determine calibration source
    calibration_data = None
    calibration_path = args.calibration

    if args.calibration_dir and not args.calibration:
        calibration_path, calibration_data = select_best_calibration(args.calibration_dir)
        print(f"Auto-selected best calibration: {calibration_path}")
    elif args.calibration and args.calibration.exists():
        calibration_data = json.loads(args.calibration.read_text(encoding="utf-8"))
    elif args.calibration and not args.calibration.exists():
        print(f"WARNING: Calibration file not found: {calibration_path}")

    temperature = extract_temperature(calibration_data)
    method = calibration_data.get("method", "temperature_scaling") if calibration_data else "temperature"

    # Get class labels from metadata
    class_labels = metadata.get("classes", {}).get("labels", [])
    if not class_labels and "data" in metadata:
        # Fallback: infer from class indices
        num_classes = metadata.get("model", {}).get("num_classes", 8)
        class_labels = [f"class_{i}" for i in range(num_classes)]

    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "backend": "onnxruntime",
        "calibration_method": method,
        "model_path": str(args.onnx),
        "metadata_path": str(args.metadata),
        "calibration_path": str(calibration_path) if calibration_path else None,
        "temperature": temperature,
        "class_labels": class_labels,
        "display_labels": metadata.get("classes", {}).get(
            "display_labels", class_labels
        ),
        "preprocessing": metadata.get("preprocessing", {}),
        "source_run_id": metadata.get("source", {}).get("run_id", "unknown"),
        "contract": "Phase 5 should replace this JSON pointer, "
        "not edit UI prediction code, when final winner changes.",
    }

    # Include test metrics if available
    if calibration_data and "test_eval" in calibration_data:
        payload["test_metrics"] = {
            "f1_macro": calibration_data["test_eval"].get("f1_macro"),
            "f1_weighted": calibration_data["test_eval"].get("f1_weighted"),
            "precision_macro": calibration_data["test_eval"].get("precision_macro"),
            "recall_macro": calibration_data["test_eval"].get("recall_macro"),
            "accuracy": calibration_data["test_eval"]
                .get("after", {})
                .get("accuracy"),
            "ece": calibration_data["test_eval"]
                .get("after", {})
                .get("ece"),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"active_demo_artifact={args.output}")
    print(f"  method={method}, temperature={temperature}")


if __name__ == "__main__":
    main()