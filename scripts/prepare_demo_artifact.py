#!/usr/bin/env python
"""Write the active Gradio/demo artifact pointer."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from autolens_ai.inference import load_artifact_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("artifacts/export/efficientnet_b2_current/metadata.json"),
    )
    parser.add_argument(
        "--onnx", type=Path, default=Path("artifacts/export/efficientnet_b2_current/model.onnx")
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("artifacts/export/efficientnet_b2_current/calibration.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/demo/active_model.json"))
    args = parser.parse_args()

    metadata = load_artifact_metadata(args.metadata)
    calibration = (
        json.loads(args.calibration.read_text(encoding="utf-8"))
        if args.calibration.exists()
        else None
    )
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "backend": "onnxruntime",
        "model_path": str(args.onnx),
        "metadata_path": str(args.metadata),
        "calibration_path": str(args.calibration) if args.calibration.exists() else None,
        "temperature": calibration["temperature"] if calibration else 1.0,
        "class_labels": metadata["classes"]["labels"],
        "display_labels": metadata["classes"].get("display_labels", metadata["classes"]["labels"]),
        "preprocessing": metadata["preprocessing"],
        "source_run_id": metadata["source"]["run_id"],
        "contract": "Phase 5 should replace this JSON pointer, not edit UI prediction code, when final winner changes.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"active_demo_artifact={args.output}")


if __name__ == "__main__":
    main()
