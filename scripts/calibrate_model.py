#!/usr/bin/env python
"""Fit validation-only temperature scaling for an AutoLens ONNX model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autolens_ai.evaluation import fit_temperature_scaling
from autolens_ai.inference import load_artifact_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--onnx", type=Path, default=Path("artifacts/export/efficientnet_b2_current/model.onnx")
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("artifacts/export/efficientnet_b2_current/metadata.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/export/efficientnet_b2_current/calibration.json"),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional smoke-test limit; omit for full validation calibration",
    )
    args = parser.parse_args()

    metadata = load_artifact_metadata(args.metadata)
    payload = fit_temperature_scaling(args.onnx, metadata, args.batch_size, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    metadata.setdefault("artifacts", {})["calibration"] = str(args.output)
    args.metadata.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"calibration={args.output}")
    print(f"temperature={payload['temperature']:.6f}")
    print(f"before={payload['before']}")
    print(f"after={payload['after']}")


if __name__ == "__main__":
    main()
