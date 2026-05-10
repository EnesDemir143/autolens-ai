#!/usr/bin/env python
"""Export an AutoLens safetensors artifact to ONNX and run a smoke check."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch

from autolens_ai.inference import load_artifact_metadata, load_model_from_artifact

MAX_SIZE_MB = 95.0


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def smoke_onnx(onnx_path: Path, image_size: int, num_classes: int) -> dict[str, Any]:
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    dummy = np.zeros((1, 3, image_size, image_size), dtype=np.float32)
    outputs = session.run(None, {input_name: dummy})
    logits = outputs[0]
    if logits.shape != (1, num_classes):
        raise RuntimeError(f"Unexpected ONNX logits shape: {logits.shape}")
    return {
        "provider": session.get_providers()[0],
        "input_name": input_name,
        "output_shape": list(logits.shape),
    }


def maybe_simplify(onnx_path: Path, simplified_path: Path) -> dict[str, Any]:
    try:
        from onnxsim import simplify  # type: ignore[import-untyped]
        import onnx  # type: ignore[import-untyped]

        model = onnx.load(str(onnx_path))
        simplified, ok = simplify(model)
        if not ok:
            return {"attempted": True, "ok": False, "reason": "onnxsim check returned false"}
        onnx.save(simplified, str(simplified_path))
        return {"attempted": True, "ok": True, "path": str(simplified_path)}
    except Exception as exc:  # pragma: no cover - optional tool can fail on exporter details
        return {"attempted": True, "ok": False, "reason": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--safetensors",
        type=Path,
        default=Path("artifacts/export/efficientnet_b2_current/model.safetensors"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("artifacts/export/efficientnet_b2_current/metadata.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/export/efficientnet_b2_current/model.onnx")
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--skip-simplify", action="store_true")
    args = parser.parse_args()

    metadata = load_artifact_metadata(args.metadata)
    image_size = int(metadata["preprocessing"]["image_size"])
    num_classes = int(metadata["model"]["num_classes"])
    model = load_model_from_artifact(args.safetensors, args.metadata)
    dummy = torch.zeros(1, 3, image_size, image_size, dtype=torch.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # EVA (DINOv3) patch: TorchScript tracer requires is_causal to be bool, not Tensor.
    try:
        import timm.models.eva as _eva_module  # type: ignore[import-untyped]
        _orig_attn_fwd = _eva_module.EvaAttention.forward

        def _patched_attn_fwd(self, x, rope=None, attn_mask=None, is_causal=False):  # type: ignore[override]
            if isinstance(is_causal, torch.Tensor):
                is_causal = bool(is_causal.item())
            return _orig_attn_fwd(self, x, rope=rope, attn_mask=attn_mask, is_causal=is_causal)

        _eva_module.EvaAttention.forward = _patched_attn_fwd  # type: ignore[method-assign]
    except (ImportError, AttributeError):
        pass  # Not an EVA model — no patch needed

    torch.onnx.export(
        model,
        dummy,
        args.output,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False,
    )

    smoke = smoke_onnx(args.output, image_size, num_classes)
    simplified: dict[str, Any] = {"attempted": False}
    simplified_path = args.output.with_name("model.simplified.onnx")
    if not args.skip_simplify:
        simplified = maybe_simplify(args.output, simplified_path)
        if simplified.get("ok"):
            smoke_onnx(simplified_path, image_size, num_classes)

    size_report = {
        "max_size_mb": MAX_SIZE_MB,
        "artifacts": {
            "safetensors": {
                "path": str(args.safetensors),
                "size_mb": round(file_size_mb(args.safetensors), 3),
                "under_limit": file_size_mb(args.safetensors) <= MAX_SIZE_MB,
            },
            "onnx": {
                "path": str(args.output),
                "size_mb": round(file_size_mb(args.output), 3),
                "under_limit": file_size_mb(args.output) <= MAX_SIZE_MB,
            },
        },
        "onnxruntime_smoke": smoke,
        "simplification": simplified,
    }
    if simplified.get("ok"):
        size_report["artifacts"]["simplified_onnx"] = {
            "path": str(simplified_path),
            "size_mb": round(file_size_mb(simplified_path), 3),
            "under_limit": file_size_mb(simplified_path) <= MAX_SIZE_MB,
        }

    report_path = args.output.parent / "size_report.json"
    report_path.write_text(json.dumps(size_report, indent=2) + "\n", encoding="utf-8")

    metadata.setdefault("artifacts", {})["onnx"] = str(args.output)
    metadata["artifacts"]["size_report"] = str(report_path)
    args.metadata.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"onnx={args.output}")
    print(f"size_report={report_path}")
    print(f"onnxruntime_smoke={smoke}")


if __name__ == "__main__":
    main()
