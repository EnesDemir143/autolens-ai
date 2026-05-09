#!/usr/bin/env python
"""Convert an AutoLens Lightning checkpoint to safetensors plus metadata."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors.torch import save_file

from autolens_ai.data.labels import TARGET_CLASSES

BEST_RE = re.compile(r"best-(?P<epoch>\d+)-(?P<score>\d+\.\d+)\.ckpt$")
DEFAULT_CONFIG = Path("configs/experiments/baseline_0_efficientnet_b2.yaml")
DEFAULT_EXPORT_DIR = Path("artifacts/export/efficientnet_b2_current")


def locate_best_efficientnet_checkpoint(root: Path = Path("checkpoints")) -> Path:
    """Return the highest-score EfficientNet-B2 F1 checkpoint."""
    candidates: list[tuple[float, Path]] = []
    for path in root.glob("baseline_0_efficientnet_b2_*/best-*.ckpt"):
        match = BEST_RE.match(path.name)
        if match:
            candidates.append((float(match.group("score")), path))
    if not candidates:
        raise FileNotFoundError(
            "No EfficientNet-B2 best checkpoint found under "
            "checkpoints/baseline_0_efficientnet_b2_*/best-*.ckpt"
        )
    return max(candidates, key=lambda item: item[0])[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state() -> dict[str, str]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(["git", "status", "--short"], text=True).strip()
    except Exception as exc:  # pragma: no cover - git may be unavailable in archives
        return {"error": str(exc)}
    return {"commit": commit, "dirty": str(bool(status)).lower()}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)  # type: ignore[no-any-return]


def latest_epoch_metrics(metrics_csv: Path) -> dict[str, float | int]:
    if not metrics_csv.exists():
        return {}
    rows: list[dict[str, str]] = []
    with metrics_csv.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    metric_rows = [row for row in rows if row.get("val/f1_macro") or row.get("test/f1_macro")]
    if not metric_rows:
        return {}
    row = metric_rows[-1]
    out: dict[str, float | int] = {}
    for key in (
        "epoch",
        "val/acc",
        "val/f1_macro",
        "val/f1_weighted",
        "val/loss",
        "test/acc",
        "test/f1_macro",
        "test/f1_weighted",
        "test/loss",
    ):
        value = row.get(key)
        if value:
            out[key] = int(float(value)) if key == "epoch" else float(value)
    return out


def strip_lightning_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Keep only `model.*` tensors and remove the Lightning module prefix."""
    weights: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        if key.startswith("model."):
            weights[key.removeprefix("model.")] = tensor.detach().cpu()
    if not weights:
        raise ValueError("Checkpoint state_dict does not contain model.* tensors")
    return weights


def build_metadata(
    checkpoint: Path, config_path: Path, output_dir: Path, ckpt: dict[str, Any]
) -> dict[str, Any]:
    config = load_yaml(config_path)
    hparams = ckpt.get("hyper_parameters", {})
    run_dir = checkpoint.parent
    return {
        "artifact_format_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "source": {
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "run_id": run_dir.name,
            "config_path": str(config_path),
            "metrics_csv": str(run_dir / "metrics.csv"),
            "git": git_state(),
        },
        "model": {
            "architecture": "timm",
            "model_name": hparams.get("model_name", config["model_name"]),
            "num_classes": int(hparams.get("num_classes", config["num_classes"])),
            "pretrained_at_training": bool(
                hparams.get("pretrained", config.get("pretrained", True))
            ),
            "use_lora": bool(hparams.get("use_lora", False)),
            "lora_r": int(hparams.get("lora_r", 8)),
            "lora_alpha": int(hparams.get("lora_alpha", 16)),
            "lora_dropout": float(hparams.get("lora_dropout", 0.1)),
            "lora_target_modules": hparams.get("lora_target_modules"),
        },
        "classes": {
            "labels": list(TARGET_CLASSES),
            "class_to_idx": {label: idx for idx, label in enumerate(TARGET_CLASSES)},
            "display_labels": list(TARGET_CLASSES),
        },
        "preprocessing": {
            "image_size": int(config["crop_size"]),
            "resize_size": int(config["resize_size"]),
            "crop_size": int(config["crop_size"]),
            "mean": [float(value) for value in config["dataset_mean"]],
            "std": [float(value) for value in config["dataset_std"]],
            "color_mode": "RGB",
        },
        "data": {
            "train_csv": config["train_csv"],
            "val_csv": config["val_csv"],
            "test_csv_reserved_not_used_for_calibration": config["test_csv"],
            "data_root": config["data_root"],
        },
        "training": {
            "epoch": ckpt.get("epoch"),
            "global_step": ckpt.get("global_step"),
            "learning_rate": hparams.get("learning_rate"),
            "weight_decay": hparams.get("weight_decay"),
            "label_smoothing": hparams.get("label_smoothing"),
            "metrics_summary": latest_epoch_metrics(run_dir / "metrics.csv"),
        },
        "artifacts": {
            "directory": str(output_dir),
            "safetensors": str(output_dir / "model.safetensors"),
            "metadata": str(output_dir / "metadata.json"),
            "onnx": str(output_dir / "model.onnx"),
            "calibration": str(output_dir / "calibration.json"),
        },
        "license_notes": {
            "base_model": f"{hparams.get('model_name', config['model_name'])} pretrained backbone; verify upstream license before public publishing.",
            "dataset": "Do not publish merged raw dataset images; document source links/counts instead.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Lightning .ckpt path")
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="Source experiment YAML"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_EXPORT_DIR, help="Export directory"
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint or locate_best_efficientnet_checkpoint()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    weights = strip_lightning_prefix(ckpt["state_dict"])
    safetensors_path = args.output_dir / "model.safetensors"
    metadata_path = args.output_dir / "metadata.json"

    save_file(weights, str(safetensors_path))
    metadata = build_metadata(checkpoint, args.config, args.output_dir, ckpt)
    metadata["artifacts"]["safetensors_sha256"] = sha256_file(safetensors_path)
    metadata["artifacts"]["safetensors_size_bytes"] = safetensors_path.stat().st_size
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"checkpoint={checkpoint}")
    print(f"safetensors={safetensors_path}")
    print(f"metadata={metadata_path}")


if __name__ == "__main__":
    main()
