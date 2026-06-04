#!/usr/bin/env python3
"""Download AutoLens AI model artifacts from HuggingFace Hub.

Downloads the flagship DINOv3 weighted model (or any specified model)
from the morpeN1/autolens-* repos and writes an active_model.json
pointing to the local files.

Usage:
    python scripts/download_from_hf.py                  # download flagship
    python scripts/download_from_hf.py --model resnet18  # download specific
    python scripts/download_from_hf.py --list           # list available
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download, list_repo_files
except ImportError:
    print("ERROR: huggingface_hub not installed. Run: uv add huggingface_hub", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Model registry — maps short names to HF repo IDs
# ---------------------------------------------------------------------------
MODELS: dict[str, str] = {
    "dinov3-weighted": "morpeN1/autolens-dinov3-safe-weighted",
    "dinov3-focal":    "morpeN1/autolens-dinov3-safe-focal",
    "dinov3-vits16":   "morpeN1/autolens-dinov3-vits16",
    "efficientnet-b2": "morpeN1/autolens-efficientnet-b2",
    "resnet18":        "morpeN1/autolens-resnet18",
    "mobilenetv4":     "morpeN1/autolens-mobilenetv4",
}

EXPORT_ROOT = Path("artifacts/export")
ACTIVE_MODEL_JSON = Path("artifacts/demo/active_model.json")


def list_available() -> None:
    print("Available models:")
    for short, repo in MODELS.items():
        print(f"  {short:20s} → {repo}")


def download_model(short_name: str, force: bool = False) -> Path:
    """Download all artifacts for one model, return the export directory."""
    repo_id = MODELS[short_name]
    # Map short name to local dir name (same convention as existing exports)
    dir_map = {
        "dinov3-weighted": "dinov3_safe_weighted_latest",
        "dinov3-focal":    "dinov3_safe_focal_latest",
        "dinov3-vits16":   "dinov3_vits16_latest",
        "efficientnet-b2": "efficientnet_b2_latest",
        "resnet18":        "resnet18_latest",
        "mobilenetv4":     "mobilenetv4_latest",
    }
    local_dir = EXPORT_ROOT / dir_map[short_name]
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {short_name} from {repo_id} …")
    print(f"  → {local_dir}")

    # Files to download. ONNX is used for fast prediction; safetensors is used lazily for Grad-CAM.
    files = ["model.onnx", "model.safetensors", "metadata.json", "metrics.csv",
             "per_class_metrics.txt", "internal_test_results.json"]

    for fname in files:
        try:
            path = hf_hub_download(repo_id=repo_id, filename=fname,
                                   force_download=force)
            dest = local_dir / fname
            if not dest.exists() or force:
                dest.write_bytes(Path(path).read_bytes())
                print(f"  ✓ {fname}")
            else:
                print(f"  - {fname} (already exists)")
        except Exception as exc:
            print(f"  ✗ {fname}: {exc}", file=sys.stderr)

    # Calibration directory
    try:
        cal_files = list_repo_files(repo_id=repo_id, repo_type="model")
        cal_files = [f for f in cal_files if f.startswith("calibration/")]
        for cal_file in cal_files:
            rel = cal_file[len("calibration/"):]
            dest = local_dir / "calibration" / rel
            if not dest.exists() or force:
                dest.parent.mkdir(parents=True, exist_ok=True)
                path = hf_hub_download(repo_id=repo_id, filename=cal_file,
                                       force_download=force)
                dest.write_bytes(Path(path).read_bytes())
        if cal_files:
            print(f"  ✓ calibration/ ({len(cal_files)} files)")
    except Exception as exc:
        print(f"  ✗ calibration/: {exc}", file=sys.stderr)

    # Plots
    for plot in ["confusion_matrix.png", "training_loss.png",
                 "training_accuracy.png", "roc_curves.png"]:
        try:
            path = hf_hub_download(repo_id=repo_id, filename=plot,
                                   force_download=force)
            dest = local_dir / plot
            if not dest.exists() or force:
                dest.write_bytes(Path(path).read_bytes())
                print(f"  ✓ {plot}")
        except Exception:
            pass  # optional

    return local_dir


def write_active_model(export_dir: Path) -> None:
    """Write artifacts/demo/active_model.json pointing to the export dir."""
    meta = json.loads((export_dir / "metadata.json").read_text())

    # Find calibration: prefer dirichlet best, then top-level
    cal_dirichlet = export_dir / "calibration" / "dirichlet" / "best_calibration.json"
    cal_top = export_dir / "calibration.json"
    if cal_dirichlet.exists():
        cal_path = cal_dirichlet
        cal_method = "dirichlet"
    elif cal_top.exists():
        cal_path = cal_top
        cal_method = json.loads(cal_top.read_text()).get("method", "temperature")
    else:
        cal_path = None
        cal_method = "temperature"

    temperature = 1.0
    if cal_path and cal_method == "temperature":
        temperature = float(json.loads(cal_path.read_text()).get("temperature", 1.0))

    active = {
        "backend": "onnxruntime",
        "model_path": str(export_dir / "model.onnx"),
        "metadata_path": str(export_dir / "metadata.json"),
        "calibration_path": str(cal_path) if cal_path else None,
        "calibration_method": cal_method,
        "temperature": temperature,
        "class_labels": meta["classes"]["labels"],
        "display_labels": meta["classes"].get("display_labels",
                                               meta["classes"]["labels"]),
        "preprocessing": meta["preprocessing"],
        "source_run_id": meta["source"]["run_id"],
    }

    ACTIVE_MODEL_JSON.parent.mkdir(parents=True, exist_ok=True)
    ACTIVE_MODEL_JSON.write_text(json.dumps(active, indent=2))
    print(f"\n✓ Active model written to {ACTIVE_MODEL_JSON}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download AutoLens AI models from HuggingFace Hub"
    )
    parser.add_argument(
        "--model", "-m",
        default="dinov3-weighted",
        help=f"Model to download. One of: {', '.join(MODELS)}",
    )
    parser.add_argument("--list", "-l", action="store_true",
                        help="List available models and exit.")
    parser.add_argument("--force", "-f", action="store_true",
                        help="Force re-download even if files exist.")
    args = parser.parse_args()

    if args.list:
        list_available()
        return

    if args.model not in MODELS:
        print(f"ERROR: Unknown model '{args.model}'. Use --list to see options.",
              file=sys.stderr)
        sys.exit(1)

    export_dir = download_model(args.model, force=args.force)
    write_active_model(export_dir)
    print(f"\nDone. Run: make demo-gradio  OR  make demo-web")


if __name__ == "__main__":
    main()
