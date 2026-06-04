"""Load deployable AutoLens model artifacts.

The Phase 4 artifact contract is intentionally small:
`model.safetensors` contains only model weights, while `metadata.json` contains
all non-tensor context needed to rebuild the model and preprocessing pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from autolens_ai.models import create_model


def load_artifact_metadata(metadata_path: str | Path) -> dict[str, Any]:
    """Load artifact metadata JSON."""
    path = Path(metadata_path)
    with path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return metadata  # type: ignore[no-any-return]


def load_model_from_artifact(
    safetensors_path: str | Path,
    metadata_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> torch.nn.Module:
    """Rebuild a PyTorch model from safetensors plus metadata."""
    metadata = load_artifact_metadata(metadata_path)
    model = create_model(
        model_name=metadata["model"]["model_name"],
        num_classes=int(metadata["model"]["num_classes"]),
        pretrained=False,
        use_lora=bool(metadata["model"].get("use_lora", False)),
        lora_r=int(metadata["model"].get("lora_r", 8)),
        lora_alpha=int(metadata["model"].get("lora_alpha", 16)),
        lora_dropout=float(metadata["model"].get("lora_dropout", 0.1)),
        lora_target_modules=metadata["model"].get("lora_target_modules"),
    )
    state_dict = load_file(str(safetensors_path), device=str(map_location))
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Artifact state mismatch: missing={missing}, unexpected={unexpected}")
    model.eval()
    return model
