"""CNN baseline model factory for AutoLens AI.

Requirements: TRN-01, TRN-02, TRN-03
Phase decision: D-02, D-03, D-05

This module provides MobileNetV4 Conv Medium, EfficientNet-B2, and ResNet18
as baseline/comparison models. DINOv3 is the fourth model family and belongs
to Phase 4.
"""

from __future__ import annotations

from typing import Any

import timm
import torch.nn as nn
from torchvision import models  # type: ignore[import-untyped]


# Model registry for Phase 3 CNN baselines
BASELINE_MODELS = {
    "mobilenetv4_conv_medium": {
        "source": "timm",
        "name": "mobilenetv4_conv_medium",
        "pretrained": True,
        "requirement": "TRN-01",
    },
    "efficientnet_b2": {
        "source": "timm",
        "name": "efficientnet_b2",
        "pretrained": True,
        "requirement": "TRN-02",
    },
    "resnet18": {
        "source": "torchvision",
        "name": "resnet18",
        "pretrained": True,
        "requirement": "TRN-03",
    },
    "vit_small_patch16_dinov3": {
        "source": "timm",
        "name": "vit_small_patch16_dinov3",
        "pretrained": True,
        "requirement": "TRN-04",
    },
}


def create_model(
    model_name: str,
    num_classes: int = 8,
    pretrained: bool = True,
    use_lora: bool = False,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    lora_target_modules: list[str] | None = None,
) -> nn.Module:
    """Create a model.

    Args:
        model_name: One of 'mobilenetv4_conv_medium', 'efficientnet_b2', 'resnet18', 'vit_small_patch16_dinov3'
        num_classes: Number of output classes (default: 8 for AutoLens)
        pretrained: Whether to use pretrained weights
        use_lora: Apply LoRA adapters — only supported for ViT models
        lora_r: LoRA rank
        lora_alpha: LoRA alpha scaling
        lora_dropout: LoRA dropout

    Returns:
        PyTorch model with modified classifier head

    Raises:
        ValueError: If model_name is not in BASELINE_MODELS
    """
    if model_name not in BASELINE_MODELS:
        available = ", ".join(BASELINE_MODELS.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")

    config = BASELINE_MODELS[model_name]
    source = config["source"]

    if source == "timm":
        model = timm.create_model(
            config["name"],  # type: ignore[arg-type]
            pretrained=pretrained,
            num_classes=num_classes,
        )
    elif source == "torchvision":
        if model_name == "resnet18":
            if pretrained:
                model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            else:
                model = models.resnet18(weights=None)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unknown source: {source}")

    # Apply LoRA — only for ViT-based models
    if use_lora:
        if "vit" not in model_name:
            raise ValueError(f"LoRA is only supported for ViT models, got: {model_name}")
        from peft import LoraConfig, get_peft_model

        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules or ["qkv", "proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)

        # Unfreeze classifier head
        for name, param in model.named_parameters():
            if "head" in name or "classifier" in name:
                param.requires_grad = True

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"LoRA: trainable {trainable/1e6:.2f}M / {total/1e6:.2f}M params ({100*trainable/total:.1f}%)")

    return model


def list_available_models() -> list[str]:
    """List available baseline models."""
    return list(BASELINE_MODELS.keys())


def get_model_info(model_name: str) -> dict[str, Any]:
    """Get information about a baseline model.
    
    Args:
        model_name: Model name
        
    Returns:
        Dictionary with model metadata
    """
    if model_name not in BASELINE_MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    return BASELINE_MODELS[model_name].copy()
