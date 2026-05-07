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
}


def create_model(
    model_name: str,
    num_classes: int = 8,
    pretrained: bool = True,
) -> nn.Module:
    """Create a CNN baseline model.
    
    Args:
        model_name: One of 'mobilenetv4_conv_medium', 'efficientnet_b2', 'resnet18'
        num_classes: Number of output classes (default: 8 for AutoLens)
        pretrained: Whether to use pretrained weights
        
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
        # Create timm model
        model = timm.create_model(
            config["name"],  # type: ignore[arg-type]
            pretrained=pretrained,
            num_classes=num_classes,
        )
    elif source == "torchvision":
        # Create torchvision model
        if model_name == "resnet18":
            if pretrained:
                model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            else:
                model = models.resnet18(weights=None)
            
            # Replace final FC layer
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unknown source: {source}")
    
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
