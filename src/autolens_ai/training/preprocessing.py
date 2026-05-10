"""Preprocessing configurations for AutoLens AI training.

Baseline 0: No augmentation, deterministic preprocessing.
Requirements: TRN-06, TRN-07
Phase decision: D-06
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torchvision import transforms  # type: ignore[import-untyped]


@dataclass
class PreprocessConfig:
    """Preprocessing configuration."""

    resize_size: int = 256
    crop_size: int = 224
    # ImageNet defaults - override with dataset-specific values for better performance
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    crop_scale: tuple[float, float] = (0.8, 1.0)
    crop_ratio: tuple[float, float] = (0.85, 1.15)
    color_jitter_brightness: float = 0.1
    color_jitter_contrast: float = 0.1
    color_jitter_saturation: float = 0.1
    color_jitter_hue: float = 0.0
    random_erasing_p: float = 0.0
    random_erasing_scale: tuple[float, float] = (0.02, 0.10)
    random_erasing_ratio: tuple[float, float] = (0.3, 3.3)

    def get_train_transform(self, augment: bool = False) -> Any:
        """Get training transform.

        Args:
            augment: If True, apply light augmentation (Baseline 1+). If False, use Baseline 0.
        """
        if not augment:
            # Baseline 0: deterministic preprocessing only
            return transforms.Compose(
                [
                    transforms.Resize(self.resize_size),
                    transforms.CenterCrop(self.crop_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.mean, std=self.std),
                ]
            )
        else:
            # Baseline 1+: light augmentation
            transform_steps: list[Any] = [
                transforms.RandomResizedCrop(
                    self.crop_size,
                    scale=self.crop_scale,
                    ratio=self.crop_ratio,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=self.color_jitter_brightness,
                    contrast=self.color_jitter_contrast,
                    saturation=self.color_jitter_saturation,
                    hue=self.color_jitter_hue,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
            if self.random_erasing_p > 0:
                transform_steps.append(
                    transforms.RandomErasing(
                        p=self.random_erasing_p,
                        scale=self.random_erasing_scale,
                        ratio=self.random_erasing_ratio,
                        value=0.0,
                    )
                )
            return transforms.Compose(transform_steps)

    def get_val_transform(self) -> Any:
        """Get validation/test transform (always deterministic)."""
        return transforms.Compose(
            [
                transforms.Resize(self.resize_size),
                transforms.CenterCrop(self.crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )


def compute_dataset_stats(
    dataset: Any,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Compute mean and std from training dataset.

    Args:
        dataset: PyTorch dataset with images

    Returns:
        (mean, std) tuples for RGB channels
    """
    # Use a subset for efficiency
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )

    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_images = 0

    for images, _ in loader:
        batch_size = images.size(0)
        images = images.view(batch_size, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_size

    mean /= total_images
    std /= total_images

    return tuple(mean.tolist()), tuple(std.tolist())


# Baseline 0 config: no augmentation
BASELINE_0_CONFIG = PreprocessConfig()
