#!/usr/bin/env python3
"""Compute mean and std from training split for normalization.

Usage:
    uv run python scripts/compute_dataset_stats.py
    uv run python scripts/compute_dataset_stats.py --output artifacts/dataset/stats.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms  # type: ignore[import-untyped]
from tqdm import tqdm

from autolens_ai.training import AutoLensDataset


def compute_mean_std(
    csv_path: str = "artifacts/dataset/splits/train.csv",
    data_root: str = "datasets",
    batch_size: int = 64,
    num_workers: int = 4,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Compute mean and std from training dataset.
    
    Args:
        csv_path: Path to train.csv
        data_root: Root directory containing images
        batch_size: Batch size for loading
        num_workers: Number of workers
        
    Returns:
        (mean, std) tuples for RGB channels
    """
    # Create dataset with only resize and to_tensor (no normalization yet)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    
    dataset = AutoLensDataset(
        csv_path=csv_path,
        root_dir=data_root,
        transform=transform,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    print(f"Computing mean and std from {len(dataset)} training images...")
    
    # Compute mean
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_images = 0
    
    for images, _ in tqdm(loader, desc="Computing statistics"):
        batch_size = images.size(0)
        # Reshape to (batch, channels, height*width)
        images = images.view(batch_size, images.size(1), -1)
        # Sum over batch and pixels
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_size
    
    mean /= total_images
    std /= total_images
    
    return tuple(mean.tolist()), tuple(std.tolist())


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(description="Compute dataset statistics")
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/dataset/stats.json",
        help="Output JSON file path",
    )
    args = parser.parse_args()
    
    mean, std = compute_mean_std()
    
    # Save to JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    stats = {
        "mean": list(mean),
        "std": list(std),
        "description": "Dataset normalization statistics computed from train split",
    }
    
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Dataset Statistics (Train Split)")
    print("=" * 60)
    print(f"Mean (RGB): {mean}")
    print(f"Std  (RGB): {std}")
    print("=" * 60)
    print(f"\nSaved to: {output_path}")
    print("\nTraining will automatically load these values.")


if __name__ == "__main__":
    main()
