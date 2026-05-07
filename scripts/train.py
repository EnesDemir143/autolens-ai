#!/usr/bin/env python3
"""Training script for AutoLens AI baseline models.

Usage:
    uv run python scripts/train.py --config configs/experiments/baseline_0_resnet18.yaml
    uv run python scripts/train.py --config configs/experiments/baseline_0_resnet18.yaml --wandb
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from autolens_ai.training import (
    AutoLensClassifier,
    AutoLensDataModule,
    PreprocessConfig,
    create_trainer,
    print_device_info,
)


def load_config(config_path: str | Path) -> dict:
    """Load experiment config from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_dataset_stats(stats_path: str | Path = "artifacts/dataset/stats.json") -> tuple[list[float], list[float]]:
    """Load dataset statistics from JSON file.
    
    Returns:
        (mean, std) lists for RGB channels, or ImageNet defaults if file not found
    """
    stats_path = Path(stats_path)
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        print(f"Loaded dataset stats from: {stats_path}")
        print(f"  Mean: {stats['mean']}")
        print(f"  Std:  {stats['std']}")
        return stats["mean"], stats["std"]
    else:
        print(f"Dataset stats not found at {stats_path}, using ImageNet defaults")
        print("  Run 'make compute-stats' to compute dataset-specific values")
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def main() -> None:
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train AutoLens AI baseline model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config YAML file",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable W&B logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="autolens-ai",
        help="W&B project name",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run ID suffix for checkpoint dir",
    )
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    print(f"Loaded config from: {args.config}")
    print(f"Experiment: {config['experiment_name']}")
    
    # Add run ID to checkpoint dir if provided, otherwise use timestamp
    checkpoint_dir = config["checkpoint_dir"]
    if args.run_id:
        checkpoint_dir = f"{checkpoint_dir}_{args.run_id}"
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = f"{checkpoint_dir}_{timestamp}"
    
    print(f"Checkpoints will be saved to: {checkpoint_dir}")
    
    # Print device info
    print_device_info()
    
    # Load dataset stats (from JSON or config or defaults)
    mean, std = load_dataset_stats()
    # Override with config values if present
    mean = config.get("dataset_mean", mean)
    std = config.get("dataset_std", std)
    
    # Create preprocessing config
    preprocess_config = PreprocessConfig(
        resize_size=config["resize_size"],
        crop_size=config["crop_size"],
        mean=tuple(mean),
        std=tuple(std),
    )
    
    # Compute class weights if needed
    class_weights = None
    if config.get("use_class_weights", False):
        # Create temporary dataset to compute weights
        from autolens_ai.training import AutoLensDataset
        temp_dataset = AutoLensDataset(
            csv_path=config["train_csv"],
            root_dir=config["data_root"],
            transform=None,
        )
        class_counts = temp_dataset.get_class_counts()
        class_weights = AutoLensDataModule.compute_class_weights(
            class_counts,
            max_weight=config.get("max_class_weight", 5.0),
        )
        print(f"\nClass counts: {class_counts}")
        print(f"Class weights: {class_weights.tolist()}")
    
    # Create data module
    datamodule = AutoLensDataModule(
        data_root=config["data_root"],
        train_csv=config["train_csv"],
        val_csv=config["val_csv"],
        test_csv=config["test_csv"],
        preprocess_config=preprocess_config,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        use_augmentation=config["use_augmentation"],
        use_weighted_sampler=config.get("use_weighted_sampler", False),
        class_weights=class_weights,
    )
    
    # Create model
    model = AutoLensClassifier(
        model_name=config["model_name"],
        num_classes=config["num_classes"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        class_weights=class_weights,
        pretrained=config["pretrained"],
    )
    
    # Create trainer
    trainer = create_trainer(
        max_epochs=config["max_epochs"],
        accelerator=config.get("accelerator"),
        checkpoint_dir=checkpoint_dir,  # Use updated checkpoint_dir with timestamp
        early_stopping_patience=config["early_stopping_patience"],
        monitor_metric=config["monitor_metric"],
        monitor_mode=config["monitor_mode"],
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_name=config["experiment_name"],
    )
    
    # Train
    print("\nStarting training...")
    trainer.fit(model, datamodule)
    
    # Test with best checkpoint
    print("\nRunning test evaluation with best checkpoint...")
    trainer.test(model, datamodule, ckpt_path="best")
    
    print(f"\nTraining complete!")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"Best model: {checkpoint_dir}/best-*.ckpt")
    print(f"Last model: {checkpoint_dir}/last.ckpt")


if __name__ == "__main__":
    main()
