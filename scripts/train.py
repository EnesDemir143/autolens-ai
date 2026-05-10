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

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.tuner import Tuner

from autolens_ai.training import (
    AutoLensClassifier,
    AutoLensDataModule,
    PreprocessConfig,
    create_trainer,
    print_device_info,
)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across all libraries."""
    import random
    import numpy as np
    
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # PyTorch Lightning
    pl.seed_everything(seed, workers=True)
    
    # CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    
    print(f"✓ Random seed set to {seed} (Python, NumPy, PyTorch, Lightning)")


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


def _save_training_graphs(trainer: pl.Trainer, checkpoint_dir: str) -> None:
    """Save training/validation loss and accuracy graphs as PNG files."""
    import matplotlib.pyplot as plt

    # Read metrics from Lightning CSV logger
    metrics_path = None
    for logger in trainer.loggers:
        if hasattr(logger, "log_dir") and logger.log_dir is not None:
            candidate = Path(logger.log_dir) / "metrics.csv"
            if candidate.exists():
                metrics_path = candidate
                break

    if metrics_path is None:
        print("WARNING: metrics.csv not found, skipping training graphs")
        return

    import csv

    rows: list[dict] = []
    with open(metrics_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    def _collect(key: str) -> tuple[list[float], list[float]]:
        epochs, vals = [], []
        for row in rows:
            if row.get(key, "") != "":
                try:
                    epochs.append(float(row["epoch"]))
                    vals.append(float(row[key]))
                except (ValueError, KeyError):
                    pass
        return epochs, vals

    save_dir = Path(checkpoint_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- Loss graph ---
    fig, ax = plt.subplots(figsize=(9, 5))
    te, tv = _collect("train/loss_epoch")
    ve, vv = _collect("val/loss")
    if te:
        ax.plot(te, tv, label="Train Loss", marker="o", markersize=3)
    if ve:
        ax.plot(ve, vv, label="Val Loss", marker="s", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    loss_path = save_dir / "training_loss.png"
    fig.savefig(loss_path, dpi=150)
    plt.close(fig)
    print(f"✓ Loss graph saved: {loss_path}")

    # --- Accuracy graph ---
    fig, ax = plt.subplots(figsize=(9, 5))
    te, tv = _collect("train/acc")
    ve, vv = _collect("val/acc")
    if te:
        ax.plot(te, tv, label="Train Accuracy", marker="o", markersize=3)
    if ve:
        ax.plot(ve, vv, label="Val Accuracy", marker="s", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training & Validation Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    acc_path = save_dir / "training_accuracy.png"
    fig.savefig(acc_path, dpi=150)
    plt.close(fig)
    print(f"✓ Accuracy graph saved: {acc_path}")


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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint (requires --run-id)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: from config or 42)",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override max_epochs from config",
    )
    parser.add_argument(
        "--find-lr",
        action="store_true",
        help="Run learning rate finder before training",
    )
    parser.add_argument(
        "--find-batch-size",
        action="store_true",
        help="Run batch size finder before training",
    )
    parser.add_argument(
        "--lora",
        action="store_true",
        help="Use LoRA fine-tuning (only for ViT models, freezes backbone)",
    )
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    print(f"Loaded config from: {args.config}")
    print(f"Experiment: {config['experiment_name']}")
    
    # Set seed for reproducibility
    seed = args.seed if args.seed is not None else config.get("seed", 42)
    set_seed(seed)
    print(f"Random seed: {seed}")
    
    # Handle checkpoint directory and resume
    checkpoint_dir = config["checkpoint_dir"]
    resume_ckpt = None
    
    if args.resume:
        if not args.run_id:
            print("ERROR: --resume requires --run-id to specify which run to resume")
            return
        checkpoint_dir = f"{checkpoint_dir}_{args.run_id}"
        last_ckpt = Path(checkpoint_dir) / "last.ckpt"
        if last_ckpt.exists():
            resume_ckpt = str(last_ckpt)
            print(f"Resuming from: {resume_ckpt}")
        else:
            print(f"WARNING: No checkpoint found at {last_ckpt}, starting fresh")
    elif args.run_id:
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
        crop_scale=tuple(config.get("aug_crop_scale", (0.8, 1.0))),
        crop_ratio=tuple(config.get("aug_crop_ratio", (0.85, 1.15))),
        color_jitter_brightness=config.get("aug_brightness", 0.1),
        color_jitter_contrast=config.get("aug_contrast", 0.1),
        color_jitter_saturation=config.get("aug_saturation", 0.1),
        color_jitter_hue=config.get("aug_hue", 0.0),
        random_erasing_p=config.get("random_erasing_p", 0.0),
        random_erasing_scale=tuple(config.get("random_erasing_scale", (0.02, 0.10))),
        random_erasing_ratio=tuple(config.get("random_erasing_ratio", (0.3, 3.3))),
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
        drop_path_rate=config.get("drop_path_rate", 0.0),
        label_smoothing=config.get("label_smoothing", 0.0),
        use_focal_loss=config.get("use_focal_loss", False),
        focal_alpha=config.get("focal_alpha", 1.0),
        focal_gamma=config.get("focal_gamma", 2.0),
        max_epochs=args.max_epochs if args.max_epochs is not None else config["max_epochs"],
        use_lora=args.lora,
        lora_r=config.get("lora_r", 8),
        lora_alpha=config.get("lora_alpha", 16),
        lora_dropout=config.get("lora_dropout", 0.1),
        lora_target_modules=config.get("lora_target_modules", None),
    )
    
    # Create trainer
    trainer = create_trainer(
        max_epochs=args.max_epochs if args.max_epochs is not None else config["max_epochs"],
        accelerator=config.get("accelerator"),
        checkpoint_dir=checkpoint_dir,  # Use updated checkpoint_dir with timestamp
        early_stopping_patience=config["early_stopping_patience"],
        monitor_metric=config["monitor_metric"],
        monitor_mode=config["monitor_mode"],
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_name=config["experiment_name"],
        # Stability & performance
        gradient_clip_val=config.get("gradient_clip_val", 1.0),
        precision=config.get("precision", "bf16-mixed"),
        # Advanced features
        use_ema=config.get("use_ema", False),
        ema_decay=config.get("ema_decay", 0.999),
    )
    
    # Train
    print("\nStarting training...")
    if resume_ckpt:
        print(f"Resuming from checkpoint: {resume_ckpt}")
    
    # Optional: Find optimal learning rate
    if args.find_lr:
        print("\nRunning learning rate finder...")
        lr_finder = Tuner(trainer).lr_find(model, datamodule)
        if lr_finder:
            suggested_lr = lr_finder.suggestion()
            fig = lr_finder.plot(suggest=True)
            fig.savefig(f"{checkpoint_dir}/lr_finder.png")
            print(f"Suggested LR: {suggested_lr}")
            print(f"LR finder plot saved to: {checkpoint_dir}/lr_finder.png")
            
            # Save to JSON
            import json
            tune_results = {
                "learning_rate": {
                    "suggested": float(suggested_lr),
                    "current": config["learning_rate"],
                    "plot": f"{checkpoint_dir}/lr_finder.png",
                },
                "config_file": args.config,
                "timestamp": str(Path(checkpoint_dir).name.split("_")[-1]),
            }
            tune_path = Path(checkpoint_dir) / "tune_results.json"
            with open(tune_path, "w") as f:
                json.dump(tune_results, f, indent=2)
            print(f"✓ Results saved to: {tune_path}")
            print("\nTo use suggested LR, update config:")
            print(f"  learning_rate: {suggested_lr}")
            return
    
    # Optional: Find optimal batch size
    if args.find_batch_size:
        print("\nRunning batch size finder...")
        Tuner(trainer).scale_batch_size(model, datamodule, mode="power")
        suggested_bs = datamodule.batch_size
        print(f"Optimal batch size: {suggested_bs}")
        
        # Save to JSON
        import json
        tune_results = {
            "batch_size": {
                "suggested": int(suggested_bs),
                "current": config["batch_size"],
            },
            "config_file": args.config,
            "timestamp": str(Path(checkpoint_dir).name.split("_")[-1]),
        }
        tune_path = Path(checkpoint_dir) / "tune_results.json"
        with open(tune_path, "w") as f:
            json.dump(tune_results, f, indent=2)
        print(f"✓ Results saved to: {tune_path}")
        print("\nTo use suggested batch size, update config:")
        print(f"  batch_size: {suggested_bs}")
        return
    
    trainer.fit(model, datamodule, ckpt_path=resume_ckpt)
    
    # Test with best checkpoint
    print("\nRunning test evaluation with best checkpoint...")
    trainer.test(model, datamodule, ckpt_path="best")

    # Save training graphs (loss + accuracy)
    _save_training_graphs(trainer, checkpoint_dir)

    print("\nTraining complete!")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"Best model: {checkpoint_dir}/best-*.ckpt")
    print(f"Last model: {checkpoint_dir}/last.ckpt")


if __name__ == "__main__":
    main()
