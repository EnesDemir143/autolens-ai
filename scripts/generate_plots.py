#!/usr/bin/env python3
"""Generate confusion matrix, per-class metrics, and training graphs from existing checkpoints.

Usage:
    uv run python scripts/generate_plots.py --ckpt checkpoints/baseline_0_resnet18_20260508_014023/best-13-0.8431.ckpt --log checkpoints/wandb/wandb/run-20260508_014024-ufyxb7fd/files/output.log
    uv run python scripts/generate_plots.py --ckpt checkpoints/baseline_0_mobilenetv4_20260508_025041/best-16-0.8340.ckpt --log checkpoints/wandb/wandb/run-20260508_025042-y1phlhks/files/output.log
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def parse_epoch_metrics(log_path: Path) -> dict[str, list]:
    """Parse val/loss, val/acc, val/f1_macro, train/loss_epoch, train/acc from output.log."""
    metrics: dict[str, list] = {
        "epoch": [], "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_f1_macro": [],
    }

    # Each completed epoch line looks like:
    # Epoch X/99 ━━━ 198/198 ... val/loss: 0.517 val/acc: 0.843 val/f1_macro: 0.782 train/loss_epoch: 0.027
    epoch_pattern = re.compile(
        r"Epoch\s+(\d+)/\d+.*?198/198.*?"
        r"val/loss:\s*([\d.]+).*?"
        r"val/acc:\s*([\d.]+).*?"
        r"val/f1_macro:\s*([\d.]+).*?"
        r"train/loss_epoch:\s*([\d.]+)",
        re.DOTALL,
    )
    # Also capture train/acc from the line that follows
    train_acc_pattern = re.compile(r"train/acc:\s*([\d.]+)")

    text = log_path.read_text(errors="ignore")
    # Strip ANSI escape codes
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    text = ansi_escape.sub("", text)

    # Find all epoch summary blocks — take the LAST occurrence per epoch
    # (progress bar updates many times, we want the final one)
    seen: dict[int, tuple] = {}
    for m in epoch_pattern.finditer(text):
        epoch = int(m.group(1))
        seen[epoch] = (
            float(m.group(2)),  # val_loss
            float(m.group(3)),  # val_acc
            float(m.group(4)),  # val_f1_macro
            float(m.group(5)),  # train_loss
        )

    # train/acc appears on a separate line after each epoch
    # pair it by finding "train/acc: X.XXX" near each epoch block
    train_acc_map: dict[int, float] = {}
    # Split by epoch markers and find train/acc in each segment
    epoch_splits = re.split(r"Epoch\s+(\d+)/\d+", text)
    current_epoch = -1
    for i, chunk in enumerate(epoch_splits):
        if chunk.isdigit():
            current_epoch = int(chunk)
        elif current_epoch >= 0:
            acc_matches = train_acc_pattern.findall(chunk)
            if acc_matches:
                train_acc_map[current_epoch] = float(acc_matches[-1])

    for epoch in sorted(seen.keys()):
        val_loss, val_acc, val_f1, train_loss = seen[epoch]
        metrics["epoch"].append(epoch)
        metrics["val_loss"].append(val_loss)
        metrics["val_acc"].append(val_acc)
        metrics["val_f1_macro"].append(val_f1)
        metrics["train_loss"].append(train_loss)
        metrics["train_acc"].append(train_acc_map.get(epoch, float("nan")))

    return metrics


def save_training_graphs(metrics: dict, save_dir: Path) -> None:
    epochs = metrics["epoch"]

    # Loss
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, metrics["train_loss"], label="Train Loss", marker="o", markersize=4)
    ax.plot(epochs, metrics["val_loss"], label="Val Loss", marker="s", markersize=4)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = save_dir / "training_loss.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"✓ Loss graph: {p}")

    # Accuracy
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, metrics["train_acc"], label="Train Accuracy", marker="o", markersize=4)
    ax.plot(epochs, metrics["val_acc"], label="Val Accuracy", marker="s", markersize=4)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.set_title("Training & Validation Accuracy")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = save_dir / "training_accuracy.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"✓ Accuracy graph: {p}")


def save_confusion_matrix(ckpt_path: Path, save_dir: Path, batch_size: int = 64) -> None:
    """Run test set through the model and save confusion matrix + per-class metrics."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    from autolens_ai.data.labels import TARGET_CLASSES
    from autolens_ai.training import AutoLensClassifier, AutoLensDataModule, PreprocessConfig

    class_names = list(TARGET_CLASSES)

    # Load model
    model = AutoLensClassifier.load_from_checkpoint(str(ckpt_path), map_location="cpu", strict=False)
    model.eval()

    # Load dataset stats
    stats_path = Path("artifacts/dataset/stats.json")
    with open(stats_path) as f:
        stats = json.load(f)

    preprocess_config = PreprocessConfig(
        resize_size=232, crop_size=224,
        mean=tuple(stats["mean"]), std=tuple(stats["std"]),
    )

    datamodule = AutoLensDataModule(
        data_root="datasets",
        train_csv="artifacts/dataset/splits/train.csv",
        val_csv="artifacts/dataset/splits/val.csv",
        test_csv="artifacts/dataset/splits/internal_test.csv",
        preprocess_config=preprocess_config,
        batch_size=batch_size,
        num_workers=0,
        use_augmentation=False,
    )
    datamodule.setup("test")

    # Collect predictions
    all_preds, all_labels = [], []
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)

    with torch.no_grad():
        for batch in datamodule.test_dataloader():
            images, labels = batch
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    import numpy as np
    from sklearn.metrics import classification_report, confusion_matrix

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(8)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=range(8), yticks=range(8),
        xticklabels=class_names, yticklabels=class_names,
        xlabel="Predicted", ylabel="True",
        title="Normalized Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i in range(8):
        for j in range(8):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=8)
    fig.tight_layout()
    p = save_dir / "confusion_matrix.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"✓ Confusion matrix: {p}")

    # Per-class report
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print("\nPer-class metrics:\n")
    print(report)
    p = save_dir / "per_class_metrics.txt"
    p.write_text(report)
    print(f"✓ Per-class metrics: {p}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to best .ckpt file")
    parser.add_argument("--log", required=True, help="Path to output.log from wandb run")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    save_dir = ckpt_path.parent
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving plots to: {save_dir}\n")

    # Training graphs from log
    print("Parsing training metrics from log...")
    metrics = parse_epoch_metrics(Path(args.log))
    print(f"  Found {len(metrics['epoch'])} epochs")
    save_training_graphs(metrics, save_dir)

    # Confusion matrix + per-class from checkpoint
    print("\nRunning test set inference...")
    save_confusion_matrix(ckpt_path, save_dir, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
