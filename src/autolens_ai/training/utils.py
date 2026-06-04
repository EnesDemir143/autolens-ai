"""Training utilities for AutoLens AI.

Requirements: TRN-06, TRN-07
Phase decision: D-01
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
    RichProgressBar,
)
from pytorch_lightning.loggers import WandbLogger, CSVLogger


def get_device() -> str:
    """Get the best available device (MPS > CUDA > CPU).

    Returns:
        Device string: 'mps', 'cuda', or 'cpu'
    """
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def create_trainer(
    max_epochs: int = 50,
    accelerator: str | None = None,
    checkpoint_dir: str | Path = "checkpoints",
    early_stopping_patience: int = 10,
    monitor_metric: str = "val/loss",
    monitor_mode: str = "min",
    use_wandb: bool = False,
    wandb_project: str = "autolens-ai",
    wandb_name: str | None = None,
    gradient_clip_val: float | None = None,
    precision: Literal[
        "16-true", "16-mixed", "bf16-true", "bf16-mixed", "32-true", "64-true"
    ] = "32-true",
    use_ema: bool = False,
    ema_decay: float = 0.999,
    **trainer_kwargs: Any,
) -> pl.Trainer:
    """Create a Lightning Trainer with standard callbacks.

    Args:
        max_epochs: Maximum number of epochs
        accelerator: Device accelerator ('mps', 'cuda', 'cpu', or None for auto)
        checkpoint_dir: Directory for saving checkpoints
        early_stopping_patience: Patience for early stopping
        monitor_metric: Metric to monitor for checkpointing/early stopping
        monitor_mode: 'min' or 'max' for the monitored metric
        use_wandb: Whether to use W&B logging
        wandb_project: W&B project name
        wandb_name: W&B run name
        **trainer_kwargs: Additional arguments for pl.Trainer

    Returns:
        Configured Lightning Trainer
    """
    # Auto-detect device if not specified
    if accelerator is None:
        accelerator = get_device()

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Callbacks
    callbacks = [
        # Primary checkpointing - optimize for the assignment metric
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="best-{epoch:02d}-{val/f1_macro:.4f}",
            monitor="val/f1_macro",
            mode="max",
            save_top_k=3,
            save_last=True,
            auto_insert_metric_name=False,
        ),
        # Optional secondary checkpointing when early stopping / selection
        # metric differs from the primary F1-macro target. Lightning does not
        # allow multiple stateful ModelCheckpoint callbacks with the same state key.
    ]

    if monitor_metric != "val/f1_macro" or monitor_mode != "max":
        callbacks.append(
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename="best_loss-{epoch:02d}-{val/loss:.4f}",
                monitor=monitor_metric,
                mode=monitor_mode,
                save_top_k=1,
                auto_insert_metric_name=False,
            )
        )

    callbacks.extend(
        [
            # Early stopping
            EarlyStopping(
                monitor=monitor_metric,
                mode=monitor_mode,
                patience=early_stopping_patience,
                verbose=True,
            ),
            # Learning rate monitoring
            LearningRateMonitor(logging_interval="epoch"),
            # Rich progress bar
            RichProgressBar(),
        ]
    )

    # Optional: EMA
    if use_ema:
        from autolens_ai.training.ema import EMA

        callbacks.append(EMA(decay=ema_decay))
        print(f"✓ EMA enabled (decay={ema_decay})")

    # Logger
    csv_logger = CSVLogger(save_dir=str(checkpoint_dir), name="", version="")
    logger: Any = csv_logger
    if use_wandb:
        try:
            wandb_logger = WandbLogger(
                project=wandb_project,
                name=wandb_name,
                save_dir=str(checkpoint_dir.parent / "wandb"),
            )
            logger = [wandb_logger, csv_logger]
        except Exception as e:
            print(f"W&B logger failed, falling back to CSV: {e}")

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1,
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=10,
        gradient_clip_val=gradient_clip_val,
        precision=precision,
        **trainer_kwargs,
    )

    return trainer


def print_device_info() -> None:
    """Print available device information."""
    print("Device Information:")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"  MPS available: {torch.backends.mps.is_available()}")
    print(f"  Selected device: {get_device()}")
