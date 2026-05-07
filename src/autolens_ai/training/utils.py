"""Training utilities for AutoLens AI.

Requirements: TRN-06, TRN-07
Phase decision: D-01
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor, RichProgressBar
from pytorch_lightning.loggers import WandbLogger


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
        # Model checkpointing - save top 3 and last
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="best-{epoch:02d}-{val_f1_macro:.4f}",
            monitor="val/f1_macro",
            mode="max",
            save_top_k=3,
            save_last=True,
            auto_insert_metric_name=False,
        ),
        # Also save best by loss
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="best_loss-{epoch:02d}-{val_loss:.4f}",
            monitor=monitor_metric,
            mode=monitor_mode,
            save_top_k=1,
            auto_insert_metric_name=False,
        ),
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
    
    # Logger
    logger: Any = True  # Default CSV logger
    if use_wandb:
        try:
            logger = WandbLogger(
                project=wandb_project,
                name=wandb_name,
                save_dir=str(checkpoint_dir.parent / "wandb"),
            )
        except Exception as e:
            print(f"W&B logger failed, falling back to CSV: {e}")
            logger = True
    
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
