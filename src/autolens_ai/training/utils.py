"""Training utilities for AutoLens AI.

Requirements: TRN-06, TRN-07
Phase decision: D-01
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor


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
        # Model checkpointing
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="{epoch:02d}-{val_loss:.4f}",
            monitor=monitor_metric,
            mode=monitor_mode,
            save_top_k=3,
            save_last=True,
            verbose=True,
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
    ]
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1,
        callbacks=callbacks,
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
