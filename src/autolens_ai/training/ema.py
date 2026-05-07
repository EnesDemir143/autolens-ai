"""Exponential Moving Average (EMA) callback for model weights."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback


class EMA(Callback):
    """Exponential Moving Average of model weights.
    
    Maintains a moving average of model parameters during training.
    Often leads to better generalization and more stable predictions.
    
    Args:
        decay: EMA decay rate (default: 0.999)
        validate_original_weights: If True, validate with original weights (default: False)
    """

    def __init__(self, decay: float = 0.999, validate_original_weights: bool = False):
        super().__init__()
        self.decay = decay
        self.validate_original_weights = validate_original_weights
        self.ema_model: torch.nn.Module | None = None

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Initialize EMA model."""
        self.ema_model = deepcopy(pl_module.model)
        for param in self.ema_model.parameters():
            param.detach_()

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Update EMA weights after each training batch."""
        if self.ema_model is None:
            return
        
        with torch.no_grad():
            for ema_param, model_param in zip(
                self.ema_model.parameters(),
                pl_module.model.parameters(),
            ):
                ema_param.data.mul_(self.decay).add_(
                    model_param.data, alpha=1 - self.decay
                )

    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Swap to EMA weights for validation."""
        if self.ema_model is None or self.validate_original_weights:
            return
        
        # Store original weights
        self.original_model = pl_module.model
        # Use EMA weights for validation
        pl_module.model = self.ema_model

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Restore original weights after validation."""
        if self.ema_model is None or self.validate_original_weights:
            return
        
        # Restore original weights
        pl_module.model = self.original_model

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Use EMA weights for testing."""
        if self.ema_model is None:
            return
        
        self.original_model = pl_module.model
        pl_module.model = self.ema_model

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Restore original weights after testing."""
        if self.ema_model is None:
            return
        
        pl_module.model = self.original_model
