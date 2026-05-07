"""PyTorch Lightning module for AutoLens AI training.

Requirements: TRN-06, TRN-07
Phase decision: D-01
"""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, F1Score, Precision, Recall
from torchmetrics.classification import MulticlassConfusionMatrix

from autolens_ai.models import create_model


class AutoLensClassifier(pl.LightningModule):
    """Lightning module for AutoLens AI classification.
    
    Supports:
    - MPS/CPU device handling
    - Class-weighted loss
    - Comprehensive metrics (Accuracy, Precision, Recall, F1, Confusion Matrix)
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int = 8,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        class_weights: torch.Tensor | None = None,
        pretrained: bool = True,
        label_smoothing: float = 0.0,
        use_focal_loss: bool = False,
        focal_alpha: float = 1.0,
        focal_gamma: float = 2.0,
    ):
        """Initialize Lightning module.
        
        Args:
            model_name: Model architecture name
            num_classes: Number of classes
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            class_weights: Optional class weights for loss
            pretrained: Whether to use pretrained weights
            label_smoothing: Label smoothing factor (0.0 = no smoothing, 0.1 = 10% smoothing)
            use_focal_loss: Use Focal Loss instead of CrossEntropy
            focal_alpha: Focal loss alpha parameter
            focal_gamma: Focal loss gamma parameter
        """
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])
        
        # Create model
        self.model = create_model(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
        )
        
        # Loss function
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing
        self.use_focal_loss = use_focal_loss
        if class_weights is not None:
            self.register_buffer("_class_weights", class_weights)
        
        # Focal loss
        if use_focal_loss:
            from autolens_ai.training.focal_loss import FocalLoss
            self.focal_loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        
        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        
        self.val_f1_macro = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.val_f1_weighted = F1Score(task="multiclass", num_classes=num_classes, average="weighted")
        self.val_precision = Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.val_recall = Recall(task="multiclass", num_classes=num_classes, average="macro")
        
        self.test_f1_macro = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.test_f1_weighted = F1Score(task="multiclass", num_classes=num_classes, average="weighted")
        self.test_precision = Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.test_recall = Recall(task="multiclass", num_classes=num_classes, average="macro")
        self.test_confusion = MulticlassConfusionMatrix(num_classes=num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)
    
    def _compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss with optional class weights, label smoothing, or focal loss."""
        if self.use_focal_loss:
            return self.focal_loss_fn(logits, targets)
        elif self.class_weights is not None:
            return F.cross_entropy(
                logits, 
                targets, 
                weight=self._class_weights,  # type: ignore[arg-type]
                label_smoothing=self.label_smoothing,
            )
        return F.cross_entropy(logits, targets, label_smoothing=self.label_smoothing)
    
    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        images, labels = batch
        logits = self(images)
        loss = self._compute_loss(logits, labels)
        
        # Metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, labels)
        
        # Logging
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Validation step."""
        images, labels = batch
        logits = self(images)
        loss = self._compute_loss(logits, labels)
        
        # Metrics
        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, labels)
        self.val_f1_macro(preds, labels)
        self.val_f1_weighted(preds, labels)
        self.val_precision(preds, labels)
        self.val_recall(preds, labels)
        
        # Logging
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1_macro", self.val_f1_macro, on_step=False, on_epoch=True)
        self.log("val/f1_weighted", self.val_f1_weighted, on_step=False, on_epoch=True)
        self.log("val/precision", self.val_precision, on_step=False, on_epoch=True)
        self.log("val/recall", self.val_recall, on_step=False, on_epoch=True)
    
    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Test step."""
        images, labels = batch
        logits = self(images)
        loss = self._compute_loss(logits, labels)
        
        # Metrics
        preds = torch.argmax(logits, dim=1)
        self.test_acc(preds, labels)
        self.test_f1_macro(preds, labels)
        self.test_f1_weighted(preds, labels)
        self.test_precision(preds, labels)
        self.test_recall(preds, labels)
        self.test_confusion(preds, labels)
        
        # Logging
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True)
        self.log("test/f1_macro", self.test_f1_macro, on_step=False, on_epoch=True)
        self.log("test/f1_weighted", self.test_f1_weighted, on_step=False, on_epoch=True)
        self.log("test/precision", self.test_precision, on_step=False, on_epoch=True)
        self.log("test/recall", self.test_recall, on_step=False, on_epoch=True)
    
    def configure_optimizers(self) -> Any:
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,  # type: ignore[attr-defined]
            weight_decay=self.hparams.weight_decay,  # type: ignore[attr-defined]
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
