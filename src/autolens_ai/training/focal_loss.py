"""Focal Loss for handling class imbalance.

Alternative to class-weighted CrossEntropyLoss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.
    
    Focal Loss = -alpha * (1 - p_t)^gamma * log(p_t)
    
    where p_t is the model's estimated probability for the true class.
    
    Args:
        alpha: Weighting factor (default: 1.0)
        gamma: Focusing parameter (default: 2.0)
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.
        
        Args:
            inputs: Logits (batch_size, num_classes)
            targets: Ground truth labels (batch_size,)
            
        Returns:
            Focal loss
        """
        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        
        # Compute p_t
        p_t = torch.exp(-ce_loss)
        
        # Compute focal loss
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
