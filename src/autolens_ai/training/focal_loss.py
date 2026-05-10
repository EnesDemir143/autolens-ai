"""Focal Loss for handling class imbalance.

Alternative to class-weighted CrossEntropyLoss.
"""

from __future__ import annotations

from typing import Any

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
        alpha: float | torch.Tensor = 1.0,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        if isinstance(alpha, torch.Tensor):
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = float(alpha)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.
        
        Args:
            inputs: Logits (batch_size, num_classes)
            targets: Ground truth labels (batch_size,)
            
        Returns:
            Focal loss
        """
        num_classes = inputs.size(1)
        log_probs = F.log_softmax(inputs, dim=1)
        probs = log_probs.exp()

        with torch.no_grad():
            true_dist = torch.zeros_like(inputs)
            if self.label_smoothing > 0:
                true_dist.fill_(self.label_smoothing / (num_classes - 1))
                true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            else:
                true_dist.scatter_(1, targets.unsqueeze(1), 1.0)

        # A tensor alpha provides per-class weighting (class-weighted focal loss),
        # while a float alpha is the standard scalar form.
        alpha: Any = self.alpha
        if isinstance(alpha, torch.Tensor):
            alpha_t = alpha.to(device=inputs.device, dtype=inputs.dtype).unsqueeze(0)
        else:
            alpha_t = torch.as_tensor(alpha, device=inputs.device, dtype=inputs.dtype)

        focal_loss = -alpha_t * (1 - probs) ** self.gamma * true_dist * log_probs
        focal_loss = focal_loss.sum(dim=1)
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
