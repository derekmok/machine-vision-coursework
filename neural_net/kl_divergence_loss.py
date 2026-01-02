"""KL Divergence-based loss for density map regression.

This module provides a loss function that decouples shape learning from scale learning
by using KL divergence for matching the temporal distribution of push-ups and a
separate count loss for ensuring the predicted density sums to the true count.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KLDivergenceDensityLoss(nn.Module):
    """KL Divergence loss for density map regression with count supervision.
    
    This loss decouples shape learning from scale learning:
    - Shape Loss (KL Divergence): Both predicted and target density maps are
      normalized to probability distributions, then KL divergence measures
      how well the predicted temporal distribution matches the target.
    - Count Loss: L1 loss between sum(predicted_map) and true_count ensures
      the predicted density integrates to the correct count.
    
    Args:
        lambda_count: Weight for the count loss term (default: 1.0)
        eps: Small epsilon for numerical stability (default: 1e-8)
    """
    
    def __init__(self, lambda_count: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.lambda_count = lambda_count
        self.eps = eps

    def forward(self, predicted_map, target_map, true_count):
        """Compute the combined KL divergence and count loss.
        
        Args:
            predicted_map: [Batch, 1, Time] - Network output (positive values from Softplus)
            target_map: [Batch, Time] or [Batch, 1, Time] - Gaussian pseudo-labels
            true_count: [Batch] - Integer ground truth count
            
        Returns:
            Combined loss scalar
        """
        # Flatten to [Batch, Time]
        pred = predicted_map.squeeze(1)  # [Batch, Time]
        target = target_map.squeeze()    # [Batch, Time]
        
        # Handle edge case: if target is all zeros (no pushups in video)
        target_sum = target.sum(dim=-1, keepdim=True)
        has_pushups = target_sum.squeeze() > self.eps
        
        # --- Shape Loss: KL Divergence ---
        # Only compute KL for samples with pushups (non-zero target distribution)
        if has_pushups.any():
            # Select samples with pushups
            pred_with_pushups = pred[has_pushups]
            target_with_pushups = target[has_pushups]
            target_sum_with_pushups = target_sum[has_pushups]
            
            # Normalize to probability distributions
            pred_dist = pred_with_pushups / (pred_with_pushups.sum(dim=-1, keepdim=True) + self.eps)
            target_dist = target_with_pushups / (target_sum_with_pushups + self.eps)
            
            # Add small epsilon for numerical stability in log
            pred_dist = pred_dist + self.eps
            target_dist = target_dist + self.eps
            
            # Renormalize after adding epsilon
            pred_dist = pred_dist / pred_dist.sum(dim=-1, keepdim=True)
            target_dist = target_dist / target_dist.sum(dim=-1, keepdim=True)
            
            # KL(target || pred) = sum(target * log(target / pred))
            # F.kl_div expects log-probabilities for first argument
            kl_loss = F.kl_div(
                pred_dist.log(),
                target_dist,
                reduction='batchmean'
            )
        else:
            # No pushups in any sample, skip KL loss
            kl_loss = torch.tensor(0.0, device=pred.device)
        
        # --- Count Loss: Sum should equal true count ---
        predicted_count = pred.sum(dim=-1)  # [Batch]
        count_loss = F.l1_loss(predicted_count, true_count.float())
        
        return kl_loss + self.lambda_count * count_loss
