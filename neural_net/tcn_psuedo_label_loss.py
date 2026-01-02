import torch
import torch.nn as nn

class PseudoLabelLoss(nn.Module):
    def __init__(self, use_count_consistency=True, lambda_count=1.0):
        super().__init__()
        self.mse = nn.L1Loss()
        self.use_count_consistency = use_count_consistency
        self.lambda_count = lambda_count

    def forward(self, predicted_map, target_map, true_count):
        """
        predicted_map: [Batch=1, Channels=1, Time] - Output from TCN
        target_map:    [Batch=1, Channels=1, Time] - The Gaussian Pseudo-labels
        true_count:    [Batch=1] - Actual integer count (e.g., 10)
        """
        
        # 1. Map Loss: Make the prediction look like the Gaussian bumps
        # We squeeze to shape [Time] since Batch=1
        loss_map = self.mse(predicted_map.squeeze(), target_map.squeeze())
        
        # 2. (Optional) Count Consistency Loss
        # Ensures that even if the shape is slightly off, the sum is exact.
        if self.use_count_consistency:
            predicted_count = torch.sum(predicted_map)
            loss_count = torch.abs(predicted_count - true_count)
            
            return loss_map + (self.lambda_count * loss_count)
        
        return loss_map
