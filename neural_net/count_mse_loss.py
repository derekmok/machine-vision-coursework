import torch.nn as nn
import torch.nn.functional as F

class CountMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predicted_map, _, true_count):
        predicted_count = predicted_map.squeeze(1).sum(dim=-1)
        return F.mse_loss(predicted_count, true_count)
