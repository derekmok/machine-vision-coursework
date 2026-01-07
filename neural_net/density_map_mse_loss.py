import torch.nn as nn

class DensityMapMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predicted_map, target_map, _):
        return nn.functional.mse_loss(predicted_map.squeeze(), target_map.squeeze())
