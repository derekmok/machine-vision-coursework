import torch
import torch.nn as nn

from feature_engineering.density_map import SCALE_FACTOR
from neural_net.inference_result import InferenceResult


class ConvBlock(nn.Module):
    """A convolutional block with padding, conv, norm, activation, and dropout."""
    
    def __init__(self, in_channels, out_channels, kernel_size=5, dilation=1, dropout=0.3, num_groups=4):
        super(ConvBlock, self).__init__()
        padding_size = (kernel_size - 1) * dilation // 2
        
        self.pad = nn.ReplicationPad1d(padding_size)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=0, dilation=dilation, bias=False)
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=dropout)
    
    def forward(self, x):
        x = self.act(self.norm(self.conv(self.pad(x))))
        return self.drop(x), x.clone()


class TCNPushUpCounter(nn.Module):
    def __init__(self, input_channels=6, hidden_channels=16):
        super(TCNPushUpCounter, self).__init__()
        
        # Layer 1: Local features (kernel=5, dilation=1)
        self.block1 = ConvBlock(input_channels, hidden_channels, kernel_size=5, dilation=1)
        
        # Layer 2: Temporal context (kernel=5, dilation=4)
        self.block2 = ConvBlock(hidden_channels, hidden_channels, kernel_size=5, dilation=4)
        
        # Layer 3: Wider context (kernel=5, dilation=16)
        self.block3 = ConvBlock(hidden_channels, hidden_channels, kernel_size=5, dilation=16)

        # Head: Regress to density map (1 channel output, no padding needed for 1x1 conv)
        self.classifier = nn.Conv1d(hidden_channels, 1, kernel_size=1)
        self.final_act = nn.Softplus()

    def forward(self, x):
        # x shape: [Batch=1, Time, Channels]
        result = self.forward_with_internals(x)
        return result.total_count, result.density_map

    def forward_with_internals(self, x):
        """
        Forward pass that also returns internal layer activations.
        
        Args:
            x: Input tensor of shape [Batch, Time, Channels]
            
        Returns:
            InferenceResult with outputs and layer activations
        """
        # x shape: [Batch=1, Time, Channels]
        # Reshape to [Batch, Channels, Time] for Conv1d
        x = x.permute(0, 2, 1)
        
        x, act1_out = self.block1(x)
        x, act2_out = self.block2(x)
        x, act3_out = self.block3(x)
        
        density_map = self.final_act(self.classifier(x))
        
        total_count = torch.sum(density_map, dim=2) / SCALE_FACTOR
        
        return InferenceResult(
            total_count=total_count,
            density_map=density_map,
            act1=act1_out,
            act2=act2_out,
            act3=act3_out,
        )
