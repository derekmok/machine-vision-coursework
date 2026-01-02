import torch
import torch.nn as nn

class TCNPushUpCounter(nn.Module):
    def __init__(self, input_channels=6, hidden_channels=16):
        super(TCNPushUpCounter, self).__init__()
        
        # Layer 1: Local features (Receptive Field: 5 frames)
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=hidden_channels, kernel_size=5, padding=2, bias=False)
        self.norm1 = nn.GroupNorm(num_groups=4, num_channels=hidden_channels)
        self.act1  = nn.SiLU()
        self.drop1 = nn.Dropout(p=0.3)

        # Layer 2: Temporal context (Dilation=2 -> Receptive Field increases)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=4, dilation=2, bias=False)
        self.norm2 = nn.GroupNorm(num_groups=4, num_channels=hidden_channels)
        self.act2  = nn.SiLU()
        self.drop2 = nn.Dropout(p=0.3)
        
        # Layer 3: Wider context (Dilation=4)
        # This allows the net to see "cycles" (down and up motion)
        self.conv3 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=8, dilation=4, bias=False)
        self.norm3 = nn.GroupNorm(num_groups=4, num_channels=hidden_channels)
        self.act3  = nn.SiLU()
        self.drop3 = nn.Dropout(p=0.3)

        # Head: Regress to density map (1 channel output)
        self.classifier = nn.Conv1d(hidden_channels, 1, kernel_size=1)
        # Force positive output because push-up counts cannot be negative
        self.final_act = nn.Softplus()

    def forward(self, x):
        # x shape: [Batch=1, Time, Channels]
        # Reshape to [Batch, Channels, Time] for Conv1d
        x = x.permute(0, 2, 1)
        
        x = self.drop1(self.act1(self.norm1(self.conv1(x))))
        x = self.drop2(self.act2(self.norm2(self.conv2(x))))
        x = self.drop3(self.act3(self.norm3(self.conv3(x))))
        
        # Density map
        density_map = self.final_act(self.classifier(x))
        
        # Sum density map to get total count
        # We return both for potential visualization/debugging
        total_count = torch.sum(density_map, dim=2)
        
        return total_count, density_map
