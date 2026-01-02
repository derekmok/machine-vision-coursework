import torch
import torch.nn as nn

class GRUPushUpCounter(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=32, num_layers=2):
        super(GRUPushUpCounter, self).__init__()
        
        # Bi-directional GRU
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True, 
            dropout=0.5 # High dropout for small data
        )
        
        # Regressor
        # Input is hidden_dim * 2 because of bidirectionality
        self.fc1 = nn.Linear(hidden_dim * 2, 16)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(16, 1) # Output count

    def forward(self, x):
        # x shape: (Batch=1, Seq_Len, Features)
        
        # 1. Run GRU
        # out shape: (Batch, Seq_Len, Hidden_Dim * 2)
        out, _ = self.gru(x)
        
        # 2. Global Average Pooling
        # Average across the Sequence dimension (dim=1)
        # This condenses the "video" into a single feature vector
        out = torch.mean(out, dim=1) 
        
        # 3. Regression Head
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out
