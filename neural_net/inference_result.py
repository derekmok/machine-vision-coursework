"""Dataclass for inference results with exposed internal activations."""

from dataclasses import dataclass

import torch


@dataclass
class InferenceResult:
    """Result of the forward pass with internal activations exposed.
    
    Attributes:
        total_count: Predicted push-up count [Batch, 1]
        density_map: Predicted density map [Batch, 1, Time]
        act1: Activations after layer 1 [Batch, HiddenChannels, Time]
        act2: Activations after layer 2 [Batch, HiddenChannels, Time]
        act3: Activations after layer 3 [Batch, HiddenChannels, Time]
    """
    total_count: torch.Tensor
    density_map: torch.Tensor
    act1: torch.Tensor
    act2: torch.Tensor
    act3: torch.Tensor
