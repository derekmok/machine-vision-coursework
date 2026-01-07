from typing import List, Dict

import torch
import torch.nn as nn

from neural_net.inference_result import InferenceResult


class EnsembleModel(nn.Module):
    """
    A wrapper model that combines multiple models into an ensemble.
    
    Predictions are made by calling all models and averaging their outputs.
    """
    
    def __init__(self, models: List[nn.Module]):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass that averages predictions from all ensemble members.
        
        Args:
            x: Input tensor to pass to all models.
            
        Returns:
            A tuple of (total_count, density_map) averaged from all models.
        """
        result = self.forward_with_internals(x)
        return result.total_count, result.density_map
    
    def forward_with_internals(self, x: torch.Tensor):
        """
        Forward pass with averaged outputs and internals from all ensemble members.
        
        Args:
            x: Input tensor to pass to all models.
            
        Returns:
            InferenceResult with all fields averaged across ensemble members.
        """
        all_results = [model.forward_with_internals(x) for model in self.models]
        
        return InferenceResult(
            total_count=torch.stack([r.total_count for r in all_results], dim=0).mean(dim=0),
            density_map=torch.stack([r.density_map for r in all_results], dim=0).mean(dim=0),
            act1=torch.stack([r.act1 for r in all_results], dim=0).mean(dim=0),
            act2=torch.stack([r.act2 for r in all_results], dim=0).mean(dim=0),
            act3=torch.stack([r.act3 for r in all_results], dim=0).mean(dim=0),
        )
    
    def _load_member_weights(self, state_dicts: List[Dict[str, torch.Tensor]]) -> None:
        """
        Load weights into each ensemble member from a list of state dictionaries.
        
        This is useful for hydrating the ensemble after training individual models
        (e.g., from k-fold cross-validation).
        
        Args:
            state_dicts: A list of state dictionaries, one per ensemble member.
                        The list must have the same length as the number of models.
        """
        for model, state_dict in zip(self.models, state_dicts):
            model.load_state_dict(state_dict)
    
    def __len__(self) -> int:
        """Return the number of models in the ensemble."""
        return len(self.models)
    
    @staticmethod
    def from_pretrained_models(
        models: List[nn.Module],
        state_dicts: List[Dict[str, torch.Tensor]]
    ) -> 'EnsembleModel':
        """
        Create an EnsembleModel from a list of models and load their weights.
        
        Args:
            models: A list of nn.Module instances to ensemble together.
            state_dicts: A list of state dictionaries to load into the models.
        
        Returns:
            EnsembleModel with models loaded with the provided weights.
        
        """
        ensemble = EnsembleModel(models)
        ensemble._load_member_weights(state_dicts)
        return ensemble
