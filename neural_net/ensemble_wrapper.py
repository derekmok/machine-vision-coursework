import torch
import torch.nn as nn
from typing import List, Dict


class EnsembleWrapper(nn.Module):
    """
    A wrapper module that combines multiple neural networks into an ensemble.
    
    Predictions are made by calling all models and averaging their outputs.
    This is useful for reducing variance and improving generalization.
    """
    
    def __init__(self, models: List[nn.Module]):
        """
        Initialize the ensemble wrapper.
        
        Args:
            models: A list of nn.Module instances to ensemble together.
                   All models should have the same input/output signature.
        """
        super(EnsembleWrapper, self).__init__()
        # Use nn.ModuleList to properly register sub-modules
        # This ensures parameters are tracked for .to(device), .eval(), etc.
        self.models = nn.ModuleList(models)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass that averages predictions from all ensemble members.
        
        Args:
            x: Input tensor to pass to all models.
            
        Returns:
            A tuple of averaged outputs from all models.
            The structure matches the output of the individual models.
        """
        # Collect outputs from all models
        all_outputs = [model(x) for model in self.models]
        
        # Assuming all models return the same structure (tuple of tensors)
        # Average each element of the tuple separately
        num_outputs = len(all_outputs[0])
        averaged_outputs = []
        
        for i in range(num_outputs):
            stacked = torch.stack([output[i] for output in all_outputs], dim=0)
            averaged = torch.mean(stacked, dim=0)
            averaged_outputs.append(averaged)
        
        return tuple(averaged_outputs)
    
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
    ) -> 'EnsembleWrapper':
        """
        Create an EnsembleWrapper from a list of models and load their weights.
        
        This is a generic factory method that can work with any model type.
        
        Args:
            models: A list of nn.Module instances to ensemble together.
                   All models should have the same input/output signature.
            state_dicts: A list of state dictionaries to load into the models.
                        Must have the same length as models.
        
        Returns:
            EnsembleWrapper with models loaded with the provided weights.
        
        Raises:
            ValueError: If the number of models and state_dicts don't match.
        
        Example:
            >>> models = [MyModel() for _ in range(5)]
            >>> state_dicts = [torch.load(f'model_{i}.pt') for i in range(5)]
            >>> ensemble = EnsembleWrapper.from_pretrained_models(models, state_dicts)
        """
        if len(models) != len(state_dicts):
            raise ValueError(
                f"Number of models ({len(models)}) must match number of "
                f"state_dicts ({len(state_dicts)})"
            )
        
        # Create the ensemble wrapper
        ensemble = EnsembleWrapper(models)
        
        # Load the weights
        ensemble._load_member_weights(state_dicts)
        
        return ensemble
