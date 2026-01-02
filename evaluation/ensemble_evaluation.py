"""Ensemble model evaluation and visualization."""

import os
from typing import Optional, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from neural_net.ensemble_wrapper import EnsembleWrapper
from .metrics import compute_prediction_metrics


class EnsembleEvaluationResult(TypedDict):
    """Result dictionary from evaluate_ensemble_on_dataset."""
    predictions: np.ndarray
    rounded_predictions: np.ndarray
    targets: np.ndarray
    density_maps: list[np.ndarray]
    mae: float
    exact_match_accuracy: float
    off_by_one_accuracy: float


def evaluate_ensemble_on_dataset(
    ensemble: EnsembleWrapper,
    dataset: Dataset,
    device: Optional[torch.device] = None,
) -> EnsembleEvaluationResult:
    """Evaluate ensemble on a dataset and compute statistics.
    
    Args:
        ensemble: EnsembleWrapper model
        dataset: Dataset to evaluate on
        device: torch.device (defaults to cuda if available)
        
    Returns:
        Dictionary with predictions, targets, and metrics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ensemble = ensemble.to(device)
    ensemble.eval()

    all_predictions = []
    all_targets = []
    all_density_maps = []

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for sequences, density_maps, labels, _ in loader:
            sequences = sequences.to(device)

            # Get ensemble predictions
            predictions, density_maps = ensemble(sequences)

            all_predictions.append(predictions.cpu().squeeze().item())
            all_targets.append(labels.item())
            all_density_maps.append(density_maps.cpu().squeeze().numpy())

    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    
    # Compute metrics using shared utility
    metrics = compute_prediction_metrics(predictions, targets)

    return {
        'predictions': predictions,
        'rounded_predictions': metrics['rounded_predictions'],
        'targets': targets,
        'density_maps': all_density_maps,
        'mae': metrics['mae'],
        'exact_match_accuracy': metrics['exact_match_accuracy'],
        'off_by_one_accuracy': metrics['off_by_one_accuracy'],
    }


def plot_density_maps(results: EnsembleEvaluationResult, num_samples: int = 6) -> None:
    """Plot density maps for a handful of samples.
    
    Args:
        results: Dictionary from evaluate_ensemble_on_dataset
        num_samples: Number of samples to plot
    """
    indices = np.linspace(0, len(results['density_maps']) - 1, num_samples, dtype=int)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        density_map = results['density_maps'][idx]
        target = results['targets'][idx]
        pred = results['predictions'][idx]
        rounded_pred = results['rounded_predictions'][idx]

        ax = axes[i]
        ax.plot(density_map, color='steelblue', linewidth=1.5)
        ax.fill_between(range(len(density_map)), density_map, alpha=0.3, color='steelblue')
        ax.set_title(f'Sample {idx + 1}\nTarget: {int(target)}, Pred: {pred:.2f} (→ {int(rounded_pred)})',
                     fontsize=10)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Density')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    fig.suptitle('Ensemble Density Maps', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    plt.savefig('plots/ensemble_density_maps.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Density maps saved to 'plots/ensemble_density_maps.png'")


def plot_predicted_vs_true(results: EnsembleEvaluationResult) -> None:
    """Plot predicted counts vs true counts.
    
    Args:
        results: Dictionary from evaluate_ensemble_on_dataset
    """
    predictions = results['predictions']
    targets = results['targets']

    fig, ax = plt.subplots(figsize=(8, 8))

    # Scatter plot of predictions vs targets
    ax.scatter(targets, predictions, alpha=0.7, s=80, c='steelblue', edgecolors='white', linewidth=0.5)

    # Perfect prediction line
    min_val = min(targets.min(), predictions.min()) - 0.5
    max_val = max(targets.max(), predictions.max()) + 0.5
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Prediction')

    # Off-by-one bounds
    ax.fill_between([min_val, max_val], [min_val - 1, max_val - 1], [min_val + 1, max_val + 1],
                    alpha=0.15, color='green', label='±1 Tolerance')

    ax.set_xlabel('True Count', fontsize=12)
    ax.set_ylabel('Predicted Count', fontsize=12)
    ax.set_title('Ensemble: Predicted vs True Push-Up Counts', fontsize=14, fontweight='bold')
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')

    plt.tight_layout()
    
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    plt.savefig('plots/predicted_vs_true.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Plot saved to 'plots/predicted_vs_true.png'")
