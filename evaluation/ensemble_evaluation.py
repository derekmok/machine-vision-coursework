"""Ensemble model evaluation and visualization."""

import os
from typing import Optional, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluation.metrics import compute_prediction_metrics
from neural_net.data_loader import VideoDataset
from neural_net.ensemble_model import EnsembleModel


class EnsembleEvaluationResult(TypedDict):
    """Result dictionary from evaluate_ensemble_on_dataset."""
    predictions: np.ndarray
    rounded_predictions: np.ndarray
    targets: np.ndarray
    density_maps: list[np.ndarray]
    filenames: list[str]
    mae: float
    exact_match_accuracy: float
    off_by_one_accuracy: float


def evaluate_ensemble_on_dataset(
    ensemble: EnsembleModel,
    video_dir: str,
    device: Optional[torch.device] = None,
) -> EnsembleEvaluationResult:
    """Evaluate ensemble on a dataset and compute statistics.
    
    Args:
        ensemble: EnsembleModel model
        video_dir: Directory containing video files
        device: torch.device (defaults to cuda if available)
        
    Returns:
        Dictionary with predictions, targets, and metrics
    """
    dataset = VideoDataset(
        video_dir,
        feature_processor=lambda features: (
            features.angle_sequence,
            features.density_map
        )
    )
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ensemble = ensemble.to(device)
    ensemble.eval()

    all_predictions = []
    all_targets = []
    all_density_maps = []
    all_filenames = dataset.video_files.copy()

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for sequences, density_maps, labels in loader:
            sequences = sequences.to(device)

            predictions, density_maps = ensemble(sequences)

            all_predictions.append(predictions.cpu().squeeze().item())
            all_targets.append(labels.item())
            all_density_maps.append(density_maps.cpu().squeeze().numpy())

    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    
    metrics = compute_prediction_metrics(predictions, targets)

    return {
        'predictions': predictions,
        'rounded_predictions': metrics['rounded_predictions'],
        'targets': targets,
        'density_maps': all_density_maps,
        'filenames': all_filenames,
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
        filename = os.path.splitext(results['filenames'][idx])[0]

        ax = axes[i]
        ax.plot(density_map, color='steelblue', linewidth=1.5)
        ax.fill_between(range(len(density_map)), density_map, alpha=0.3, color='steelblue')
        ax.set_title(f'{filename}\nTarget: {int(target)}, Pred: {pred:.2f} (→ {int(rounded_pred)})',
                     fontsize=10)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Density')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    fig.suptitle('Ensemble Density Maps', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs('plots', exist_ok=True)
    
    plt.savefig('plots/ensemble_density_maps.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Density maps saved to 'plots/ensemble_density_maps.png'")


def plot_wrong_predictions_density_maps(results: EnsembleEvaluationResult, num_samples: int = 6) -> None:
    """Plot density maps for wrongly predicted samples.
    
    Args:
        results: Dictionary from evaluate_ensemble_on_dataset
        num_samples: Maximum number of samples to plot
    """
    # Identify wrongly predicted samples
    rounded_predictions = results['rounded_predictions']
    targets = results['targets']
    wrong_indices = np.where(rounded_predictions != targets)[0]
    
    if len(wrong_indices) == 0:
        print("No wrong predictions found! The model is perfect!")
        return
    
    num_to_plot = min(num_samples, len(wrong_indices))
    if num_to_plot < num_samples:
        print(f"Only {num_to_plot} wrong predictions found (requested {num_samples})")
    
    if len(wrong_indices) > num_samples:
        selected_indices = wrong_indices[np.linspace(0, len(wrong_indices) - 1, num_samples, dtype=int)]
    else:
        selected_indices = wrong_indices
    
    n_cols = 3
    n_rows = (num_to_plot + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for i, idx in enumerate(selected_indices):
        density_map = results['density_maps'][idx]
        target = results['targets'][idx]
        pred = results['predictions'][idx]
        rounded_pred = results['rounded_predictions'][idx]
        error = int(rounded_pred - target)
        filename = os.path.splitext(results['filenames'][idx])[0]
        
        ax = axes[i]
        ax.plot(density_map, color='crimson', linewidth=1.5)
        ax.fill_between(range(len(density_map)), density_map, alpha=0.3, color='crimson')
        ax.set_title(f'{filename} (ERROR: {error:+d})\nTarget: {int(target)}, Pred: {pred:.2f} (→ {int(rounded_pred)})',
                     fontsize=10)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Density')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
    
    for i in range(num_to_plot, len(axes)):
        axes[i].axis('off')
    
    fig.suptitle(f'Wrong Predictions - Density Maps ({num_to_plot}/{len(wrong_indices)} errors shown)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs('plots', exist_ok=True)
    
    plt.savefig('plots/wrong_predictions_density_maps.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Wrong predictions density maps saved to 'plots/wrong_predictions_density_maps.png'")
    print(f"Total wrong predictions: {len(wrong_indices)} out of {len(targets)} samples ({len(wrong_indices)/len(targets)*100:.1f}%)")


def plot_predicted_vs_true(results: EnsembleEvaluationResult) -> None:
    """Plot predicted counts vs true counts.
    
    Args:
        results: Dictionary from evaluate_ensemble_on_dataset
    """
    predictions = results['predictions']
    targets = results['targets']

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(targets, predictions, alpha=0.7, s=80, c='steelblue', edgecolors='white', linewidth=0.5)

    min_val = min(targets.min(), predictions.min()) - 0.5
    max_val = max(targets.max(), predictions.max()) + 0.5
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Prediction')

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
    
    os.makedirs('plots', exist_ok=True)
    
    plt.savefig('plots/predicted_vs_true.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Plot saved to 'plots/predicted_vs_true.png'")
