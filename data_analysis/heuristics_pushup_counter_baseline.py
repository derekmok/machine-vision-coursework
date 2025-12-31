# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Heuristic based push-up counter
#
# This file uses a signal processing approach instead of deep learning.
# The key insight is that during a push-up, the shoulder elbow angle oscillates
# periodically. We detect these oscillations using peak detection.
#
# This serves as a BASELINE for comparison with deep learning approaches.

# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


from data_loader import VideoDataset

def get_dataloaders(video_dir, batch_size=4, val_split=0.2):
    """Create train and validation dataloaders for landmark sequences.
    
    Args:
        video_dir: Path to directory containing video files.
        batch_size: Number of samples per batch.
        val_split: Fraction of data to use for validation.
        
    Returns:
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
    """
    
    # First, create a dataset without transforms to determine the split indices
    full_dataset = VideoDataset(video_dir)
    
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    # Get the indices for train and validation splits
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(full_dataset), generator=generator).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create separate datasets for train (with augmentation) and val (without)
    train_dataset = VideoDataset(video_dir)
    val_dataset = VideoDataset(video_dir)
    
    # Use Subset to apply the split indices
    from torch.utils.data import Subset
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(f"Train: {len(train_subset)} videos, Val: {len(val_subset)} videos")
    
    return train_loader, val_loader


video_dir = '../video-data'

# %%
from count_pushups_heuristic import (
    HeuristicPushupCounter,
    evaluate_on_dataset,
    grid_search_parameters,
)


# %% [markdown] id="3Bou97f8czAu"
# # Evaluate the Heuristic Model
#
# Since this is a heuristic approach (not learned), there is no "training" phase.
# Instead, we evaluate the signal processing based counter on our data and tune 
# the hyperparameters using grid search.

# %%
def run_heuristic_evaluation(
    video_dir: str = './video-data',
):
    """Evaluate the heuristic push-up counter on the dataset.
    
    Args:
        video_dir: Path to the video data directory.
    """
    train_loader, val_loader = get_dataloaders(video_dir, batch_size=1, val_split=0.2)
    
    print("Running grid search for optimal parameters...")
    
    # Use training set for parameter tuning
    search_results = grid_search_parameters(
        train_loader,
        smoothing_windows=[11, 15, 21, 31],
        min_prominences=[0.03, 0.05, 0.08, 0.11],
        min_distances=[5, 10, 15, 20, 30],
    )
    
    print(f"Best parameters: {search_results['best_params']}")
    print(f"Best MAE on training set: {search_results['best_mae']:.4f}")
    
    counter = HeuristicPushupCounter(**search_results['best_params'])
    
    # Evaluate on training set
    print("Training Set Results:")
    train_metrics = evaluate_on_dataset(train_loader, counter)
    print(f"MAE: {train_metrics['mae']:.4f}")
    print(f"Exact Accuracy: {train_metrics['accuracy'] * 100:.2f}%")
    print(f"Within-1 Accuracy: {train_metrics['within_1_accuracy'] * 100:.2f}%")
    
    # Per-sample breakdown
    print("\nPer-sample breakdown:")
    for pred, label in zip(train_metrics['predictions'], train_metrics['labels']):
        status = "✓" if pred == label else "✗"
        print(f"{status} Predicted: {pred}, Actual: {label}")
    
    # Evaluate on validation set
    print()
    print("Validation Set Results:")
    val_metrics = evaluate_on_dataset(val_loader, counter)
    print(f"MAE: {val_metrics['mae']:.4f}")
    print(f"Exact Accuracy: {val_metrics['accuracy'] * 100:.2f}%")
    print(f"Within-1 Accuracy: {val_metrics['within_1_accuracy'] * 100:.2f}%")
    
    # Per-sample breakdown
    print("\nPer-sample breakdown:")
    for pred, label in zip(val_metrics['predictions'], val_metrics['labels']):
        status = "✓" if pred == label else "✗"
        print(f"{status} Predicted: {pred}, Actual: {label}")
    
    print()
    print("EVALUATION COMPLETE")
    
    return counter, train_metrics, val_metrics


# %% colab={"base_uri": "https://localhost:8080/"} id="lHI68u4XhCGB" outputId="c28ba0d4-cb87-4076-eb13-ab93d597ef10"
# Run the evaluation
counter, train_metrics, val_metrics = run_heuristic_evaluation(
    video_dir='../video-data',
)

# %% [markdown]
# Visualize the signal processing on sample videos to understand how the heuristic works.

# %%
import matplotlib.pyplot as plt

# Visualize detection on a few sample videos
def visualize_sample_detections(video_dir: str = './video-data', num_samples: int = 3):
    """Visualize the push-up detection on sample videos."""
    dataset = VideoDataset(video_dir)
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(14, 4 * num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for i in range(min(num_samples, len(dataset))):
        landmarks, label, _ = dataset[i]
        counter.plot_debug(
            landmarks, 
            title=f"Video {i+1} (Ground Truth: {label} push-ups)",
            ax=axes[i]
        )
    
    plt.tight_layout()
    plt.savefig('pushup_detection_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved visualization to 'pushup_detection_visualization.png'")

visualize_sample_detections()
