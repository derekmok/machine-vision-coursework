"""PCA trajectory embedding visualization

This module visualizes how the hidden state of the TCN evolves over time
by projecting the 16-dimensional hidden states to 2D using PCA.
"""

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from neural_net.data_loader import VideoDataset
from neural_net.ensemble_model import EnsembleModel


def compute_pca_trajectory(
    activations: torch.Tensor,
) -> np.ndarray:
    """
    Project 16D activations to 2D using PCA.
    
    Args:
        activations: Tensor of shape [Time, 16]
        
    Returns:
        2D trajectory array of shape [Time, 2]
    """
    act_np = activations.cpu().numpy()
    
    pca = PCA(n_components=2)
    trajectory_2d = pca.fit_transform(act_np)
    
    return trajectory_2d


def _plot_pca_trajectory(
    trajectory: np.ndarray,
    ax: plt.Axes,
    title: str = "PCA Trajectory",
    cmap: str = "viridis",
) -> None:
    """
    Plot a single PCA trajectory with points connected in time order.
    
    Args:
        trajectory: 2D array of shape [Time, 2]
        ax: Matplotlib axes to plot on
        title: Plot title
        cmap: Colormap for time progression
    """
    n_points = len(trajectory)
    colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, n_points))
    
    for i in range(n_points - 1):
        ax.plot(
            [trajectory[i, 0], trajectory[i + 1, 0]],
            [trajectory[i, 1], trajectory[i + 1, 1]],
            color=colors[i],
            linewidth=1.5,
            alpha=0.7,
        )
    
    scatter = ax.scatter(
        trajectory[:, 0],
        trajectory[:, 1],
        c=np.arange(n_points),
        cmap=cmap,
        s=30,
        edgecolors="white",
        linewidth=0.5,
        zorder=5,
    )
    
    ax.scatter(
        trajectory[0, 0], trajectory[0, 1],
        color="green", s=100, marker="o", edgecolors="white",
        linewidth=2, zorder=10, label="Start"
    )
    ax.scatter(
        trajectory[-1, 0], trajectory[-1, 1],
        color="red", s=100, marker="s", edgecolors="white",
        linewidth=2, zorder=10, label="End"
    )
    
    ax.set_xlabel("PC1", fontsize=10)
    ax.set_ylabel("PC2", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper right", fontsize=8)
    
    cbar = plt.colorbar(scatter, ax=ax, label="Time (frames)")
    cbar.ax.tick_params(labelsize=8)


def plot_pca_trajectories_for_samples(
    ensemble: EnsembleModel,
    video_dir: str,
    video_filenames: list[str],
    figsize: Tuple[float, float] = (12, 5),
) -> plt.Figure:
    """
    Plot PCA trajectories for specified video samples.
    
    Uses the first ensemble member to extract activations.
    
    Args:
        ensemble: Trained EnsembleModel
        video_dir: Path to video data directory
        video_filenames: List of video filenames to visualize
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    dataset = VideoDataset(
        video_dir,
        feature_processor=lambda features: (features.angle_sequence,)
    )
    
    filename_to_idx = {f: i for i, f in enumerate(dataset.video_files)}
    
    ensemble.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_samples = len(video_filenames)
    
    fig, axes = plt.subplots(1, num_samples, figsize=figsize)
    if num_samples == 1:
        axes = [axes]
    
    for plot_idx, filename in enumerate(video_filenames):
        if filename not in filename_to_idx:
            raise ValueError(f"Video file '{filename}' not found in dataset")
        sample_idx = filename_to_idx[filename]
        
        angles, label = dataset[sample_idx]
        angles = angles.to(device).unsqueeze(0) # [1, Time, Channels]
        
        # Extract activations using forward_with_internals
        with torch.no_grad():
            result = ensemble.forward_with_internals(angles)

        # act3 shape: [Batch=1, HiddenChannels, Time] -> [Time, HiddenChannels]
        activations = result.act3[0].permute(1, 0)
        
        trajectory = compute_pca_trajectory(activations)
        
        title = f"{filename}\n(True count: {label})"
        _plot_pca_trajectory(trajectory, axes[plot_idx], title=title)
    
    fig.suptitle(
        "PCA Trajectory of Hidden States (after Layer 3)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    
    return fig
