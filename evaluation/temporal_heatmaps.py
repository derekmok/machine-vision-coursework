from typing import Tuple

import matplotlib.pyplot as plt
import torch

from neural_net.data_loader import VideoDataset
from neural_net.ensemble_model import EnsembleModel


def plot_temporal_activation_heatmaps(
    ensemble: EnsembleModel,
    video_dir: str,
    filename: str,
    figsize: Tuple[float, float] = (14, 10),
) -> plt.Figure:
    """
    Plot layer-wise temporal activation heatmaps for a specific sample.
    
    Creates 3 vertically stacked heatmaps showing activations from
    act1, act2, and act3 layers with hidden channels on Y-axis and time on X-axis.
    
    Args:
        ensemble: Trained EnsembleModel
        video_dir: Path to video data directory
        filename: Name of the video file to visualize
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    dataset = VideoDataset(
        video_dir,
        feature_processor=lambda features: (features.angle_sequence,)
    )
    
    try:
        sample_idx = dataset.video_files.index(filename)
    except ValueError:
        raise ValueError(
            f"Filename '{filename}' not found in dataset. "
            f"Available files: {dataset.video_files[:5]}..."
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    angles, label = dataset[sample_idx]
    angles = angles.to(device).unsqueeze(0) # [1, Time, Channels]
    
    ensemble.eval()
    with torch.no_grad():
        result = ensemble.forward_with_internals(angles)

    # Activations shape: [Batch=1, HiddenChannels, Time] -> [HiddenChannels, Time]
    activations = [result.act1[0], result.act2[0], result.act3[0]]
    layer_names = ["Layer 1 (Local Features)", "Layer 2 (Temporal Context)", "Layer 3 (Wider Context)"]
    
    # Create figure with 3 vertically stacked subplots
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    
    for idx, (act, name) in enumerate(zip(activations, layer_names)):
        ax = axes[idx]
        act_np = act.cpu().numpy()
        
        im = ax.imshow(
            act_np,
            aspect="auto",
            cmap="viridis",
            interpolation="nearest",
        )
        
        ax.set_ylabel("Hidden Channel", fontsize=10)
        ax.set_xlabel("Time (frames)", fontsize=10)
        ax.set_title(name, fontsize=11, fontweight="bold")
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Activation", fontsize=9)
    
    fig.suptitle(
        f"Internal Activation Heatmaps\n{filename} (True count: {label})",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    
    return fig
