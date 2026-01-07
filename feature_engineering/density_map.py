import numpy as np
import torch

SCALE_FACTOR = 8.


def generate_gaussian_density_map(valleys: np.ndarray, num_frames: int) -> torch.Tensor:
    """Generate a gaussian density map from detected valley positions.

    Args:
        valleys: Array of valley indices (bottom positions of push-ups).
        num_frames: Total number of frames in the sequence.

    Returns:
        Tensor of shape (num_frames,) containing the gaussian density map.
    """
    density_map = np.zeros(num_frames, dtype=np.float64)

    if len(valleys) <= 0:
        return torch.tensor(density_map, dtype=torch.float32)

    sigma = 3.0

    # Each Gaussian is normalized to sum to 1, so the total sum equals the count
    frame_indices = np.arange(num_frames)
    for valley_idx in valleys:
        gaussian = np.exp(-0.5 * ((frame_indices - valley_idx) / sigma) ** 2)
        gaussian_normalized = gaussian / (gaussian.sum() + 1e-8)
        density_map += gaussian_normalized

    return torch.tensor(density_map, dtype=torch.float32) * SCALE_FACTOR
