"""
Heuristic Push-up Counter using Signal Processing.

This module provides a baseline push-up counting approach using classical signal
processing techniques. It serves as a comparison baseline for deep learning approaches.

Key Approach:
1. Extract the primary elbow angle signal from the raw landmarks (whichever elbow has more movement).
2. Apply a two-stage smoothing process to the landmark sequence:
   - Stage 1: Median filter to remove outliers and spikes from pose detection failures.
   - Stage 2: Savitzky-Golay filter to smooth jitter while preserving local peaks.
3. Detect peaks/valleys on the smoothed version of the extracted signal.
4. Count push-ups based on the number of detected valleys (bottom positions).
"""

from dataclasses import dataclass
from typing import Optional

import itertools
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter, find_peaks
from tqdm.auto import tqdm
from feature_engineering.constants import LEFT_ELBOW_ANGLE_IDX, RIGHT_ELBOW_ANGLE_IDX


@dataclass
class HeuristicParameters:
    """Parameters for HeuristicPushupCounter.
    
    Attributes:
        smoothing_window: Window size for Savitzky-Golay filter (must be odd)
        poly_order: Polynomial order for smoothing
        min_prominence: Minimum prominence threshold for peak detection
        min_distance: Minimum distance between detected peaks
        median_filter_size: Kernel size for median filter to remove outliers (must be odd)
    """
    smoothing_window: int
    poly_order: int
    min_prominence: float
    min_distance: int
    median_filter_size: int


@dataclass
class GridSearchResults:
    """Results from grid search parameter optimization.
    
    Attributes:
        best_params: Best parameter configuration
        best_mae: Best Mean Absolute Error achieved with best_params
        all_results: List of all configurations tested and their metrics
    """
    best_params: HeuristicParameters
    best_mae: float
    all_results: list


@dataclass
class CountPushupResults:
    """Results from heuristic push-up counting.
    
    Attributes:
        count: Number of push-ups detected
        raw_signal: Original extracted signal (from input landmarks)
        smoothed_signal: Smoothed signal after filtering
        valleys: Indices of detected valleys (bottom positions)
        peaks: Indices of detected peaks (top positions)
        elbow_index: Index of the elbow angle used (0=left, 1=right)
        smoothed_landmarks: Smoothed landmarks sequence of shape (T, 6)
    """
    count: int
    raw_signal: np.ndarray
    smoothed_signal: np.ndarray
    valleys: np.ndarray
    peaks: np.ndarray
    elbow_index: int
    smoothed_landmarks: np.ndarray


class HeuristicPushupCounter:
    """Signal processing based push-up counter.
    
    This class counts push-ups by detecting oscillations in the elbow angle
    extracted from video frames using MediaPipe.
    
    The key insight is that during a push-up, the elbow angle oscillates
    periodically. By tracking the elbow angle with more movement and
    detecting the valleys in this signal, we can count push-ups.
    """

    def __init__(self, params: HeuristicParameters):
        self.smoothing_window = params.smoothing_window
        self.poly_order = params.poly_order
        self.min_prominence = params.min_prominence
        self.min_distance = params.min_distance
        self.median_filter_size = params.median_filter_size

    def _extract_signal(self, landmarks: np.ndarray) -> tuple[np.ndarray, int]:
        """Extract the primary push-up signal from angle features.
        
        Uses the elbow angle with more movement (higher range) as the primary signal.
        The elbow angle oscillates as the person moves up and down during push-ups.
        
        Args:
            landmarks: Numpy array of shape (T, 6) containing angle features
                in degrees for T frames.
                
        Returns:
            Tuple of (signal, elbow_index):
            - signal: 1D numpy array of shape (T,) containing the selected elbow angle signal.
            - elbow_index: Index of the elbow used (0=left, 1=right). Caller can use this
              to extract the raw signal from the original input landmarks.
        """
        left_elbow_angle = landmarks[:, LEFT_ELBOW_ANGLE_IDX]
        right_elbow_angle = landmarks[:, RIGHT_ELBOW_ANGLE_IDX]

        left_range = np.ptp(left_elbow_angle)
        right_range = np.ptp(right_elbow_angle)
        
        if left_range >= right_range:
            return left_elbow_angle, LEFT_ELBOW_ANGLE_IDX
        else:
            return right_elbow_angle, RIGHT_ELBOW_ANGLE_IDX
    
    def _smooth_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """Apply two-stage smoothing to the entire landmarks sequence.
        
        1. Median filter removes outliers/spikes from pose detection failures or incorrect landmark positions.
        2. Savitzky-Golay filter smooths high-frequency jitter while preserving signal peaks and timing.
        
        Args:
            landmarks: Numpy array of shape (T, 6) containing angle features.
            
        Returns:
            Numpy array of shape (T, 6) with smoothed angle features.
        """
        smoothed = median_filter(landmarks, size=(self.median_filter_size, 1))
        
        smoothed = savgol_filter(smoothed, self.smoothing_window, self.poly_order, axis=0)
        
        return smoothed
    
    def _detect_peaks(self, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Detect peaks and valleys in the push-up signal.
        
        In the push-up motion:
        - Valleys = bottom position (arms bent, body low)
        - Peaks = top position (arms extended, body high)
        
        We detect valleys by finding peaks in the negated signal.
        
        Args:
            signal: 1D numpy array of the (smoothed) signal.
            
        Returns:
            Tuple of (valleys, peaks) where each is an array of indices.
        """
        peaks, _ = find_peaks(
            signal,
            prominence=self.min_prominence,
            distance=self.min_distance
        )
        
        valleys, _ = find_peaks(
            -signal,
            prominence=self.min_prominence,
            distance=self.min_distance
        )
        
        return valleys, peaks

    def count_pushups(self, landmarks: torch.Tensor) -> CountPushupResults:
        """Count push-ups based on heuristics.
        
        Process:
        1. Determine which elbow to use based on raw landmarks. This determines the signal we use
        2. Smooth the entire landmarks sequence
        3. Detect peaks/valleys on the smoothed version of the selected signal
        
        Args:
            landmarks: torch.Tensor of shape (T, 6).
            
        Returns:
            CountPushupResults See documentation of CountPushupReesults
        """
        landmarks = landmarks.numpy()

        _, elbow_index = self._extract_signal(landmarks)
        
        smoothed_landmarks = self._smooth_landmarks(landmarks)
        
        raw_signal = landmarks[:, elbow_index]
        smoothed_signal = smoothed_landmarks[:, elbow_index]
        
        if len(smoothed_signal) < self.smoothing_window:
            return CountPushupResults(
                count=0,
                raw_signal=raw_signal,
                smoothed_signal=smoothed_signal,
                valleys=np.array([]),
                peaks=np.array([]),
                elbow_index=elbow_index,
                smoothed_landmarks=smoothed_landmarks,
            )
        
        # Detect peaks on the smoothed signal
        valleys, peaks = self._detect_peaks(smoothed_signal)
        
        return CountPushupResults(
            count=len(valleys),
            raw_signal=raw_signal,
            smoothed_signal=smoothed_signal,
            valleys=valleys,
            peaks=peaks,
            elbow_index=elbow_index,
            smoothed_landmarks=smoothed_landmarks,
        )
    
    def plot_pushup_results(
        self, 
        landmarks: torch.Tensor, 
        title: str = "Push-up Detection",
        ax: Optional[plt.Axes] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot the signal with detected peaks for visualization.
        
        Args:
            landmarks: torch.Tensor of shape (T, 6).
            title: Plot title.
            ax: Optional matplotlib axes to plot on.
            save_path: If provided, save the figure to this path.
            
        Returns:
            The matplotlib Figure object.
        """
        results = self.count_pushups(landmarks)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 5))
        else:
            fig = ax.get_figure()
            
        frames = np.arange(len(results.raw_signal))
        
        # Plot raw signal
        ax.plot(frames, results.raw_signal, 'b-', alpha=0.3, label='Raw Signal')
        
        # Plot smoothed signal
        ax.plot(frames, results.smoothed_signal, 'b-', linewidth=2, label='Smoothed Signal')
        
        # Mark valleys (bottom positions = push-ups)
        if len(results.valleys) > 0:
            ax.scatter(
                results.valleys,
                results.smoothed_signal[results.valleys],
                c='red', s=100, marker='v', zorder=5,
                label=f"Valleys (Push-ups: {results.count})"
            )
        
        # Mark peaks (top positions)
        if len(results.peaks) > 0:
            ax.scatter(
                results.peaks,
                results.smoothed_signal[results.peaks],
                c='green', s=80, marker='^', zorder=5,
                label='Peaks (Top position)'
            )
        
        ax.set_xlabel('Frame')
        ax.set_ylabel('Elbow Angle (degrees)')
        ax.set_title(f"{title} - Detected: {results.count}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            
        return fig


def evaluate_on_dataset(
    data_loader,
    counter: HeuristicPushupCounter,
) -> dict:
    """Evaluate the heuristic counter on a dataset.
    
    Args:
        data_loader: PyTorch DataLoader yielding (landmarks, labels, lengths) tuples.
        counter: HeuristicPushupCounter instance.
        
    Returns:
        Dictionary with evaluation metrics:
            - predictions: List of predicted counts
            - labels: List of actual counts
            - mae: Mean Absolute Error
            - accuracy: Exact match rate
            - within_1_accuracy: Rate where |pred - actual| <= 1
    """
    predictions = []
    labels = []
    
    for batch_landmarks, _, batch_labels, batch_lengths in data_loader:
        for i in range(len(batch_labels)):
            length = batch_lengths[i].item()
            landmarks = batch_landmarks[i, :length]
            
            pred = counter.count_pushups(landmarks).count
            actual = batch_labels[i].item()
            
            predictions.append(pred)
            labels.append(actual)
    
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # Calculate metrics
    mae = np.mean(np.abs(predictions - labels))
    accuracy = np.mean(predictions == labels)
    within_1 = np.mean(np.abs(predictions - labels) <= 1)
    
    return {
        'predictions': predictions.tolist(),
        'labels': labels.tolist(),
        'mae': mae,
        'accuracy': accuracy,
        'within_1_accuracy': within_1,
    }


def grid_search_parameters(
    data_loader,
    smoothing_windows: list[int] = [11, 15, 21, 31],
    min_prominences: list[float] = [5.0, 10.0, 15.0, 20.0],
    min_distances: list[int] = [5, 10, 15, 20, 30],
    median_filter_sizes: list[int] = [3],
    poly_order: int = 3,
) -> GridSearchResults:
    """Grid search to find optimal parameters.
    
    Args:
        data_loader: PyTorch DataLoader for evaluation.
        smoothing_windows: List of window sizes to try (must be odd).
        min_prominences: List of prominence values to try.
        min_distances: List of distance values to try.
        median_filter_sizes: List of median filter kernel sizes to try (must be odd).
        poly_order: Polynomial order for all configurations.
        
    Returns:
        GridSearchResults dataclass containing:
            - best_params: Best parameter configuration
            - best_mae: Best MAE achieved
            - all_results: All configurations and their results
    """
    best_mae = float('inf')
    best_params = None
    all_results = []
    
    total = (len(smoothing_windows) * len(min_prominences) * 
             len(min_distances) * len(median_filter_sizes))
    
    iterator = itertools.product(
        smoothing_windows, min_prominences, min_distances, median_filter_sizes
    )

    for window, prominence, distance, median_size in tqdm(iterator, total=total, desc="Grid Search"):
        params = HeuristicParameters(
            smoothing_window=window,
            poly_order=poly_order,
            min_prominence=prominence,
            min_distance=distance,
            median_filter_size=median_size,
        )
        counter = HeuristicPushupCounter(params)
        
        metrics = evaluate_on_dataset(data_loader, counter)
        
        result = {
            'smoothing_window': window,
            'min_prominence': prominence,
            'min_distance': distance,
            'median_filter_size': median_size,
            'mae': metrics['mae'],
            'accuracy': metrics['accuracy'],
            'within_1_accuracy': metrics['within_1_accuracy'],
        }
        all_results.append(result)
        
        if metrics['mae'] < best_mae:
            best_mae = metrics['mae']
            best_params = HeuristicParameters(
                smoothing_window=window,
                poly_order=poly_order,
                min_prominence=prominence,
                min_distance=distance,
                median_filter_size=median_size,
            )
    
    print("\n" + "="*50)
    print(f"Grid Search Complete.")
    print(f"Best MAE: {best_mae:.4f}")
    print(f"Best Parameters: {best_params}")
    print("="*50 + "\n")

    return GridSearchResults(
        best_params=best_params,
        best_mae=best_mae,
        all_results=all_results,
    )
