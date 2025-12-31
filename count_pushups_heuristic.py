"""
Heuristic Push-up Counter using Signal Processing.

This module provides a baseline push-up counting approach using classical signal
processing techniques. It serves as a comparison baseline for deep learning approaches.

Key Approach:
1. Extract elbow angle as a 1D time series (whichever elbow has more movement)
2. Apply Savitzky-Golay smoothing to reduce noise
3. Detect peaks/valleys using scipy's find_peaks
4. Count complete push-up cycles
"""

import numpy as np
from scipy.signal import savgol_filter, find_peaks
from typing import Optional
import matplotlib.pyplot as plt


class HeuristicPushupCounter:
    """Signal processing based push-up counter.
    
    This class counts push-ups by detecting oscillations in the elbow angle
    extracted from video frames using MediaPipe.
    
    The key insight is that during a push-up, the elbow angle oscillates
    periodically. By tracking the elbow angle with more movement and
    detecting the peaks (or valleys) in this signal, we can count push-ups.
    
    Attributes:
        smoothing_window: Window size for Savitzky-Golay filter (must be odd)
        poly_order: Polynomial order for Savitzky-Golay smoothing
        min_prominence: Minimum prominence for peak detection (filters noise)
        min_distance: Minimum frames between peaks (prevents double-counting)
    """
    
    # Angle feature indices in the 6-dimensional feature vector from data_loader.py
    # Format: [left_elbow, right_elbow, left_shoulder, right_shoulder, 
    #          left_body, right_body]
    LEFT_ELBOW_ANGLE_IDX = 0
    RIGHT_ELBOW_ANGLE_IDX = 1
    
    def __init__(
        self,
        smoothing_window: int = 15,
        poly_order: int = 3,
        min_prominence: float = 0.02,
        min_distance: int = 10,
    ):
        """Initialize the heuristic push-up counter.
        
        Args:
            smoothing_window: Window size for Savitzky-Golay filter. Must be odd.
                Larger values = more smoothing. Typical range: 11-21 for 30fps video.
            poly_order: Polynomial order for Savitzky-Golay filter. 
                Higher values preserve peak shapes better. Typical: 2-4.
            min_prominence: Minimum amplitude difference from surrounding signal
                for a point to be considered a valid peak. Filters out noise.
                Should be tuned based on the typical push-up amplitude.
            min_distance: Minimum number of frames between detected peaks.
                Set based on the fastest expected push-up speed. At 30fps,
                a distance of 10 means at most 3 push-ups per second.
        """
        if smoothing_window % 2 == 0:
            raise ValueError("smoothing_window must be odd")
        if poly_order >= smoothing_window:
            raise ValueError("poly_order must be less than smoothing_window")
            
        self.smoothing_window = smoothing_window
        self.poly_order = poly_order
        self.min_prominence = min_prominence
        self.min_distance = min_distance
    
    def extract_signal(self, landmarks: np.ndarray) -> np.ndarray:
        """Extract the primary push-up signal from angle features.
        
        Uses the elbow angle with more movement (higher range) as the primary signal.
        The elbow angle oscillates as the person moves up and down during push-ups.
        
        Args:
            landmarks: Numpy array of shape (T, 6) containing angle features
                in degrees for T frames.
                
        Returns:
            1D numpy array of shape (T,) containing the selected elbow angle signal.
        """
        left_elbow_angle = landmarks[:, self.LEFT_ELBOW_ANGLE_IDX]
        right_elbow_angle = landmarks[:, self.RIGHT_ELBOW_ANGLE_IDX]
        
        # Select the elbow with more movement (higher range)
        left_range = np.ptp(left_elbow_angle)  # peak-to-peak (max - min)
        right_range = np.ptp(right_elbow_angle)
        
        if left_range >= right_range:
            return left_elbow_angle
        else:
            return right_elbow_angle
    
    def smooth_signal(self, signal: np.ndarray) -> np.ndarray:
        """Apply Savitzky-Golay smoothing to reduce noise.
        
        Savitzky-Golay filtering is preferred over simple moving average because
        it better preserves peak shapes and locations, which is critical for
        accurate peak detection.
        
        Args:
            signal: 1D numpy array of the raw signal.
            
        Returns:
            1D numpy array of the smoothed signal (same length as input).
        """
        # Handle edge case where signal is shorter than window
        if len(signal) < self.smoothing_window:
            # Fall back to simple padding or return as-is
            return signal
            
        return savgol_filter(signal, self.smoothing_window, self.poly_order)
    
    def detect_peaks(self, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Detect peaks and valleys in the push-up signal.
        
        In the push-up motion:
        - Valleys (minima) = bottom position (arms bent, body low)
        - Peaks (maxima) = top position (arms extended, body high)
        
        We detect valleys by finding peaks in the negated signal.
        
        Args:
            signal: 1D numpy array of the (smoothed) signal.
            
        Returns:
            Tuple of (valleys, peaks) where each is an array of indices.
        """
        # Find peaks (top position)
        peaks, _ = find_peaks(
            signal,
            prominence=self.min_prominence,
            distance=self.min_distance
        )
        
        # Find valleys (bottom position) by inverting the signal
        valleys, _ = find_peaks(
            -signal,
            prominence=self.min_prominence,
            distance=self.min_distance
        )
        
        return valleys, peaks
    
    def count_pushups(self, landmarks: np.ndarray) -> int:
        """Count push-ups in an angle feature sequence.
        
        A complete push-up is counted as one valley (minimum elbow angle).
        
        Args:
            landmarks: Numpy array of shape (T, 6) or torch.Tensor.
                Can be the direct output from VideoDataset.
                
        Returns:
            Integer count of push-ups detected.
        """
        # Convert torch tensor if needed
        if hasattr(landmarks, 'numpy'):
            landmarks = landmarks.numpy()
            
        # Handle potential zero-padded sequences by removing trailing zeros
        # Check if entire rows are zeros (missing detection frames)
        valid_mask = ~np.all(landmarks == 0, axis=1)
        # Find the last valid frame
        if valid_mask.any():
            last_valid = np.where(valid_mask)[0][-1] + 1
            landmarks = landmarks[:last_valid]
        
        if len(landmarks) < self.smoothing_window:
            # Too short to analyze
            return 0
            
        # Extract and smooth the signal
        raw_signal = self.extract_signal(landmarks)
        smoothed_signal = self.smooth_signal(raw_signal)
        
        # Detect peaks
        valleys, peaks = self.detect_peaks(smoothed_signal)
        
        # Count valleys as push-ups (bottom position is more reliable)
        return len(valleys)
    
    def count_pushups_with_debug(self, landmarks: np.ndarray) -> dict:
        """Count push-ups and return debug information.
        
        Useful for visualization and parameter tuning.
        
        Args:
            landmarks: Numpy array of shape (T, 6).
            
        Returns:
            Dictionary containing:
                - count: Push-up count
                - raw_signal: Original extracted signal
                - smoothed_signal: After Savitzky-Golay smoothing
                - valleys: Indices of detected valleys (bottom positions)
                - peaks: Indices of detected peaks (top positions)
        """
        # Convert torch tensor if needed
        if hasattr(landmarks, 'numpy'):
            landmarks = landmarks.numpy()
            
        # Handle zero-padded sequences
        valid_mask = ~np.all(landmarks == 0, axis=1)
        if valid_mask.any():
            last_valid = np.where(valid_mask)[0][-1] + 1
            landmarks = landmarks[:last_valid]
            
        raw_signal = self.extract_signal(landmarks)
        
        if len(raw_signal) < self.smoothing_window:
            return {
                'count': 0,
                'raw_signal': raw_signal,
                'smoothed_signal': raw_signal,
                'valleys': np.array([]),
                'peaks': np.array([]),
            }
            
        smoothed_signal = self.smooth_signal(raw_signal)
        valleys, peaks = self.detect_peaks(smoothed_signal)
        
        return {
            'count': len(valleys),
            'raw_signal': raw_signal,
            'smoothed_signal': smoothed_signal,
            'valleys': valleys,
            'peaks': peaks,
        }
    
    def plot_debug(
        self, 
        landmarks: np.ndarray, 
        title: str = "Push-up Detection",
        ax: Optional[plt.Axes] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot the signal with detected peaks for visualization.
        
        Args:
            landmarks: Numpy array of shape (T, 6).
            title: Plot title.
            ax: Optional matplotlib axes to plot on.
            save_path: If provided, save the figure to this path.
            
        Returns:
            The matplotlib Figure object.
        """
        debug = self.count_pushups_with_debug(landmarks)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 5))
        else:
            fig = ax.get_figure()
            
        frames = np.arange(len(debug['raw_signal']))
        
        # Plot raw signal
        ax.plot(frames, debug['raw_signal'], 'b-', alpha=0.3, label='Raw Signal')
        
        # Plot smoothed signal
        ax.plot(frames, debug['smoothed_signal'], 'b-', linewidth=2, label='Smoothed Signal')
        
        # Mark valleys (bottom positions = push-ups)
        if len(debug['valleys']) > 0:
            ax.scatter(
                debug['valleys'], 
                debug['smoothed_signal'][debug['valleys']], 
                c='red', s=100, marker='v', zorder=5,
                label=f"Valleys (Push-ups: {debug['count']})"
            )
        
        # Mark peaks (top positions)
        if len(debug['peaks']) > 0:
            ax.scatter(
                debug['peaks'], 
                debug['smoothed_signal'][debug['peaks']], 
                c='green', s=80, marker='^', zorder=5,
                label='Peaks (Top position)'
            )
        
        ax.set_xlabel('Frame')
        ax.set_ylabel('Elbow Angle (degrees)')
        ax.set_title(f"{title} - Detected: {debug['count']}")
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
    
    for batch_landmarks, batch_labels, batch_lengths in data_loader:
        for i in range(len(batch_labels)):
            length = batch_lengths[i].item()
            landmarks = batch_landmarks[i, :length].numpy()
            
            pred = counter.count_pushups(landmarks)
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
    poly_order: int = 3,
) -> dict:
    """Grid search to find optimal parameters.
    
    Args:
        data_loader: PyTorch DataLoader for evaluation.
        smoothing_windows: List of window sizes to try (must be odd).
        min_prominences: List of prominence values to try.
        min_distances: List of distance values to try.
        poly_order: Polynomial order for all configurations.
        
    Returns:
        Dictionary with:
            - best_params: Best parameter configuration
            - best_mae: Best MAE achieved
            - all_results: All configurations and their results
    """
    best_mae = float('inf')
    best_params = None
    all_results = []
    
    total = len(smoothing_windows) * len(min_prominences) * len(min_distances)
    current = 0
    
    for window in smoothing_windows:
        for prominence in min_prominences:
            for distance in min_distances:
                current += 1
                print(f"[{current}/{total}] Testing window={window}, "
                      f"prominence={prominence}, distance={distance}")
                
                counter = HeuristicPushupCounter(
                    smoothing_window=window,
                    poly_order=poly_order,
                    min_prominence=prominence,
                    min_distance=distance,
                )
                
                metrics = evaluate_on_dataset(data_loader, counter)
                
                result = {
                    'smoothing_window': window,
                    'min_prominence': prominence,
                    'min_distance': distance,
                    'mae': metrics['mae'],
                    'accuracy': metrics['accuracy'],
                    'within_1_accuracy': metrics['within_1_accuracy'],
                }
                all_results.append(result)
                
                if metrics['mae'] < best_mae:
                    best_mae = metrics['mae']
                    best_params = {
                        'smoothing_window': window,
                        'poly_order': poly_order,
                        'min_prominence': prominence,
                        'min_distance': distance,
                    }
                    print(f"  -> New best MAE: {best_mae:.4f}")
    
    return {
        'best_params': best_params,
        'best_mae': best_mae,
        'all_results': all_results,
    }
