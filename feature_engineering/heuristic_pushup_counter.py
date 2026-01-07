"""
Heuristic Push-up Counter using Signal Processing.

This module provides a baseline push-up counting approach using classical signal
processing techniques. It serves as a comparison baseline for deep learning approaches
and is used to generate pseudo ground truth labels.

Key Approach:
1. Extract the primary elbow angle signal from the raw landmarks (whichever elbow has more movement).
2. Apply a two-stage smoothing process to the landmark sequence:
   - Stage 1: Median filter to remove outliers and spikes from pose detection failures.
   - Stage 2: Savitzky-Golay filter to smooth jitter while preserving local peaks.
3. Detect peaks/valleys on the smoothed version of the extracted signal.
4. Count push-ups based on the number of detected valleys (bottom positions).
"""

from dataclasses import dataclass

import numpy as np
import torch
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter, find_peaks

from feature_engineering.constants import LEFT_ELBOW_ANGLE_IDX, RIGHT_ELBOW_ANGLE_IDX


@dataclass
class CounterParameters:
    """Parameters for HeuristicPushupCounter.
    
    Attributes:
        smoothing_window: Window size for Savitzky-Golay filter (must be odd)
        poly_order: Polynomial order for the Savitzky-Golay filter
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
class CountPushupResults:
    """Results from heuristic push-up counting.
    
    Attributes:
        count: Number of push-ups detected
        smoothed_angle_sequence: Smoothed landmarks sequence of shape (T, 6)
        raw_angle_sequence: Original input landmarks sequence of shape (T, 6)
        signal_index: Index of the angle used for signal extraction (e.g., 0=left elbow, 1=right elbow)
        peaks: Indices of detected peaks (top positions)
        valleys: Indices of detected valleys (bottom positions)
    """
    count: int
    smoothed_angle_sequence: np.ndarray
    raw_angle_sequence: np.ndarray
    signal_index: int
    peaks: np.ndarray
    valleys: np.ndarray


class HeuristicPushupCounter:
    """Signal processing based push-up counter.
    
    This class counts push-ups by detecting oscillations in elbow angle sequences.
    
    The key insight is that during a push-up, the elbow angle oscillates
    periodically. By tracking the elbow angle with more movement and
    detecting the valleys in this signal, we can count push-ups.
    """

    def __init__(self, params: CounterParameters):
        self.smoothing_window = params.smoothing_window
        self.poly_order = params.poly_order
        self.min_prominence = params.min_prominence
        self.min_distance = params.min_distance
        self.median_filter_size = params.median_filter_size

    def _extract_signal(self, angle_sequence: np.ndarray) -> int:
        """Extract the primary push-up signal from angle features.
        
        Uses the elbow angle with more movement (higher range) as the primary signal.
        The elbow angle oscillates as the person moves up and down during push-ups.
        
        Args:
            angle_sequence: Numpy array of shape (T, 6) containing angle features
                in degrees for T frames.
                
        Returns:
            - signal_index: Index of the joint angle used (see feature_engineering.constants). This is used
              to extract the raw signal from the original input landmarks.
        """
        left_elbow_angle = angle_sequence[:, LEFT_ELBOW_ANGLE_IDX]
        right_elbow_angle = angle_sequence[:, RIGHT_ELBOW_ANGLE_IDX]

        left_var = np.var(left_elbow_angle)
        right_var = np.var(right_elbow_angle)
        
        if left_var >= right_var:
            return LEFT_ELBOW_ANGLE_IDX

        return RIGHT_ELBOW_ANGLE_IDX
    
    def _smooth(self, angle_sequence: np.ndarray) -> np.ndarray:
        """Apply two-stage smoothing to the entire angle sequence.
        
        1. Median filter removes outliers/spikes from pose detection failures or incorrect landmark positions.
        2. Savitzky-Golay filter smooths high-frequency jitter while keeping signal peaks.
        
        Args:
            angle_sequence: Numpy array of shape (T, 6) containing angle features.
            
        Returns:
            Numpy array of shape (T, 6) with smoothed angle features.
        """
        smoothed = median_filter(angle_sequence, size=(self.median_filter_size, 1))
        
        if len(smoothed) < self.smoothing_window:
            return smoothed
        
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

    def count_pushups(self, angle_sequence: torch.Tensor) -> CountPushupResults:
        """Count push-ups based on signal processing.
        
        Process:
        1. Determine which elbow to use based on raw angle sequences. This
            determines the signal we use
        2. Smooth the signal
        3. Detect peaks/valleys on the smoothed version of the selected signal
        
        Args:
            angle_sequence: torch.Tensor of shape (T, 6).
            
        Returns:
            CountPushupResults See documentation of CountPushupReesults
        """
        angle_sequence = angle_sequence.numpy()

        signal_index = self._extract_signal(angle_sequence)
        
        smoothed_signals = self._smooth(angle_sequence)

        smoothed_signal = smoothed_signals[:, signal_index]
        
        valleys, peaks = self._detect_peaks(smoothed_signal)
        
        return CountPushupResults(
            count=len(valleys),
            smoothed_angle_sequence=smoothed_signals,
            raw_angle_sequence=angle_sequence,
            signal_index=signal_index,
            peaks=peaks,
            valleys=valleys,
        )
    
