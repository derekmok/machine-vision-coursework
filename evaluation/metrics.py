"""Shared metrics computation utilities for model evaluation."""

from typing import TypedDict

import numpy as np


class MetricsResult(TypedDict):
    """Result dictionary from compute_prediction_metrics."""
    mae: float
    exact_match_accuracy: float
    off_by_one_accuracy: float
    rounded_predictions: np.ndarray


def compute_prediction_metrics(predictions: np.ndarray, targets: np.ndarray) -> MetricsResult:
    """Compute standard prediction metrics.
    
    Args:
        predictions: Raw model predictions (float)
        targets: Ground truth values
        
    Returns:
        Dictionary with mae, exact_match_accuracy, and off_by_one_accuracy
    """
    rounded_preds = np.round(predictions)
    
    mae = np.mean(np.abs(predictions - targets))
    exact_match = np.mean(rounded_preds == targets)
    off_by_one = np.mean(np.abs(rounded_preds - targets) <= 1)
    
    return {
        'mae': mae,
        'exact_match_accuracy': exact_match,
        'off_by_one_accuracy': off_by_one,
        'rounded_predictions': rounded_preds,
    }
