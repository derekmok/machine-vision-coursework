"""
Evaluation and Hyperparameter Tuning for Heuristic Push-up Counter.

This module provides utilities for evaluating and optimizing the heuristic
push-up counting algorithm. It includes functions for:
1. Evaluating the counter's performance on datasets
2. Grid search for optimal hyperparameters
"""

import itertools
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm

from data_loader import VideoDataset
from feature_engineering.count_pushups_heuristic import (
    CounterParameters,
    HeuristicPushupCounter,
)


@dataclass
class GridSearchResults:
    """Results from grid search parameter optimization.
    
    Attributes:
        best_params: Best parameter configuration
        best_mae: Best Mean Absolute Error achieved with best_params
        all_results: List of all configurations tested and their metrics
        train_loader: Training DataLoader used for parameter search
        val_loader: Validation DataLoader used for evaluation
    """
    best_params: CounterParameters
    best_mae: float
    all_results: list
    train_loader: DataLoader  # DataLoader type
    val_loader: DataLoader  # DataLoader type


def evaluate_on_dataset(
    data_loader: DataLoader,
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
    
    for raw_landmarks, lengths, batch_label in data_loader:
        for i in range(len(batch_label)):
            length = lengths[i].item()
            landmarks = raw_landmarks[i, :length]
            
            pred = counter.count_pushups(landmarks).count
            actual = batch_label[i].item()
            
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
    video_dir: str,
    val_split: float = 0.2,
    smoothing_windows: list[int] = [11, 15, 21, 31],
    min_prominences: list[float] = [5.0, 10.0, 15.0, 20.0],
    min_distances: list[int] = [5, 10, 15, 20, 30],
    median_filter_sizes: list[int] = [3],
    poly_order: int = 3,
) -> GridSearchResults:
    """Grid search to find optimal parameters.
    
    Args:
        video_dir: Path to directory containing video files.
        val_split: Fraction of data to use for validation.
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
            - val_loader: Validation DataLoader for final evaluation
    """
    # Create train and validation dataloaders
    full_dataset = VideoDataset(
        video_dir, 
        feature_processor=lambda features: (features.raw_landmarks, len(features.raw_landmarks))
    )
    
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    # Perform train/val split with fixed seed for reproducibility
    train_subset, val_subset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    print(f"Train: {len(train_subset)} videos, Val: {len(val_subset)} videos")
    
    best_mae = float('inf')
    best_params = None
    all_results = []
    
    total = (len(smoothing_windows) * len(min_prominences) * 
             len(min_distances) * len(median_filter_sizes))
    
    iterator = itertools.product(
        smoothing_windows, min_prominences, min_distances, median_filter_sizes
    )

    for window, prominence, distance, median_size in tqdm(iterator, total=total, desc="Grid Search"):
        params = CounterParameters(
            smoothing_window=window,
            poly_order=poly_order,
            min_prominence=prominence,
            min_distance=distance,
            median_filter_size=median_size,
        )
        counter = HeuristicPushupCounter(params)
        
        metrics = evaluate_on_dataset(train_loader, counter)
        
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
            best_params = CounterParameters(
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
        train_loader=train_loader,
        val_loader=val_loader,
    )
