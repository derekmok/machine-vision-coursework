"""Ensemble training module with k-fold cross-validation.

This module provides functionality to train an ensemble of neural network models
using k-fold cross-validation with early stopping, custom loss functions, and
comprehensive metrics tracking.
"""

from dataclasses import dataclass
from typing import Callable, Optional
import copy
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
from tqdm.auto import tqdm

from data_loader import TransformDataset


@dataclass
class FoldMetrics:
    """Metrics tracked for each epoch within a fold."""
    loss: float
    mean_absolute_error: float
    exact_match_accuracy: float
    off_by_one_accuracy: float


@dataclass
class FoldResult:
    """Complete results from training a single fold."""
    fold_index: int
    model_state_dict: dict
    train_history: list[FoldMetrics]
    val_history: list[FoldMetrics]
    best_epoch: int
    best_val_loss: float
    training_time_seconds: float


@dataclass
class EnsembleResult:
    """Results from training the complete ensemble."""
    fold_results: list[FoldResult]
    k: int
    total_training_time_seconds: float


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.0f}s"


class EarlyStopping:
    """Early stopping to terminate training when validation loss stops improving.
    
    Args:
        patience: Number of epochs with no improvement before stopping
        min_delta: Minimum change to qualify as an improvement
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self._is_best = False
    
    def __call__(self, val_loss: float) -> bool:
        """Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            True if training should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self._is_best = True
        else:
            self.counter += 1
            self._is_best = False
        
        return self.counter >= self.patience
    
    def is_best(self) -> bool:
        """Returns True if the last call had the best loss so far."""
        return self._is_best




class EnsembleTrainer:
    """K-fold cross-validation ensemble trainer.
    
    Uses batch_size=1 to avoid padding issues with variable-length sequences.
    
    Args:
        model_factory: Callable that returns a fresh model instance
        loss_fn: nn.Module loss function (required)
        k: Number of folds (default 5)
        optimizer_factory: Callable (model.parameters()) -> Optimizer
        device: torch.device to use for training
        patience: Early stopping patience (epochs)
        max_epochs: Maximum training epochs per fold
    """
    
    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        loss_fn: nn.Module,
        k: int = 5,
        optimizer_factory: Optional[Callable[..., torch.optim.Optimizer]] = None,
        device: Optional[torch.device] = None,
        patience: int = 10,
        max_epochs: int = 100,
    ):
        self.model_factory = model_factory
        self.loss_fn = loss_fn
        self.k = k
        self.optimizer_factory = optimizer_factory or (
            lambda params: torch.optim.Adam(params, lr=1e-3)
        )
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.patience = patience
        self.max_epochs = max_epochs
    
    def train(
        self,
        dataset: Dataset,
        train_transform: Optional[Callable] = None,
        num_workers: int = 0,
    ) -> EnsembleResult:
        """Train the ensemble using k-fold cross-validation.
        
        Always uses batch_size=1 to avoid padding issues with variable-length sequences.
        
        Args:
            dataset: PyTorch Dataset (without augmentation transforms)
            train_transform: Transform to apply to training data (augmentations)
            num_workers: Number of workers for DataLoaders
            
        Returns:
            EnsembleResult containing all fold results
        """
        overall_start_time = time.time()
        print(f"Training on device: {self.device}")
        print()
        print("Note that the first epoch of the first fold will take a long time to train")
        print("as pose detection is run on every video (takes up to 30 minutes).")
        print()
        print("Subsequent epochs will be much faster as pose detection is cached.")
        print()
        kfold = KFold(n_splits=self.k, shuffle=True, random_state=42)
        fold_results = []
        
        # Create progress bar for folds
        fold_splits = list(kfold.split(dataset))
        fold_pbar = tqdm(
            enumerate(fold_splits),
            total=self.k,
            desc="Training Folds",
            unit="fold",
            leave=True,
        )
        
        for fold_idx, (train_indices, val_indices) in fold_pbar:
            fold_pbar.set_description(f"Fold {fold_idx + 1}/{self.k}")
            
            # Create subsets
            train_subset = Subset(dataset, train_indices)
            val_subset = Subset(dataset, val_indices)
            
            # Wrap training data with transforms
            train_dataset = TransformDataset(train_subset, train_transform)
            
            # Create data loaders with batch_size=1
            train_loader = DataLoader(
                train_dataset,
                batch_size=1,
                shuffle=True,
                num_workers=num_workers,
            )
            val_loader = DataLoader(
                val_subset,
                batch_size=1,
                shuffle=False,
                num_workers=num_workers,
            )
            
            # Train this fold
            fold_result = self._train_fold(
                fold_idx, train_loader, val_loader
            )
            fold_results.append(fold_result)
            
            # Update fold progress bar with results
            best_metrics = fold_result.val_history[fold_result.best_epoch]
            fold_pbar.set_postfix({
                "time": _format_duration(fold_result.training_time_seconds),
                "best_epoch": fold_result.best_epoch + 1,
                "val_loss": f"{fold_result.best_val_loss:.4f}",
                "exact": f"{best_metrics.exact_match_accuracy:.1%}",
            })
        
        total_training_time = time.time() - overall_start_time
        
        # Print training summary
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"{'='*60}")
        print(f"Total time: {_format_duration(total_training_time)}")
        if fold_results:
            avg_fold_time = sum(r.training_time_seconds for r in fold_results) / len(fold_results)
            print(f"Average time per fold: {_format_duration(avg_fold_time)}")
        print(f"{'='*60}")
        
        return EnsembleResult(
            fold_results=fold_results,
            k=self.k,
            total_training_time_seconds=total_training_time,
        )
    
    def _train_fold(
        self,
        fold_index: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> FoldResult:
        """Train a single fold."""
        fold_start_time = time.time()
        
        # Create fresh model and optimizer for this fold
        model = self.model_factory().to(self.device)
        optimizer = self.optimizer_factory(model.parameters())
        early_stopping = EarlyStopping(patience=self.patience)
        
        train_history = []
        val_history = []
        best_model_state = None
        best_epoch = 0
        best_val_loss = float('inf')
        
        # Create tqdm progress bar for epochs
        epoch_pbar = tqdm(
            range(self.max_epochs),
            desc=f"Fold {fold_index + 1}/{self.k}",
            unit="epoch",
            leave=True,
        )
        
        for epoch in epoch_pbar:
            # Training epoch
            train_metrics = self._train_epoch(
                model, train_loader, optimizer
            )
            train_history.append(train_metrics)
            
            # Validation epoch
            val_metrics = self._evaluate(model, val_loader)
            val_history.append(val_metrics)
            
            # Update progress bar with current metrics
            status = ""
            if early_stopping.is_best():
                status = "★"
            epoch_pbar.set_postfix({
                "val_loss": f"{val_metrics.loss:.4f}",
                "mae": f"{val_metrics.mean_absolute_error:.2f}",
                "exact": f"{val_metrics.exact_match_accuracy:.1%}",
                "status": status,
            })
            
            # Check for best model
            if early_stopping(val_metrics.loss):
                epoch_pbar.set_postfix({
                    "val_loss": f"{val_metrics.loss:.4f}",
                    "mae": f"{val_metrics.mean_absolute_error:.2f}",
                    "exact": f"{val_metrics.exact_match_accuracy:.1%}",
                    "status": "early_stop",
                })
                break
            
            if early_stopping.is_best():
                best_model_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                best_val_loss = val_metrics.loss
        
        fold_training_time = time.time() - fold_start_time
        
        best_val_metrics = val_history[best_epoch]
        print(
            f"  Fold {fold_index + 1} complete in {_format_duration(fold_training_time)} | "
            f"Best epoch: {best_epoch + 1} | "
            f"Val Loss: {best_val_loss:.4f}, MAE: {best_val_metrics.mean_absolute_error:.2f}, "
            f"Exact: {best_val_metrics.exact_match_accuracy:.1%}, "
            f"Off1: {best_val_metrics.off_by_one_accuracy:.1%}"
        )
        
        return FoldResult(
            fold_index=fold_index,
            model_state_dict=best_model_state or model.state_dict(),
            train_history=train_history,
            val_history=val_history,
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            training_time_seconds=fold_training_time,
        )
    
    def _train_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
    ) -> FoldMetrics:
        """Run a single training epoch."""
        model.train()
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        num_samples = 0
        
        batch_iter = loader
        
        for batch in batch_iter:
            sequences, density_map, labels, _ = batch
            sequences = sequences.to(self.device)
            density_map = density_map.to(self.device)
            labels = labels.to(self.device).float()
            
            optimizer.zero_grad()
            prediction, predicted_density_map = model(sequences)
            loss = self.loss_fn(predicted_density_map, density_map, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_samples += 1
            all_predictions.append(prediction.detach().cpu())
            all_targets.append(labels.detach().cpu())
        
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        avg_loss = total_loss / num_samples
        
        return self._compute_metrics(all_predictions, all_targets, avg_loss)
    
    def _evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,
    ) -> FoldMetrics:
        """Evaluate model on a dataset."""
        model.eval()
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        num_batches = 0
        
        batch_iter = loader
        
        with torch.no_grad():
            for batch in batch_iter:
                sequences, density_map, labels, _ = batch
                sequences = sequences.to(self.device)
                density_map = density_map.to(self.device)
                labels = labels.to(self.device).float()
                
                total_count, predicted_map = model(sequences)
                loss = self.loss_fn(predicted_map, density_map, labels)
                
                total_loss += loss.item()
                num_batches += 1
                all_predictions.append(total_count.cpu())
                all_targets.append(labels.cpu())
        
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        avg_loss = total_loss / num_batches
        
        return self._compute_metrics(all_predictions, all_targets, avg_loss)
    
    def _compute_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        loss: float,
    ) -> FoldMetrics:
        """Compute all metrics from predictions and targets."""
        predictions = predictions.squeeze()
        rounded_preds = torch.round(predictions)
        
        # Mean Absolute Error
        mae = torch.abs(predictions - targets).mean().item()
        
        # Exact match accuracy
        exact_match = (rounded_preds == targets).float().mean().item()
        
        # Off-by-one accuracy (within ±1)
        off_by_one = (torch.abs(rounded_preds - targets) <= 1).float().mean().item()
        
        return FoldMetrics(
            loss=loss,
            mean_absolute_error=mae,
            exact_match_accuracy=exact_match,
            off_by_one_accuracy=off_by_one,
        )
