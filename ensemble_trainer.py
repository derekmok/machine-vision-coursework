"""Ensemble training module with k-fold cross-validation.

This module provides functionality to train an ensemble of neural network models
using k-fold cross-validation with early stopping, custom loss functions, and
comprehensive metrics tracking.
"""

from dataclasses import dataclass
from typing import Callable, Optional
import copy

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold


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


@dataclass
class EnsembleResult:
    """Results from training the complete ensemble."""
    fold_results: list[FoldResult]
    k: int


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


class TransformDataset(Dataset):
    """Wrapper dataset that applies a transform to an underlying dataset.
    
    This allows applying different transforms to training vs validation subsets.
    """
    
    def __init__(self, dataset: Dataset, transform: Optional[Callable] = None):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sequence, label = self.dataset[idx]
        
        if self.transform is not None:
            sequence, label = self.transform(sequence, label)
        
        return sequence, label


class EnsembleTrainer:
    """K-fold cross-validation ensemble trainer.
    
    Uses batch_size=1 with gradient accumulation to avoid padding issues
    with variable-length sequences while maintaining stable training.
    
    Args:
        model_factory: Callable that returns a fresh model instance
        loss_fn: nn.Module loss function (required)
        k: Number of folds (default 5)
        optimizer_factory: Callable (model.parameters()) -> Optimizer
        device: torch.device to use for training
        patience: Early stopping patience (epochs)
        max_epochs: Maximum training epochs per fold
        accumulation_steps: Number of steps to accumulate gradients before
            performing an optimizer step (default 16)
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
        accumulation_steps: int = 16,
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
        self.accumulation_steps = accumulation_steps
    
    def train(
        self,
        dataset: Dataset,
        train_transform: Optional[Callable] = None,
        val_transform: Optional[Callable] = None,
        num_workers: int = 0,
    ) -> EnsembleResult:
        """Train the ensemble using k-fold cross-validation.
        
        Always uses batch_size=1 with gradient accumulation to avoid padding
        issues with variable-length sequences.
        
        Args:
            dataset: PyTorch Dataset (without augmentation transforms)
            train_transform: Transform to apply to training data (augmentations)
            val_transform: Transform to apply to validation data (typically None)
            num_workers: Number of workers for DataLoaders
            
        Returns:
            EnsembleResult containing all fold results
        """
        kfold = KFold(n_splits=self.k, shuffle=True, random_state=42)
        fold_results = []
        
        for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(dataset)):
            print(f"\n{'='*50}")
            print(f"Fold {fold_idx + 1}/{self.k}")
            print(f"{'='*50}")
            
            # Create subsets
            train_subset = Subset(dataset, train_indices)
            val_subset = Subset(dataset, val_indices)
            
            # Wrap with transforms
            train_dataset = TransformDataset(train_subset, train_transform)
            val_dataset = TransformDataset(val_subset, val_transform)
            
            # Create data loaders with batch_size=1
            train_loader = DataLoader(
                train_dataset,
                batch_size=1,
                shuffle=True,
                num_workers=num_workers,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=num_workers,
            )
            
            # Train this fold
            fold_result = self._train_fold(fold_idx, train_loader, val_loader)
            fold_results.append(fold_result)
        
        return EnsembleResult(fold_results=fold_results, k=self.k)
    
    def _train_fold(
        self,
        fold_index: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> FoldResult:
        """Train a single fold."""
        # Create fresh model and optimizer for this fold
        model = self.model_factory().to(self.device)
        optimizer = self.optimizer_factory(model.parameters())
        early_stopping = EarlyStopping(patience=self.patience)
        
        train_history = []
        val_history = []
        best_model_state = None
        best_epoch = 0
        best_val_loss = float('inf')
        
        for epoch in range(self.max_epochs):
            # Training epoch
            train_metrics = self._train_epoch(model, train_loader, optimizer)
            train_history.append(train_metrics)
            
            # Validation epoch
            val_metrics = self._evaluate(model, val_loader)
            val_history.append(val_metrics)
            
            # Check for best model
            if early_stopping(val_metrics.loss):
                print(f"  Early stopping triggered at epoch {epoch + 1}")
                break
            
            if early_stopping.is_best():
                best_model_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                best_val_loss = val_metrics.loss
            
            # Progress logging
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(
                    f"  Epoch {epoch + 1:3d} | "
                    f"Train Loss: {train_metrics.loss:.4f}, MAE: {train_metrics.mean_absolute_error:.2f} | "
                    f"Val Loss: {val_metrics.loss:.4f}, MAE: {val_metrics.mean_absolute_error:.2f}, "
                    f"Exact: {val_metrics.exact_match_accuracy:.1%}, "
                    f"Off1: {val_metrics.off_by_one_accuracy:.1%}"
                )
        
        best_val_metrics = val_history[best_epoch]
        print(
            f"  Best epoch: {best_epoch + 1} | "
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
        )
    
    def _train_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
    ) -> FoldMetrics:
        """Run a single training epoch with gradient accumulation.
        
        Since we use batch_size=1, gradients are accumulated over
        `self.accumulation_steps` before performing an optimizer step.
        """
        model.train()
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        num_samples = 0
        
        optimizer.zero_grad()
        
        for step, batch in enumerate(loader):
            sequences, labels = batch
            sequences = sequences.to(self.device)
            labels = labels.to(self.device).float()
            
            predictions, _ = model(sequences)
            # Scale loss by accumulation steps for proper gradient averaging
            loss = self.loss_fn(predictions.squeeze(1), labels) / self.accumulation_steps
            loss.backward()
            
            # Optimizer step after accumulating gradients
            if (step + 1) % self.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # Track unscaled loss for metrics
            total_loss += loss.item() * self.accumulation_steps
            num_samples += 1
            all_predictions.append(predictions.detach().cpu())
            all_targets.append(labels.detach().cpu())
        
        # Handle remaining gradients if dataset size isn't divisible by accumulation_steps
        if num_samples % self.accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
        
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
        
        with torch.no_grad():
            for batch in loader:
                sequences, labels = batch
                sequences = sequences.to(self.device)
                labels = labels.to(self.device).float()
                
                predictions, _ = model(sequences)
                loss = self.loss_fn(predictions.squeeze(1), labels)
                
                total_loss += loss.item()
                num_batches += 1
                all_predictions.append(predictions.cpu())
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
