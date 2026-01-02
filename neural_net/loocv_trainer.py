"""Leave One Out Cross Validation (LOOCV) training module.

This module provides functionality to train neural network models using
Leave One Out Cross Validation, where each sample is used as the validation
set exactly once. This results in N models for a dataset of N samples.
"""

from dataclasses import dataclass
from typing import Callable, Optional
import copy

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

from data_loader import TransformDataset


@dataclass
class LOOCVFoldMetrics:
    """Metrics tracked for each epoch within a fold."""
    loss: float
    mean_absolute_error: float
    exact_match_accuracy: float
    off_by_one_accuracy: float


@dataclass
class LOOCVFoldResult:
    """Complete results from training a single LOOCV fold.
    
    In LOOCV, each fold validates on exactly one sample.
    """
    fold_index: int
    model_state_dict: dict
    train_history: list[LOOCVFoldMetrics]
    val_history: list[LOOCVFoldMetrics]
    best_epoch: int
    best_val_loss: float
    val_sample_index: int  # Index of the sample used for validation
    val_prediction: float  # Model's prediction on the validation sample
    val_target: float  # Ground truth for the validation sample


@dataclass
class LOOCVResult:
    """Results from training the complete LOOCV ensemble."""
    fold_results: list[LOOCVFoldResult]
    n_samples: int
    overall_mae: float
    overall_exact_match: float
    overall_off_by_one: float


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



class LOOCVTrainer:
    """Leave One Out Cross Validation trainer.
    
    Trains N models for a dataset of N samples. Each model is trained on
    N-1 samples and validated on the remaining 1 sample.
    
    Uses batch_size=1 with gradient accumulation to avoid padding issues
    with variable-length sequences while maintaining stable training.
    
    Args:
        model_factory: Callable that returns a fresh model instance
        loss_fn: nn.Module loss function (required)
        optimizer_factory: Callable (model.parameters()) -> Optimizer
        device: torch.device to use for training
        patience: Early stopping patience (epochs)
        max_epochs: Maximum training epochs per fold
        accumulation_steps: Number of steps to accumulate gradients before
            performing an optimizer step (default 16)
        prediction_extractor: Callable that extracts the prediction tensor from
            model output. Default extracts the first element (for models returning
            (predictions, ...) tuples)
        verbose: If True, print detailed progress for each fold
        print_every: Print progress every N folds (only when verbose=False)
    """
    
    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        loss_fn: nn.Module,
        optimizer_factory: Optional[Callable[..., torch.optim.Optimizer]] = None,
        device: Optional[torch.device] = None,
        patience: int = 10,
        max_epochs: int = 100,
        accumulation_steps: int = 16,
        prediction_extractor: Optional[Callable] = None,
        verbose: bool = False,
        print_every: int = 10,
    ):
        self.model_factory = model_factory
        self.loss_fn = loss_fn
        self.optimizer_factory = optimizer_factory or (
            lambda params: torch.optim.Adam(params, lr=1e-3)
        )
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.patience = patience
        self.max_epochs = max_epochs
        self.accumulation_steps = accumulation_steps
        self.prediction_extractor = prediction_extractor or (lambda output: output[0])
        self.verbose = verbose
        self.print_every = print_every
    
    def train(
        self,
        dataset: Dataset,
        train_transform: Optional[Callable] = None,
        num_workers: int = 0,
    ) -> LOOCVResult:
        """Train using Leave One Out Cross Validation.
        
        For a dataset of N samples, trains N models. Each model i is trained
        on all samples except sample i, and validated on sample i.
        
        Args:
            dataset: PyTorch Dataset (without augmentation transforms)
            train_transform: Transform to apply to training data (augmentations)
            num_workers: Number of workers for DataLoaders
            
        Returns:
            LOOCVResult containing all fold results and aggregate metrics
        """
        n_samples = len(dataset)
        fold_results = []
        all_predictions = []
        all_targets = []
        
        print(f"\nStarting Leave One Out Cross Validation with {n_samples} folds")
        print("=" * 60)
        
        for fold_idx in range(n_samples):
            if self.verbose:
                print(f"\n{'='*50}")
                print(f"Fold {fold_idx + 1}/{n_samples} (Validating on sample {fold_idx})")
                print(f"{'='*50}")
            elif (fold_idx + 1) % self.print_every == 0 or fold_idx == 0:
                print(f"Training fold {fold_idx + 1}/{n_samples}...")
            
            # Create train indices (all except current sample)
            train_indices = [i for i in range(n_samples) if i != fold_idx]
            val_indices = [fold_idx]
            
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
            fold_result = self._train_fold(fold_idx, train_loader, val_loader)
            fold_results.append(fold_result)
            
            all_predictions.append(fold_result.val_prediction)
            all_targets.append(fold_result.val_target)
        
        # Compute overall metrics
        predictions_tensor = torch.tensor(all_predictions)
        targets_tensor = torch.tensor(all_targets)
        
        rounded_preds = torch.round(predictions_tensor)
        overall_mae = torch.abs(predictions_tensor - targets_tensor).mean().item()
        overall_exact = (rounded_preds == targets_tensor).float().mean().item()
        overall_off_by_one = (torch.abs(rounded_preds - targets_tensor) <= 1).float().mean().item()
        
        print(f"\n{'='*60}")
        print("LOOCV Training Complete")
        print(f"{'='*60}")
        print(f"Overall MAE: {overall_mae:.4f}")
        print(f"Overall Exact Match: {overall_exact:.1%}")
        print(f"Overall Off-by-One: {overall_off_by_one:.1%}")
        
        return LOOCVResult(
            fold_results=fold_results,
            n_samples=n_samples,
            overall_mae=overall_mae,
            overall_exact_match=overall_exact,
            overall_off_by_one=overall_off_by_one,
        )
    
    def _train_fold(
        self,
        fold_index: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> LOOCVFoldResult:
        """Train a single LOOCV fold."""
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
            
            # Validation epoch (single sample)
            val_metrics, val_pred, val_target = self._evaluate_single(model, val_loader)
            val_history.append(val_metrics)
            
            # Check for best model
            if early_stopping(val_metrics.loss):
                if self.verbose:
                    print(f"  Early stopping triggered at epoch {epoch + 1}")
                break
            
            if early_stopping.is_best():
                best_model_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                best_val_loss = val_metrics.loss
                best_val_pred = val_pred
                best_val_target = val_target
            
            # Progress logging (only in verbose mode)
            if self.verbose and ((epoch + 1) % 5 == 0 or epoch == 0):
                print(
                    f"  Epoch {epoch + 1:3d} | "
                    f"Train Loss: {train_metrics.loss:.4f}, MAE: {train_metrics.mean_absolute_error:.2f} | "
                    f"Val Loss: {val_metrics.loss:.4f}, Pred: {val_pred:.2f}, Target: {val_target:.0f}"
                )
        
        if self.verbose:
            print(
                f"  Best epoch: {best_epoch + 1} | "
                f"Val Loss: {best_val_loss:.4f}, Pred: {best_val_pred:.2f}, Target: {best_val_target:.0f}"
            )
        
        return LOOCVFoldResult(
            fold_index=fold_index,
            model_state_dict=best_model_state or model.state_dict(),
            train_history=train_history,
            val_history=val_history,
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            val_sample_index=fold_index,
            val_prediction=best_val_pred if best_model_state else val_pred,
            val_target=best_val_target if best_model_state else val_target,
        )
    
    def _train_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
    ) -> LOOCVFoldMetrics:
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
            sequences, density_map, labels, _ = batch
            sequences = sequences.to(self.device)
            density_map = density_map.to(self.device)
            labels = labels.to(self.device).float()
            
            prediction, predicted_density_map = model(sequences)
            # Scale loss by accumulation steps for proper gradient averaging
            loss = self.loss_fn(predicted_density_map, density_map, labels) / self.accumulation_steps
            loss.backward()
            
            # Optimizer step after accumulating gradients
            if (step + 1) % self.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # Track unscaled loss for metrics
            total_loss += loss.item() * self.accumulation_steps
            num_samples += 1
            all_predictions.append(prediction.detach().cpu())
            all_targets.append(labels.detach().cpu())
        
        # Handle remaining gradients if dataset size isn't divisible by accumulation_steps
        if num_samples % self.accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
        
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        avg_loss = total_loss / num_samples
        
        return self._compute_metrics(all_predictions, all_targets, avg_loss)
    
    def _evaluate_single(
        self,
        model: nn.Module,
        loader: DataLoader,
    ) -> tuple[LOOCVFoldMetrics, float, float]:
        """Evaluate model on a single validation sample.
        
        Returns:
            Tuple of (metrics, prediction, target)
        """
        model.eval()
        
        with torch.no_grad():
            # LOOCV has exactly one sample in validation
            batch = next(iter(loader))
            sequences, density_map, labels, _ = batch
            sequences = sequences.to(self.device)
            density_map = density_map.to(self.device)
            labels = labels.to(self.device).float()
            
            total_count, predicted_map = model(sequences)
            loss = self.loss_fn(predicted_map, density_map, labels)
            
            prediction = total_count.cpu().item()
            target = labels.cpu().item()
        
        # For a single sample, compute metrics directly
        rounded_pred = round(prediction)
        mae = abs(prediction - target)
        exact_match = 1.0 if rounded_pred == target else 0.0
        off_by_one = 1.0 if abs(rounded_pred - target) <= 1 else 0.0
        
        metrics = LOOCVFoldMetrics(
            loss=loss.item(),
            mean_absolute_error=mae,
            exact_match_accuracy=exact_match,
            off_by_one_accuracy=off_by_one,
        )
        
        return metrics, prediction, target
    
    def _compute_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        loss: float,
    ) -> LOOCVFoldMetrics:
        """Compute all metrics from predictions and targets."""
        predictions = predictions.squeeze()
        rounded_preds = torch.round(predictions)
        
        # Mean Absolute Error
        mae = torch.abs(predictions - targets).mean().item()
        
        # Exact match accuracy
        exact_match = (rounded_preds == targets).float().mean().item()
        
        # Off-by-one accuracy (within ±1)
        off_by_one = (torch.abs(rounded_preds - targets) <= 1).float().mean().item()
        
        return LOOCVFoldMetrics(
            loss=loss,
            mean_absolute_error=mae,
            exact_match_accuracy=exact_match,
            off_by_one_accuracy=off_by_one,
        )
