import os

import matplotlib.pyplot as plt

from neural_net.ensemble_trainer import FoldResult


def plot_training_results(results: list[FoldResult]) -> None:
    """Plot training and validation metrics for all folds.
    
    Creates a 4x2 grid showing training (left) and validation (right) for:
    - Loss
    - Mean Absolute Error
    - Exact Match Accuracy
    - Off-by-One Accuracy
    
    Args:
        results: list of FoldResult objects from training
    """
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    fig.suptitle('Training Results Across Folds', fontsize=14, fontweight='bold')

    # Column headers
    axes[0, 0].set_title('Training', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Validation', fontsize=12, fontweight='bold')

    metrics = [
        ('loss', 'Loss', 0),
        ('mean_absolute_error', 'Mean Absolute Error', 1),
        ('exact_match_accuracy', 'Exact Match Accuracy', 2),
        ('off_by_one_accuracy', 'Off-by-One Accuracy', 3),
    ]

    colors = plt.cm.tab10.colors

    for metric_name, metric_label, row_idx in metrics:
        train_ax = axes[row_idx, 0]
        val_ax = axes[row_idx, 1]

        for fold_result in results:
            fold_idx = fold_result.fold_index
            color = colors[fold_idx % len(colors)]

            train_values = [getattr(m, metric_name) for m in fold_result.train_history]
            val_values = [getattr(m, metric_name) for m in fold_result.val_history]
            epochs = range(1, len(train_values) + 1)

            train_ax.plot(epochs, train_values, '-', color=color,
                          label=f'Fold {fold_idx + 1}')
            train_ax.axvline(x=fold_result.best_epoch + 1, color=color,
                             linestyle='--', linewidth=2, alpha=0.7)

            val_ax.plot(epochs, val_values, '-', color=color,
                        label=f'Fold {fold_idx + 1}')
            val_ax.axvline(x=fold_result.best_epoch + 1, color=color,
                           linestyle='--', linewidth=2, alpha=0.7)

        train_ax.set_xlabel('Epoch')
        train_ax.set_ylabel(metric_label)
        train_ax.grid(True, alpha=0.3)
        train_ax.legend(fontsize=8, loc='best')

        val_ax.set_xlabel('Epoch')
        val_ax.set_ylabel(metric_label)
        val_ax.grid(True, alpha=0.3)
        val_ax.legend(fontsize=8, loc='best')

    plt.tight_layout()
    
    os.makedirs('plots', exist_ok=True)
    
    plt.savefig('plots/training_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Training results saved to 'plots/training_results.png'")
