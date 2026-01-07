# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
from neural_net.count_mse_loss import CountMSELoss
from neural_net.ensemble_model import EnsembleModel
from neural_net.temporal_conv_net import TCNPushUpCounter


# %%

def create_ensemble_from_results(fold_results, input_channels=6):
    """Create an EnsembleModel and load weights from training results.

    Args:
        fold_results: List of FoldResult from training
        input_channels: Number of input channels for TCNPushUpCounter

    Returns:
        EnsembleModel with loaded weights
    """
    models = [TCNPushUpCounter(input_channels=input_channels) for _ in range(len(fold_results))]

    state_dicts = [fold.model_state_dict for fold in fold_results]

    ensemble = EnsembleModel.from_pretrained_models(models, state_dicts)

    return ensemble


# %%
import torch
import torch.optim as optim
from neural_net.ensemble_trainer import EnsembleTrainer
from feature_engineering.transforms import Compose, RandomScaling, RandomNoise, RandomTimeWarp, RandomSequenceReverse, \
    RandomSequenceRepeat, RandomHorizontalFlipLandmarks, RandomDropout


def train_model():
    trainer = EnsembleTrainer(
        model_factory=lambda: TCNPushUpCounter(),
        loss_fn=CountMSELoss(),
        optimizer_factory=lambda parameters : optim.AdamW(parameters),
        patience=100,
        max_epochs=1000
    )

    return trainer.train(
        video_data_dir="video-data",
        train_transform=Compose([
            RandomSequenceRepeat(),
            RandomSequenceReverse(),
            RandomHorizontalFlipLandmarks(),
            RandomTimeWarp(p=0.8),
            RandomScaling(),
            RandomNoise(p=1.0),
            RandomDropout()
        ])
    )

# %% colab={"base_uri": "https://localhost:8080/"} id="lHI68u4XhCGB" outputId="c28ba0d4-cb87-4076-eb13-ab93d597ef10"

# setting a manual seed for reproducibility
torch.manual_seed(100)

training_results = train_model()
model = create_ensemble_from_results(training_results, input_channels=6)
print(f"Created ensemble with {len(model)} models")


# %% [markdown] id="W7gmJS-yn2qc"
# # Evaluation

# %% [markdown] id="hzS5ADbDn6Yr"
# ## TODO 6
#
# Include any code which you feel is useful for evaluating your model performance below.

# %% id="y1KwRou4oCkj"
from evaluation.training_plots import plot_training_results

plot_training_results(training_results)

# %% [markdown]
# ## Ensemble Evaluation
#
# Evaluate the ensemble model on the full training dataset.

# %%
import pandas as pd
from evaluation.ensemble_evaluation import (
    evaluate_ensemble_on_dataset,
    plot_density_maps,
    plot_predicted_vs_true,
    plot_wrong_predictions_density_maps,
)

evaluation_results = evaluate_ensemble_on_dataset(model, "video-data")

metrics_df = pd.DataFrame({
    'Metric': [
        'Mean Absolute Error',
        'Exact Match Accuracy',
        'Off-by-One Accuracy'
    ],
    'Value': [
        f"{evaluation_results['mae']:.4f}",
        f"{evaluation_results['exact_match_accuracy']:.2%}",
        f"{evaluation_results['off_by_one_accuracy']:.2%}"
    ]
})

print()
print("Ensemble Evaluation Results on Training Set")
metrics_df.style.hide(axis='index')


# %%
plot_density_maps(evaluation_results, num_samples=6)


# %%
plot_wrong_predictions_density_maps(evaluation_results, num_samples=6)

# %%
plot_predicted_vs_true(evaluation_results)

