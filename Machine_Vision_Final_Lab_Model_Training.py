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

# %% [markdown] colab_type="text" id="view-in-github"
# <a href="https://colab.research.google.com/github/derekmok/machine-vision-coursework/blob/main/Machine_Vision_Final_Lab_Model_Training.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% colab={"base_uri": "https://localhost:8080/"} id="d4XrZuAWeoUc" outputId="704d370d-b09a-4427-c92f-1d02ca74b0d7"
# !git init .
# !git remote add origin https://github.com/derekmok/machine-vision-coursework.git
# !git pull origin main

# !pip install -r requirements.txt

# %% [markdown] id="PMr99Yo7x8N1"
# ## Download the data

# %% [markdown] id="4YhoE1nF2Pee"
# The data for this assignment has been made available and is downloadable to disk by running the below cell.
#

# %% colab={"base_uri": "https://localhost:8080/"} id="B5ekP01hR9VV" outputId="b7db6f1a-0f78-422e-9410-527901fdc226"
import os

import boto3
from botocore import UNSIGNED
from botocore.config import Config

# Connect to S3 without authentication (public bucket)
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

bucket_name = 'prism-mvta'
prefix = 'training-and-validation-data/'
download_dir = './video-data'

os.makedirs(download_dir, exist_ok=True)

# List all objects in the S3 path
paginator = s3.get_paginator('list_objects_v2')
pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

video_names = []

for page in pages:
    if 'Contents' not in page:
        print("No files found at the specified path! Go and complain to the TAs!")
        break

    for obj in page['Contents']:
        key = obj['Key']
        filename = os.path.basename(key)

        if not filename:
            continue

        video_names.append(filename)

        local_path = os.path.join(download_dir, filename)
        print(f"Downloading: {filename}")
        s3.download_file(bucket_name, key, local_path)

print("\n" + "="*50)
print("Downloaded videos:")
print("="*50)
for name in video_names:
    print(name)

print(f"\nTotal: {len(video_names)} files")

# %% [markdown] id="5wPAlvHdXj3a"
# These videos are now available in the folder "video-data". You can click on the folder icon on the left-hand-side of this screen to see the videos in a file explorer.

# %% [markdown] id="vtisgbeiYiH_"
# # Create your Datasets and Dataloaders

# %% [markdown] id="Xo9J9hXLeCdY"
# Some example code for approaching the first *two* TODOs is given below just to get you started. No starter code is given for the third TODO.
#
# Note, the below code is very rough skeleton code. Make no assumptions as to the correct manner to architect your model based on the structure of this code.
#
# Please feel free to (if not encouraged to) change every single line of the below code (change it to best suit your chosen model architecture, in the next section).

# %% [markdown] id="XzDMxGLnYa0s"
# ### TODO 1 (This is mostly already done for you - Please see the v1 provided below)
#
# Each video in the folder is prefixed by a number. That number corresponds to the number of distinct pushups visible in the video. Write code to iterate over each video in the folder, and extract the corresponding target associated with the video.

# %% [markdown] id="74PvwbsYYMlD"
# ### TODO 2 (This is also mostly already done for you - Please see the v1 provided below)
#
#
# Divide the data into training and validation sets.
#
# Optionally, you can also create out your own test set to assess your performance.

# %% [markdown] id="hEaV_5oXZQRc"
# ### TODO 3
#
# Any preprocessing or augmentation of your data which you deem required, should (probably) go here. You are also free to include your data-augmentation code later, though doing it before creating your dataloaders is probably a good idea.
#
# If you complete this TODO, to maintain experimental hygiene, feel free to modify the code which was provided for TODOs 1 and 2.

# %% colab={"base_uri": "https://localhost:8080/"} id="TvqxH1YBYCUw" outputId="e27c4b8f-b316-4313-d7bd-3cf63602e428"
# Dataset is implemented in "data_loader.py"
# Dataloaders are created in neural_net/ensemble_trainer.py
from neural_net.data_loader import VideoDataset

# %% [markdown] id="YVPYRadrZdty"
# # Create a Model

# %% [markdown] id="UrpSDGMWaBR3"
# For this assignment, we request you use PyTorch. Below is an example of how to instantiate a very basic PyTorch model.
#
# Note, this model below needs a _lot_ of work.
#
# Please include your code for creating your model below.
#
# The only constraint here is that you define a Python object which inherits from a PyTorch nn.Module object. Beyond that, please feel free to implement anything you like: Transformer, Vision Transformer, MLP, CNN, etc.

# %% [markdown] id="sRolEQeAbxsx"
# ### TODO 4
#
# Create your model.

# %% id="H5FlYz3paNxu"
# The model is implemented in the "neural_net/temporal_conv_net.py"
# and "neural_net/ensemble_model.py"
from neural_net.temporal_conv_net import TCNPushUpCounter
from neural_net.ensemble_model import EnsembleModel

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


# %% [markdown] id="3Bou97f8czAu"
#
# # Train your Model

# %% [markdown] id="VtSqu_ZkcFxs"
# ### TODO 5
#
# Training time! Please include your training code below.
#
# As per above, please feel free (and encouraged) to rip out all of the below code and replace with your (much better) code.
#
# The below should just be used as an example to get you started.

# %% id="EueH4HSdcLlE"
import torch
import torch.optim as optim
from neural_net.ensemble_trainer import EnsembleTrainer
from feature_engineering.transforms import Compose, RandomScaling, RandomNoise, RandomTimeWarp, RandomSequenceReverse, \
    RandomSequenceRepeat, RandomHorizontalFlipLandmarks, RandomDropout
from neural_net.density_map_mse_loss import DensityMapMSELoss


def train_model():
    trainer = EnsembleTrainer(
        model_factory=lambda: TCNPushUpCounter(),
        loss_fn=DensityMapMSELoss(),
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


# %% [markdown]
# ## Validation on Unseen Data
#
# Run the trained model against unseen validation data to evaluate generalization.

# %%
from evaluation.validation import (
    evaluate_on_validation_data,
    display_validation_results,
    plot_validation_angle_sequences,
    plot_validation_density_maps,
    plot_peak_frames,
)

validation_results = evaluate_on_validation_data(model)

if validation_results:
    display_validation_results(validation_results)
    plot_validation_angle_sequences(validation_results)
    plot_validation_density_maps(validation_results)
    plot_peak_frames(validation_results)

# %% [markdown]
# ## PCA Trajectory Visualization
#
# Visualize the hidden state trajectory by extracting the 16-dimensional activations
# from the `act3` layer and projecting them to 2D using PCA. Points are connected
# in time order to show how the representation evolves.

# %%
from evaluation.pca_trajectory import plot_pca_trajectories_for_samples

plot_pca_trajectories_for_samples(
    model,
    "video-data",
    video_filenames=[
        "1_dksksjfwijf.mp4",
        "2_dkjd823kjf.mp4",
    ],
);

# %% [markdown]
# ## Layer-wise Temporal Activation Heatmaps
#
# Visualize how activations evolve through each layer of the TCN.
# Shows heatmaps for act1, act2, and act3 with hidden channels on Y-axis and time on X-axis.

# %%
from evaluation.temporal_heatmaps import plot_temporal_activation_heatmaps

plot_temporal_activation_heatmaps(model, "video-data", filename="4_kling_20251206_Text_to_Video_Generate_a_28_0.mp4");

# %% [markdown] id="eAmXb-QC2ChR"
# # Hugging Face

# %% [markdown] id="cl3rU9Ec4uSI"
# It is a requirement of this assignment that you submit your trained model to a repo on Hugging Face, and make it publicly available. Below, we provide code which should help you do this.

# %% [markdown] id="hSUcEj-DoI8K"
# ## TODO 7
#
# Upload your model to HuggingFace

# %% [markdown] id="qtUkkHpCtQaB"
# Install the dependencies:

# %% colab={"base_uri": "https://localhost:8080/"} id="lYdo05DftcBC" outputId="42444b1b-cb84-4893-f6e2-c17cc829f37c"
# !pip install huggingface_hub

# %% [markdown] id="XIlD5u1U5IBo"
# You'll now need to log in to Hugging Face via the command line. To do this, you'll need to generate a token on your Hugging Face account. To generate a token, run the below command, and click on the link which appears.

# %% colab={"base_uri": "https://localhost:8080/"} id="y9d3loOxtf7v" outputId="c26a78c2-a0f1-4298-ad28-3ad896050af4"
# !hf auth login

# %% [markdown] id="IEnw3O5I5kY8"
# The below code will only run if you have already trained a model with variable name 'model'.
#
# The below code will take your trained model, and upload it to a *public* HuggingFace repo in your account called "mv-final-assignment".
#
# (Note - in this example, we have set 'private=False' in the upload_to_hub method. This makes your model public).
#
# You should double-check that your model is in fact public. To do that, you can navigate (in an incognito tab, in a browser) to https://huggingface.co/YOUR_USERNAME/YOUR_MODEL_NAME and see if that page loads. If your model is public, it will. (Simply being able to run the below code will not guarantee that your model is in fact public, because, you have now authenticated yourself with the huggingface CLI).

# %% id="lCq_stTaoeQW"
# YOUR HUGGING FACE USERNAME BELOW
hf_username = 'derekmok'

# %% id="_AdQof5XtWfS"
import torch
from huggingface_hub import HfApi


def save_model(model, path="model.pt"):
    """Save the model weights to a file."""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def upload_to_hub(local_path="model.pt", repo_id=f"{hf_username}/mv-final-assignment"):
    """
    Upload model to Hugging Face Hub.

    Args:
        local_path: Path to your saved model file
        repo_id: Your repo in format "username/model-name"
    """
    api = HfApi()

    # Create the repo first (if it already exists, this will just skip)
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        exist_ok=True,  # Don't error if it already exists
        private=False,  # Make it public so TAs can access
    )

    # Now upload the file
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo="model.pt",
        repo_id=repo_id,
        repo_type="model",
    )

    print(f"Model uploaded to https://huggingface.co/{repo_id}")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":

    save_model(model, "mv-final-assignment.pt")

    upload_to_hub("mv-final-assignment.pt", f"{hf_username}/mv-final-assignment")
