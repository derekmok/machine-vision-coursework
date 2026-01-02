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

# %% colab={"base_uri": "https://localhost:8080/"} id="TvqxH1YBYCUw" outputId="e27c4b8f-b316-4313-d7bd-3cf63602e428"
# Here is a basic implementation of the above two TODOs. You can assume the first TODO is completed correctly.

# Please modify this code to suit you best, as you decide on your preferred model architecture.

# For example, below here we are padding every video to 1,000 frames. That may or may not be a good idea.

from data_loader import VideoDataset

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
import torch
from neural_net.temporal_conv_net import TCNPushUpCounter


# %% [markdown] id="3Bou97f8czAu"
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
import torch.optim as optim
from neural_net.ensemble_trainer import EnsembleTrainer
from feature_engineering.transforms import Compose, RandomScaling, RandomNoise, RandomTimeWarp, RandomSequenceReverse, \
    RandomSequenceRepeat, RandomHorizontalFlipLandmarks, RandomDropout
from neural_net.kl_divergence_loss import KLDivergenceDensityLoss
from neural_net.loocv_trainer import LOOCVTrainer


def train_model():
    trainer = LOOCVTrainer(
        model_factory=lambda: TCNPushUpCounter(),
        loss_fn=KLDivergenceDensityLoss(lambda_count=1e-1),
        optimizer_factory=lambda parameters : optim.AdamW(parameters),
        patience=100,
        max_epochs=1000,
    )

    return trainer.train(
        dataset=VideoDataset("video-data"),
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
# !mkdir -p .models && wget --no-clobber -O .models/pose_landmarker.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
torch.manual_seed(42)
training_results = train_model()

# %% [markdown] id="W7gmJS-yn2qc"
# # Evaluation

# %% [markdown] id="hzS5ADbDn6Yr"
# ## TODO 6
#
# Include any code which you feel is useful for evaluating your model performance below.
