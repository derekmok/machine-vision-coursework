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
#     display_name: .venv (3.14.0)
#     language: python
#     name: python3
# ---

# %% [markdown] colab_type="text" id="view-in-github"
# <a href="https://colab.research.google.com/github/derekmok/machine-vision-coursework/blob/main/Machine_Vision_Final_Lab_Model_Training.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% colab={"base_uri": "https://localhost:8080/"} id="d4XrZuAWeoUc" outputId="704d370d-b09a-4427-c92f-1d02ca74b0d7"
# !pip install boto3 -q
# !pip install opencv-python torch numpy torchvision

# %% [markdown] id="PMr99Yo7x8N1"
# ## Download the data

# %% [markdown] id="4YhoE1nF2Pee"
# The data for this assignment has been made available and is downloadable to disk by running the below cell.

# %% colab={"base_uri": "https://localhost:8080/"} id="B5ekP01hR9VV" outputId="b7db6f1a-0f78-422e-9410-527901fdc226"
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import os

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
# Here is a basic implementation of the above two TODOs. You can assume the first TODO is completed correctly.

# Please modify this code to suit you best, as you decide on your preferred model architecture.

# For example, below here we are padding every video to 1,000 frames. That may or may not be a good idea.


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import os
import cv2
import numpy as np
from torchvision import tv_tensors
from torchvision.transforms import v2


class VideoDataset(Dataset):
    """Dataset for loading videos from a folder. Labels from filename prefix."""

    def __init__(self, video_dir, frame_size=(224, 224), transform=None):
        self.video_dir = video_dir
        self.frame_size = frame_size
        self.transform = transform

        self.video_files = [
            f for f in os.listdir(video_dir)
            if f.endswith(('.mp4', '.avi', '.mov'))
        ]

        self.labels = [
            int(f.split('_')[0]) for f in self.video_files
        ]

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = os.path.join(self.video_dir, self.video_files[idx])
        frames = self._load_video(video_path)
        label = self.labels[idx]

        if self.transform:
            frames = self.transform(frames)

        return frames, label

    def _load_video(self, path):
        cap = cv2.VideoCapture(path)

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.frame_size[1], self.frame_size[0]))
            frames.append(frame)

        cap.release()

        frames = torch.from_numpy(np.array(frames)).permute(3, 0, 1, 2).float() / 255.0

        return frames


def collate_fn(batch):
    """Pad all videos to 1000 frames."""
    frames_list, labels = zip(*batch)

    target_frames = 1000

    padded_frames = []
    for frames in frames_list:
        num_frames = frames.shape[1]
        if num_frames < target_frames:
            padding = torch.zeros(frames.shape[0], target_frames - num_frames, frames.shape[2], frames.shape[3])
            frames = torch.cat([frames, padding], dim=1)
        elif num_frames > target_frames:
            frames = frames[:, :target_frames, :, :]
        padded_frames.append(frames)

    frames_batch = torch.stack(padded_frames, dim=0)
    labels_batch = torch.tensor(labels)

    return frames_batch, labels_batch


def get_dataloaders(video_dir, batch_size=4, val_split=0.2, frame_size=(224, 224)):
    """Create train and validation dataloaders."""

    full_dataset = VideoDataset(video_dir, frame_size=frame_size)

    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    print(f"Train: {len(train_dataset)} videos, Val: {len(val_dataset)} videos\n")

    return train_loader, val_loader


video_dir = './video-data'

train_loader, val_loader = get_dataloaders(video_dir, batch_size=4, val_split=0.2)

for frames, labels in train_loader:
    print(f"Frames shape: {frames.shape}")  # (B, C, 1000, H, W)
    print(f"Labels: {labels}")
    break

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
import torch.nn as nn
from huggingface_hub import HfApi, hf_hub_download


class SimpleVideoClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Average over frames, then use a simple CNN
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (B, C, T, H, W)
        # Average over time dimension
        x = x.mean(dim=2)  # (B, C, H, W)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



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


def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

    return total_loss / len(train_loader), correct / total


def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    return total_loss / len(test_loader), correct / total


def train_model(epochs=5, lr=1e-3):
    """Train and return your model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    model = SimpleVideoClassifier().to(device)
    print("Instantiated model.\n")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_loader, val_loader = get_dataloaders(video_dir='./video-data')
    print("Got dataloaders.\n")

    print("Go time. Let the training commence.\n")

    for epoch in range(1, epochs + 1):

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch}/{epochs} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    return model



# %% colab={"base_uri": "https://localhost:8080/"} id="lHI68u4XhCGB" outputId="c28ba0d4-cb87-4076-eb13-ab93d597ef10"
# setting a manual seed for reproducibility
torch.manual_seed(0)
model = train_model()

# %% [markdown] id="W7gmJS-yn2qc"
# # Evaluation

# %% [markdown] id="hzS5ADbDn6Yr"
# ## TODO 6
#
# Include any code which you feel is useful for evaluating your model performance below.

# %% id="y1KwRou4oCkj"
# YOUR CODE HERE

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
hf_username = 'rossamurphy'

# %% id="_AdQof5XtWfS"
import torch
import torch.nn as nn
from huggingface_hub import HfApi, hf_hub_download


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

