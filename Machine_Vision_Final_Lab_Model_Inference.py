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
# <a href="https://colab.research.google.com/github/derekmok/machine-vision-coursework/blob/main/Machine_Vision_Final_Lab_Model_Inference.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% colab={"base_uri": "https://localhost:8080/"} id="bSu3U-SNu8v6" outputId="5ddf74be-2b61-4815-ce8a-5c180ac9858a"
# ===== INSTALL DEPENDENCIES =====
# !git init .
# !git remote add origin https://github.com/derekmok/machine-vision-coursework.git
# !git pull origin main

# !pip install -r requirements.txt

# %% id="3Lo27xcqrOMq"
# Import the required libraries
import torch
from torch.utils.data import DataLoader
from huggingface_hub import hf_hub_download
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import os
from tqdm.auto import tqdm
import time

# %% [markdown] id="PMr99Yo7x8N1"
# # Please double, triple, quadruple check that the below code runs without errors before submitting.

# %% [markdown] id="4YhoE1nF2Pee"
# ## TODO 1 - Enter your HuggingFace username below:

# %% id="ENyfncieqs6i"
hf_username = "derekmok"


# %% [markdown] id="0lC0jUquq_06"
# ## TODO 2 - Define your model EXACTLY as you did in your training code (otherwise there will be errors, and, possibly, tears).
#
# Note below the classname is 'YourModelArchitecture'. That's because it literally needs to be YOUR MODEL ARCHITECTURE. This class definition is later referred to below in the 'load_model_from_hub' method. The architecture must match here, or it will not be able to instantiate the model weights correctly once it downloads them from HuggingFace. Pay very close attention to getting this right, please.
#
# Replace the below code, and replace the corresponding line in the 'load_model_from_hub' method.

# %% id="cm-y1pPnOGkK"
# =============================================================================
# 1. MODEL DEFINITION (must match training)
# =============================================================================
from neural_net.temporal_conv_net import TCNPushUpCounter
from neural_net.ensemble_model import EnsembleModel


# %% [markdown] id="qahq0xG2rs4h"
# ## Download the test data from s3, and create the corresponding dataset + dataloader.
#
# There's no TODO for you here. This text is just here to explain to you what this code does.
#
# In this instance, the test data IS the training data you were provided in the Model Training notebook. This is by design. You do not have access to the test data. This is a simple check to make sure the mechanics of this notebook work.
#
# You should achieve the same accuracy here in this notebook, as you did in your previous notebook (random seed notwithstanding).

# %% id="XBukVn9qrnFZ"
# =============================================================================
# DOWNLOAD TEST DATA FROM S3
# =============================================================================

def download_test_data(bucket_name='training-and-validation-data',download_dir='./test-data'):
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    bucket_name = 'prism-mvta'
    prefix = 'training-and-validation-data/'

    os.makedirs(download_dir, exist_ok=True)

    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    video_names = []

    for page in pages:
        if 'Contents' not in page:
            print("No files found at the specified path!")
            break

        print("Downloading test data:\n")
        for obj in tqdm(page['Contents']):
            key = obj['Key']
            filename = os.path.basename(key)

            if not filename:
                continue

            video_names.append(filename)
            local_path = os.path.join(download_dir, filename)
            # print(f"Downloading: {filename}")
            s3.download_file(bucket_name, key, local_path)

    print(f"\nDownloaded {len(video_names)} test videos")
    return download_dir


# ============================================================================= # DATASET AND DATALOADER =============================================================================

from neural_net.data_loader import VideoDataset


# %% [markdown] id="B9PVSdWKsP94"
# ## TODO 3 - Download your model from HuggingFace and instantiate it
#
# Replace line 8 of the below code. Line 8 is where you instantiate YOUR MODEL ARCHITECTURE (which you re-defined above) with the weights you download from HuggingFace. Make sure you get the class name, and the arguments to the __init__ method correct.
#
#
# This code just downloads the same model which you uploaded in the last notebook.

# %% colab={"base_uri": "https://localhost:8080/", "height": 194, "referenced_widgets": ["cfdc1400987345568482d2a09b1bb388", "df6c93ec63634bb39453e5a4b0b7718a", "d05b7e235252405a841ebd52fb9076e7", "f8e5ff077f814a41b4a9fa6f1fe2adc1", "7366dc70bfe743bf951be84626837a38", "0e788161d33f4a26a1bf3f9fa539dc32", "d8cd8f46840a40809e52b19cd6273828", "a885dec216b84294ad34b7628c19741f", "7e8946e7a8d2476a9acc093f3be5c320", "602e1b6c06794838992edb63a52bd236", "b900927fbcee4d579949d0c4c1f185d7"]} id="LWuMOqY_sOdg" outputId="b47126a3-940b-4cc8-b9e1-b837e696fbf0"
# =============================================================================
# DOWNLOAD MODEL FROM HUGGING FACE
# =============================================================================

def load_model_from_hub(repo_id, num_classes=10):
    model_path = hf_hub_download(repo_id=repo_id, filename="model.pt")

    model = EnsembleModel([TCNPushUpCounter(input_channels=6) for _ in range(5)])
    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    print(f"Model loaded from {repo_id}")
    return model

model = load_model_from_hub(f"{hf_username}/mv-final-assignment", num_classes=10)


# %% [markdown] id="NycfLBRksum4"
# ## TODO 4
#
# Make sure the below code correctly evaluates your model performance on the given data!
#
# This is your last chance to verify this before submission.

# %% colab={"base_uri": "https://localhost:8080/"} id="QzgdieGiw4_k" outputId="d2cacecd-19be-4655-cbb9-37ec99be0657"
def evaluate(model, test_loader, dataset, device):
    model.eval()
    correct = 0
    total = 0

    all_preds = []
    all_labels = []
    all_times = []

    print("\n")

    with torch.no_grad():
        for idx, (frames, labels) in enumerate(test_loader):
            frames, labels = frames.to(device), labels.to(device)

            # Time the forward pass
            start_time = time.time()
            outputs = model(frames)
            if device.type == 'cuda':
                torch.cuda.synchronize()  # wait for GPU to finish
            end_time = time.time()

            inference_time = (end_time - start_time) * 1000  # ms
            all_times.append(inference_time)

            # For regression: model returns (count, density_map) tuple
            # Extract the count prediction (first element) and round to integer
            count_predictions = outputs[0]  # shape: [batch, 1]
            preds = torch.round(count_predictions.squeeze(-1)).long()  # shape: [batch]

            for i in range(labels.size(0)):
                batch_idx = idx * test_loader.batch_size + i
                video_name = dataset.video_files[batch_idx]
                pred = preds[i].item()
                true_label = labels[i].item()
                is_correct = "✓" if pred == true_label else "✗"

                print(f"{is_correct}  pred={pred}  true={true_label}  |  {inference_time:>7.1f}ms  |  {video_name}")

            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    return accuracy, all_preds, all_labels, all_times


# =============================================================================
# RUN INFERENCE
# =============================================================================

def run_inference(model, bucket_name='training-and-validation-data'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Download test data
    test_dir = download_test_data(bucket_name, './test-data')

    model = model.to(device)

    # Create dataloader
    test_dataset = VideoDataset.for_inference(test_dir)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    print(f"\nRunning inference on {len(test_dataset)} test videos...")

    # Warmup (optional, helps get consistent GPU timings)
    if device.type == 'cuda':
        dummy = torch.randn(1, 1000, 6).to(device)
        with torch.no_grad():
            _ = model(dummy)
        torch.cuda.synchronize()

    total_start = time.time()
    accuracy, preds, labels, times = evaluate(model, test_loader, test_dataset, device)
    total_end = time.time()

    # Summary
    num_correct = sum(p == l for p, l in zip(preds, labels))
    num_wrong = len(preds) - num_correct

    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Total videos:         {len(preds)}")
    print(f"Correct:              {num_correct}")
    print(f"Incorrect:                {num_wrong}")
    print(f"")
    print(f"ACCURACY:             {accuracy*100:.2f}%")
    print(f"")
    print(f"Total time:           {total_end - total_start:.2f}s")
    print(f"Avg per video:        {sum(times) / len(times):.1f}ms")
    print(f"Min latency:          {min(times):.1f}ms")
    print(f"Max latency:          {max(times):.1f}ms")
    print("="*50)
    return accuracy, preds, labels

_, _, _ = run_inference(model)
