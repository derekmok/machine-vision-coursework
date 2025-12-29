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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] colab_type="text" id="view-in-github"
# <a href="https://colab.research.google.com/github/derekmok/machine-vision-coursework/blob/main/Machine_Vision_Final_Lab_Model_Training_MP.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% colab={"base_uri": "https://localhost:8080/"} id="d4XrZuAWeoUc" outputId="704d370d-b09a-4427-c92f-1d02ca74b0d7"
# !pip install boto3 -q
# !pip install opencv-python torch numpy torchvision mediapipe

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
# !mkdir -p .models
# !wget -nc -O .models/pose_landmarker.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task



import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import os
import cv2
import numpy as np
from torchvision import tv_tensors
from torchvision.transforms import v2
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
import mediapipe as mp 


class RandomHorizontalFlipLandmarks:
    """Randomly flip landmark sequences horizontally.
    
    This transform performs a horizontal flip on pose landmarks by:
    1. Mirroring x-coordinates around the center (0.5 for normalized coordinates)
    2. Swapping left and right landmark pairs to maintain anatomical consistency
    
    The landmark layout is assumed to be (T, 24) where 24 = 8 landmarks × 3 coords:
    - Indices 0-2: Left Shoulder (x, y, z)
    - Indices 3-5: Right Shoulder (x, y, z)
    - Indices 6-8: Left Elbow (x, y, z)
    - Indices 9-11: Right Elbow (x, y, z)
    - Indices 12-14: Left Wrist (x, y, z)
    - Indices 15-17: Right Wrist (x, y, z)
    - Indices 18-20: Left Hip (x, y, z)
    - Indices 21-23: Right Hip (x, y, z)
    
    Args:
        p: Probability of applying the flip. Default is 0.5.
    """
    
    # Pairs of (left_start_idx, right_start_idx) for each landmark pair
    # Each landmark has 3 values (x, y, z), so we swap in groups of 3
    LANDMARK_PAIRS = [
        (0, 3),    # Left/Right Shoulder
        (6, 9),    # Left/Right Elbow
        (12, 15),  # Left/Right Wrist
        (18, 21),  # Left/Right Hip
    ]
    
    # X-coordinate indices (every 3rd value starting at 0)
    X_INDICES = [0, 3, 6, 9, 12, 15, 18, 21]
    
    def __init__(self, p: float = 0.5):
        if not 0 <= p <= 1:
            raise ValueError(f"Probability p must be in [0, 1], got {p}")
        self.p = p
    
    def __call__(self, landmarks: torch.Tensor) -> torch.Tensor:
        """Apply random horizontal flip to landmarks.
        
        Args:
            landmarks: Tensor of shape (T, 24) containing landmark coordinates.
            
        Returns:
            Flipped landmarks tensor of the same shape, or original if not flipped.
        """
        if torch.rand(1).item() >= self.p:
            return landmarks
        
        # Clone to avoid modifying original tensor
        flipped = landmarks.clone()
        
        # 1. Mirror x-coordinates around center (0.5)
        # For normalized coordinates in [0, 1], flipped_x = 1 - x
        for x_idx in self.X_INDICES:
            flipped[:, x_idx] = 1.0 - flipped[:, x_idx]
        
        # 2. Swap left and right landmark pairs
        for left_start, right_start in self.LANDMARK_PAIRS:
            # Swap all 3 coordinates (x, y, z) for each pair
            temp = flipped[:, left_start:left_start+3].clone()
            flipped[:, left_start:left_start+3] = flipped[:, right_start:right_start+3]
            flipped[:, right_start:right_start+3] = temp
        
        return flipped
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class RandomTemporalJitter:
    """Randomly subsample frames from the landmark sequence.
    
    This transform simulates temporal variations by randomly dropping frames
    from the sequence, effectively creating a faster or jittered version of
    the motion.
    
    Args:
        p: Probability of applying the jitter. Default is 0.5.
        drop_ratio: Tuple of (min_ratio, max_ratio) for the fraction of frames
                    to keep. Default is (0.7, 1.0), meaning 70-100% of frames
                    are kept.
    """
    
    def __init__(self, p: float = 0.5, drop_ratio: tuple = (0.7, 1.0)):
        if not 0 <= p <= 1:
            raise ValueError(f"Probability p must be in [0, 1], got {p}")
        if not (0 < drop_ratio[0] <= drop_ratio[1] <= 1.0):
            raise ValueError(f"drop_ratio must satisfy 0 < min <= max <= 1, got {drop_ratio}")
        self.p = p
        self.drop_ratio = drop_ratio
    
    def __call__(self, landmarks: torch.Tensor) -> torch.Tensor:
        """Apply random temporal jitter to landmarks.
        
        Args:
            landmarks: Tensor of shape (T, 24) containing landmark coordinates.
            
        Returns:
            Jittered landmarks tensor of shape (T', 24) where T' <= T.
        """
        if torch.rand(1).item() >= self.p:
            return landmarks
        
        T = landmarks.shape[0]
        if T <= 2:  # Don't jitter very short sequences
            return landmarks
        
        # Randomly determine how many frames to keep
        keep_ratio = torch.empty(1).uniform_(self.drop_ratio[0], self.drop_ratio[1]).item()
        num_keep = max(2, int(T * keep_ratio))  # Keep at least 2 frames
        
        # Randomly select frame indices (sorted to maintain temporal order)
        indices = torch.randperm(T)[:num_keep].sort().values
        
        return landmarks[indices]
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p}, drop_ratio={self.drop_ratio})"


class RandomScale:
    """Randomly scale landmark coordinates.
    
    This transform simulates distance variations by scaling all landmark
    coordinates around their center. This is useful for making the model
    robust to subjects at different distances from the camera.
    
    Args:
        p: Probability of applying the scaling. Default is 0.5.
        scale_range: Tuple of (min_scale, max_scale). Default is (0.8, 1.2).
    """
    
    def __init__(self, p: float = 0.5, scale_range: tuple = (0.8, 1.2)):
        if not 0 <= p <= 1:
            raise ValueError(f"Probability p must be in [0, 1], got {p}")
        if scale_range[0] <= 0 or scale_range[0] > scale_range[1]:
            raise ValueError(f"scale_range must satisfy 0 < min <= max, got {scale_range}")
        self.p = p
        self.scale_range = scale_range
    
    def __call__(self, landmarks: torch.Tensor) -> torch.Tensor:
        """Apply random scaling to landmarks.
        
        Args:
            landmarks: Tensor of shape (T, 24) containing landmark coordinates.
            
        Returns:
            Scaled landmarks tensor of the same shape.
        """
        if torch.rand(1).item() >= self.p:
            return landmarks
        
        # Random scale factor
        scale = torch.empty(1).uniform_(self.scale_range[0], self.scale_range[1]).item()
        
        # Clone to avoid modifying original
        scaled = landmarks.clone()
        
        # Compute the center of all landmarks across all frames
        # We scale around the mean position to keep landmarks centered
        center = scaled.mean(dim=0, keepdim=True)
        
        # Scale around center
        scaled = center + (scaled - center) * scale
        
        return scaled
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p}, scale_range={self.scale_range})"


class RandomNoise:
    """Add random Gaussian noise to landmark coordinates.
    
    This transform adds small random perturbations to simulate detection
    noise and make the model more robust to imprecise landmark detection.
    
    Args:
        p: Probability of applying noise. Default is 0.5.
        std: Standard deviation of the Gaussian noise. Default is 0.01.
             For normalized coordinates in [0, 1], 0.01 is about 1% noise.
    """
    
    def __init__(self, p: float = 0.5, std: float = 0.01):
        if not 0 <= p <= 1:
            raise ValueError(f"Probability p must be in [0, 1], got {p}")
        if std < 0:
            raise ValueError(f"std must be non-negative, got {std}")
        self.p = p
        self.std = std
    
    def __call__(self, landmarks: torch.Tensor) -> torch.Tensor:
        """Apply random Gaussian noise to landmarks.
        
        Args:
            landmarks: Tensor of shape (T, 24) containing landmark coordinates.
            
        Returns:
            Noisy landmarks tensor of the same shape.
        """
        if torch.rand(1).item() >= self.p:
            return landmarks
        
        # Add Gaussian noise
        noise = torch.randn_like(landmarks) * self.std
        return landmarks + noise
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p}, std={self.std})"


class Compose:
    """Compose multiple transforms together.
    
    Applies a sequence of transforms in order.
    
    Args:
        transforms: List of transform objects to apply.
    """
    
    def __init__(self, transforms: list):
        self.transforms = transforms
    
    def __call__(self, landmarks: torch.Tensor) -> torch.Tensor:
        """Apply all transforms in sequence.
        
        Args:
            landmarks: Tensor of shape (T, 24) containing landmark coordinates.
            
        Returns:
            Transformed landmarks tensor.
        """
        for transform in self.transforms:
            landmarks = transform(landmarks)
        return landmarks
    
    def __repr__(self) -> str:
        lines = [f"{self.__class__.__name__}(["]
        for t in self.transforms:
            lines.append(f"    {t},")
        lines.append("])")
        return "\n".join(lines)


class RandomSequenceRepeat:
    """Repeat landmark sequences to generate training examples with higher counts.
    
    This label-aware transform concatenates a sequence with itself one or more
    times to create a training example representing more repetitions of the
    exercise. For example, a sequence with 2 push-ups repeated twice becomes
    a sequence representing 4 push-ups.
    
    Note: This transform takes (landmarks, label) as input and returns
    (new_landmarks, new_label), unlike regular transforms that only modify
    landmarks.
    
    Args:
        p: Probability of applying the repeat. Default is 0.5.
        max_count: Maximum allowed count after repetition. Default is 10.
        max_repeats: Maximum number of times to repeat (not including original).
                     Default is 3 (so sequence can be 1x to 4x original).
    """
    
    def __init__(self, p: float = 0.5, max_count: int = 10, max_repeats: int = 3):
        if not 0 <= p <= 1:
            raise ValueError(f"Probability p must be in [0, 1], got {p}")
        if max_count <= 0:
            raise ValueError(f"max_count must be positive, got {max_count}")
        if max_repeats <= 0:
            raise ValueError(f"max_repeats must be positive, got {max_repeats}")
        self.p = p
        self.max_count = max_count
        self.max_repeats = max_repeats
    
    def __call__(self, landmarks: torch.Tensor, label: int) -> tuple:
        """Apply random sequence repetition.
        
        Args:
            landmarks: Tensor of shape (T, 24) containing landmark coordinates.
            label: Integer count (e.g., number of push-ups).
            
        Returns:
            Tuple of (repeated_landmarks, new_label) where:
            - repeated_landmarks: Tensor of shape (T * num_repeats, 24)
            - new_label: Integer = label * num_repeats
        """
        if torch.rand(1).item() >= self.p:
            return landmarks, label
        
        if label <= 0:
            return landmarks, label
        
        # Calculate maximum allowed repeats based on max_count constraint
        max_allowed_repeats = self.max_count // label
        
        if max_allowed_repeats <= 1:
            # Can't repeat without exceeding max_count
            return landmarks, label
        
        # Cap at max_repeats parameter (total copies, not additional copies)
        max_allowed_repeats = min(max_allowed_repeats, self.max_repeats + 1)
        
        # Randomly choose number of total copies (at least 2 for a repeat to happen)
        num_copies = torch.randint(2, max_allowed_repeats + 1, (1,)).item()
        
        # Concatenate the sequence
        repeated_landmarks = torch.cat([landmarks] * num_copies, dim=0)
        new_label = label * num_copies
        
        return repeated_landmarks, new_label
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p}, max_count={self.max_count}, max_repeats={self.max_repeats})"

class VideoDataset(Dataset):
    """Dataset for extracting pose landmarks from videos using MediaPipe.
    
    Extracts the following landmarks per frame:
    - Left/Right Shoulder (indices 11, 12)
    - Left/Right Elbow (indices 13, 14)
    - Left/Right Wrist (indices 15, 16)
    - Left/Right Hip (indices 23, 24)
    
    Returns a tensor of shape (T, 24) where T is the number of frames
    and 24 = 8 landmarks × 3 coordinates (x, y, z).
    """

    # MediaPipe Pose landmark indices for the body parts of interest
    LANDMARK_INDICES = [
        11, 12,  # Left/Right Shoulder
        13, 14,  # Left/Right Elbow
        15, 16,  # Left/Right Wrist
        23, 24,  # Left/Right Hip
    ]

    # Default path to the pose landmarker model file
    DEFAULT_MODEL_PATH = ".models/pose_landmarker.task"
    # Default directory for caching extracted landmarks
    DEFAULT_CACHE_DIR = ".landmark_cache"

    def __init__(self, video_dir, transform=None, label_transform=None, model_path=None, cache_dir=None):
        """Initialize the dataset.
        
        Args:
            video_dir: Path to directory containing video files.
            transform: Optional transform to apply to the landmark sequence only.
            label_transform: Optional transform that takes (landmarks, label) and
                           returns (transformed_landmarks, new_label). Applied after
                           regular transform. Used for augmentations like sequence
                           repetition that modify both landmarks and labels.
            model_path: Path to the .models/pose_landmarker.task model file.
                        Defaults to '.models/pose_landmarker.task' in current directory.
            cache_dir: Path to directory for caching extracted landmarks.
                       Defaults to '.landmark_cache' in current directory.
        """
        self.video_dir = video_dir
        self.transform = transform
        self.label_transform = label_transform
        self.model_path = model_path or self.DEFAULT_MODEL_PATH
        self.cache_dir = cache_dir or self.DEFAULT_CACHE_DIR
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)

        self.video_files = [
            f for f in os.listdir(video_dir)
            if f.endswith(('.mp4', '.avi', '.mov'))
        ]

        self.labels = [
            int(f.split('_')[0]) for f in self.video_files
        ]

    def _create_landmarker(self):
        """Create a PoseLandmarker instance configured for video processing."""
        base_options = mp_tasks.BaseOptions(model_asset_path=self.model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False
        )
        return vision.PoseLandmarker.create_from_options(options)

    def __len__(self):
        return len(self.video_files)

    def _get_cache_path(self, video_filename):
        """Get the cache file path for a video file.
        
        Args:
            video_filename: Name of the video file (not full path).
            
        Returns:
            Path to the corresponding cache file (.pt extension).
        """
        # Replace video extension with .pt for cache file
        cache_filename = os.path.splitext(video_filename)[0] + '.pt'
        return os.path.join(self.cache_dir, cache_filename)

    def __getitem__(self, idx):
        video_filename = self.video_files[idx]
        cache_path = self._get_cache_path(video_filename)
        
        # Try to load from cache first
        if os.path.exists(cache_path):
            landmarks_sequence = torch.load(cache_path, weights_only=True)
        else:
            # Extract landmarks and save to cache
            video_path = os.path.join(self.video_dir, video_filename)
            landmarks_sequence = self._extract_landmarks(video_path)
            torch.save(landmarks_sequence, cache_path)
        
        label = self.labels[idx]

        # Apply regular transform (landmarks only)
        if self.transform:
            landmarks_sequence = self.transform(landmarks_sequence)
        
        # Apply label-aware transform (modifies both landmarks and label)
        if self.label_transform:
            landmarks_sequence, label = self.label_transform(landmarks_sequence, label)

        return landmarks_sequence, label

    def _extract_landmarks(self, path):
        """Extract pose landmarks from a video using MediaPipe.
        
        Args:
            path: Path to the video file.
            
        Returns:
            Tensor of shape (T, 24) containing landmark coordinates,
            where T is the number of frames and 24 = 8 landmarks × 3 (x, y, z).
            If no pose is detected in a frame, landmarks are set to 0.
        """
        cap = cv2.VideoCapture(path)
        landmarks_list = []
        
        # Get video FPS for timestamp calculation
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0  # Default to 30 FPS if unknown
        
        frame_duration_ms = 1000.0 / fps
        frame_idx = 0

        # Create a new landmarker for each video (VIDEO mode requires sequential timestamps)
        landmarker = self._create_landmarker()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert BGR to RGB for MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Create MediaPipe Image from numpy array
                mp_frame = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=frame_rgb
                )
                
                # Calculate timestamp in milliseconds for this frame
                timestamp_ms = int(frame_idx * frame_duration_ms)
                
                # Detect pose landmarks
                result = landmarker.detect_for_video(mp_frame, timestamp_ms)

                # Extract landmarks for the frame
                frame_landmarks = []
                if result.pose_landmarks and len(result.pose_landmarks) > 0:
                    pose_landmarks = result.pose_landmarks[0]
                    for idx in self.LANDMARK_INDICES:
                        landmark = pose_landmarks[idx]
                        frame_landmarks.extend([landmark.x, landmark.y, landmark.z])
                else:
                    # No pose detected, fill with zeros
                    frame_landmarks = [0.0] * (len(self.LANDMARK_INDICES) * 3)

                landmarks_list.append(frame_landmarks)
                frame_idx += 1

        finally:
            landmarker.close()
            cap.release()

        # Convert to tensor: shape (T, 24)
        landmarks_tensor = torch.tensor(landmarks_list, dtype=torch.float32)

        return landmarks_tensor


def collate_fn(batch):
    """Pad all landmark sequences to the maximum length in the batch.
    
    Args:
        batch: List of (landmarks, label) tuples where landmarks has shape (T, 24).
        
    Returns:
        landmarks_batch: Tensor of shape (B, max_seq_len, 24).
        labels_batch: Tensor of shape (B,).
        lengths: Tensor of shape (B,) containing original sequence lengths.
    """
    landmarks_list, labels = zip(*batch)

    # Find the maximum sequence length in this batch
    lengths = [landmarks.shape[0] for landmarks in landmarks_list]
    max_len = max(lengths)

    padded_landmarks = []
    for landmarks in landmarks_list:
        num_frames = landmarks.shape[0]
        if num_frames < max_len:
            # Pad with zeros: shape (max_len - num_frames, 24)
            padding = torch.zeros(max_len - num_frames, landmarks.shape[1])
            landmarks = torch.cat([landmarks, padding], dim=0)
        padded_landmarks.append(landmarks)

    landmarks_batch = torch.stack(padded_landmarks, dim=0)
    labels_batch = torch.tensor(labels)
    lengths_batch = torch.tensor(lengths)

    return landmarks_batch, labels_batch, lengths_batch


def get_dataloaders(video_dir, batch_size=4, val_split=0.2, train_transform=None, train_label_transform=None):
    """Create train and validation dataloaders for landmark sequences.
    
    Args:
        video_dir: Path to directory containing video files.
        batch_size: Number of samples per batch.
        val_split: Fraction of data to use for validation.
        train_transform: Optional transform to apply to training landmarks only.
                         Validation data will not have any transform applied.
        train_label_transform: Optional label-aware transform that takes
                               (landmarks, label) and returns (new_landmarks, new_label).
                               Only applied to training data.
        
    Returns:
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
    """
    
    # First, create a dataset without transforms to determine the split indices
    full_dataset = VideoDataset(video_dir, transform=None)
    
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    # Get the indices for train and validation splits
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(full_dataset), generator=generator).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create separate datasets for train (with augmentation) and val (without)
    train_dataset = VideoDataset(
        video_dir, 
        transform=train_transform,
        label_transform=train_label_transform
    )
    val_dataset = VideoDataset(video_dir, transform=None)
    
    # Use Subset to apply the split indices
    from torch.utils.data import Subset
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    print(f"Train: {len(train_subset)} videos, Val: {len(val_subset)} videos")
    if train_transform:
        print(f"Train transform: {train_transform}")
    if train_label_transform:
        print(f"Train label transform: {train_label_transform}")
    print()

    return train_loader, val_loader


video_dir = './video-data'

train_loader, val_loader = get_dataloaders(video_dir, batch_size=4, val_split=0.2)

for landmarks, labels, lengths in train_loader:
    print(f"Landmarks shape: {landmarks.shape}")  # (B, max_seq_len, 24)
    print(f"Labels: {labels}")
    print(f"Sequence lengths: {lengths}")
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

class CausalConv1d(nn.Module):
    """ A 1D convolution that does not look into the future (causal). """
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              padding=self.padding, dilation=dilation)

    def forward(self, x):
        # x shape: (batch, channels, seq_len)
        x = self.conv(x)
        # Remove the padding from the end to keep length consistent
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        return x

class SimpleVideoClassifier(nn.Module):
    def __init__(self, num_joints, num_channels=64):
        super().__init__()
        input_dim = num_joints * 3 # X, Y, Z per joint

        # Stack of dilated convolutions
        # Dilation increases receptive field: 1, 2, 4, 8...
        # This allows the net to see long-range patterns (like a slow pushup)
        self.net = nn.Sequential(
            CausalConv1d(input_dim, num_channels, kernel_size=3, dilation=1),
            nn.ReLU(),
            CausalConv1d(num_channels, num_channels, kernel_size=3, dilation=2),
            nn.ReLU(),
            CausalConv1d(num_channels, num_channels, kernel_size=3, dilation=4),
            nn.ReLU(),
            CausalConv1d(num_channels, num_channels, kernel_size=3, dilation=8),
            nn.ReLU(),
        )
        
        # Final regressor: Output 1 value per frame (the "density")
        self.regressor = nn.Conv1d(num_channels, 1, kernel_size=1)
        
        # Force positive output (counts can't be negative)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 1. Permute for Conv1d: (batch, input_dim, seq_len)
        x = x.permute(0, 2, 1)
        
        # 2. Extract Features
        features = self.net(x)
        
        # 3. Predict per-frame density
        density = self.regressor(features) # Shape: (batch, 1, seq_len)
        density = self.relu(density)
        
        # 4. Transpose back
        density = density.permute(0, 2, 1) # Shape: (batch, seq_len, 1)
        
        return density



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
import torch.nn.functional as F


def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (landmarks, target, _) in enumerate(train_loader):
        landmarks, target = landmarks.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(landmarks)
        total_count = output.sum(dim=1).squeeze()
        loss = F.mse_loss(total_count, target.float())
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = torch.round(total_count)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

        # if batch_idx % 4 == 0:
        #     print(f"Batch {batch_idx}/{len(train_loader)}: Loss {loss.item():.4f}, Accuracy {correct / total:.4f}, Pred {pred}, Target {target}, Correct {correct}")

    return total_loss / len(train_loader), correct / total


def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for landmarks, target, _ in test_loader:
            landmarks, target = landmarks.to(device), target.to(device)
            output = model(landmarks)
            total_count = output.sum(dim=1).squeeze()
            total_loss += F.mse_loss(total_count, target.float())
            pred = torch.round(total_count)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    return total_loss / len(test_loader), correct / total


def train_model(epochs=650, lr=1e-3):
    """Train and return your model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    model = SimpleVideoClassifier(num_joints=8).to(device)
    print("Instantiated model.\n")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Create training transform with data augmentation
    # Each transform is applied independently with its own probability
    train_transform = Compose([
        RandomHorizontalFlipLandmarks(p=0.5),  # Mirror left/right
        # RandomRotation(p=0.5, angle_range=(-15.0, 15.0)),  # Simulate camera/subject tilt
        RandomTemporalJitter(p=0.5, drop_ratio=(0.7, 1.0)),  # Random frame dropping
        RandomScale(p=0.5, scale_range=(0.8, 1.2)),  # Simulate distance variations
        RandomNoise(p=0.5, std=0.01),  # Simulate detection noise
    ])
    
    # Label-aware transform for sequence repetition
    # Repeats sequences to generate higher counts (max 10 pushups)
    train_label_transform = RandomSequenceRepeat(p=0.3, max_count=10, max_repeats=3)
    
    train_loader, val_loader = get_dataloaders(
        video_dir='./video-data',
        batch_size=4,
        train_transform=train_transform,
        train_label_transform=train_label_transform
    )
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

