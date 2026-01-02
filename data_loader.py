import torch
from torch.utils.data import Dataset
import os
from typing import Callable, Optional
import time

from feature_engineering.pose_feature_extractor import PoseFeatureExtractor

DEFAULT_MODEL_PATH = ".models/pose_landmarker.task"
DEFAULT_CACHE_DIR = ".landmark_cache"


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
        sequence, density_map, label, length = self.dataset[idx]
        
        if self.transform is not None:
            sequence, density_map, label, length = self.transform(sequence, density_map, label, length)
        
        return sequence, density_map, label, length

class VideoDataset(Dataset):
    """Dataset for loading videos from a folder. Labels from filename prefix."""

    def __init__(self, video_dir, media_pipe_model_path=DEFAULT_MODEL_PATH, cache_dir=DEFAULT_CACHE_DIR):
        self.video_dir = video_dir
        self.cache_dir = cache_dir
        self.feature_extractor = PoseFeatureExtractor(media_pipe_model_path)
        
        os.makedirs(self.cache_dir, exist_ok=True)

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
        video_filename = self.video_files[idx]
        cache_path = self._get_cache_path(video_filename)
        
        if os.path.exists(cache_path):
            cache_data = torch.load(cache_path, weights_only=True)
            landmarks_sequence = cache_data['angles']
            density_map = cache_data['density_map']
        else:
            time_start = time.time()
            print(f"Extracting features for: {video_filename}")
            video_path = os.path.join(self.video_dir, video_filename)
            landmarks_sequence, density_map = self.feature_extractor.extract_joint_angles(video_path)
            print(f"Features extracted for: {video_filename} in {(time.time() - time_start):.2f}s")
            cache_data = {
                'angles': landmarks_sequence,
                'density_map': density_map,
            }
            torch.save(cache_data, cache_path)

        label = self.labels[idx]

        return landmarks_sequence, density_map, label, len(landmarks_sequence)


    def _get_cache_path(self, video_filename):
        cache_filename = os.path.splitext(video_filename)[0] + '.pt'
        return os.path.join(self.cache_dir, cache_filename)
