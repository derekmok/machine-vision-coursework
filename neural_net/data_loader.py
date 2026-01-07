import os
from typing import Callable, Tuple

import torch
from torch.utils.data import Dataset

from feature_engineering.pose_feature_extractor import PoseFeatureExtractor, PoseExtractionResult

DEFAULT_MODEL_PATH = ".models/pose_landmarker.task"
DEFAULT_CACHE_DIR = ".feature_cache"


class TransformDataset(Dataset):
    """Wrapper dataset that applies a transform to an underlying dataset."""
    
    def __init__(self, dataset: Dataset, transform: Callable = lambda x : x):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sequence, density_map, label = self.dataset[idx]
        
        return self.transform(sequence, density_map, label)

class VideoDataset(Dataset):
    """Dataset for loading videos from a folder. Labels from filename prefix.
    
    Args:
        video_dir: Directory containing video files
        media_pipe_model_path: Path to MediaPipe pose model
        cache_dir: Directory to cache extracted features
        is_inference: If True, do not compute ground truth density maps
    """

    def __init__(
            self,
            video_dir,
            feature_processor: Callable[[PoseExtractionResult], Tuple],
            media_pipe_model_path=DEFAULT_MODEL_PATH,
            cache_dir=DEFAULT_CACHE_DIR,
            is_inference=False,
    ):
        self.video_dir = video_dir
        self.cache_dir = cache_dir
        self.feature_extractor = PoseFeatureExtractor(media_pipe_model_path, compute_density_map=not is_inference)
        self.feature_processor = feature_processor

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

    def __getitem__(self, idx: int):
        video_filename = self.video_files[idx]
        cache_path = self._get_cache_path(video_filename)
        
        if os.path.exists(cache_path):
            result = torch.load(cache_path, weights_only=False)
        else:
            video_path = os.path.join(self.video_dir, video_filename)
            result = self.feature_extractor.extract(video_path)
            torch.save(result, cache_path)

        label = self.labels[idx]

        return self.feature_processor(result) + (label,)


    def _get_cache_path(self, video_filename):
        cache_filename = os.path.splitext(video_filename)[0] + '.pt'
        return os.path.join(self.cache_dir, cache_filename)

    @staticmethod
    def for_inference(video_dir: str, cache_dir: str=DEFAULT_CACHE_DIR) -> 'VideoDataset':
        return VideoDataset(
            video_dir,
            lambda features : (features.angle_sequence,),
            is_inference=True, cache_dir=cache_dir
        )
