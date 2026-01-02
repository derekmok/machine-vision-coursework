import torch
from torch.utils.data import Dataset
import os

from feature_engineering.pose_feature_extractor import PoseFeatureExtractor

DEFAULT_MODEL_PATH = ".models/pose_landmarker.task"
DEFAULT_CACHE_DIR = ".landmark_cache"

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
            landmarks_sequence = cache_data
        else:
            print(f"Extracting poses (pose detection) for: {video_filename}")
            video_path = os.path.join(self.video_dir, video_filename)
            landmarks_sequence = self.feature_extractor.extract_joint_angles(video_path)
            print(f"Pose detection complete for: {video_filename}")
            torch.save(landmarks_sequence, cache_path)

        label = self.labels[idx]

        return landmarks_sequence, label


    def _get_cache_path(self, video_filename):
        cache_filename = os.path.splitext(video_filename)[0] + '.pt'
        return os.path.join(self.cache_dir, cache_filename)
