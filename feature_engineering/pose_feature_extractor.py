"""Pose feature extraction module for computing joint angles from videos.

This module provides functionality to extract pose landmarks from videos using
MediaPipe and compute joint angles (elbow, shoulder, and hip angles).
"""

import os
import urllib.request
from dataclasses import dataclass

import cv2
import mediapipe as mp
import torch
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from feature_engineering.body_angles import calculate_angles
from feature_engineering.constants import TARGET_FPS
from feature_engineering.density_map import generate_gaussian_density_map
from feature_engineering.heuristic_pushup_counter import CounterParameters, HeuristicPushupCounter

MEDIA_PIPE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"


@dataclass
class PoseExtractionResult:
    """Data class for pose extraction results.
    
    Attributes:
        angle_sequence: Tensor of shape (T, 6) containing raw (unsmoothed) angle features.
        density_map: Tensor of shape (T,) containing the gaussian density map.
    """
    angle_sequence: torch.Tensor
    density_map: torch.Tensor


class PoseFeatureExtractor:
    """Extracts joint angle features from videos using MediaPipe pose estimation.
    
    This class processes video files and computes 6 joint angles per frame:
    - Left and right elbow angles (wrist-elbow-shoulder)
    - Left and right shoulder angles (elbow-shoulder-hip)
    - Left and right body angles (shoulder-hip-knee)
    """
    
    def __init__(self, model_path, target_fps=TARGET_FPS, compute_density_map=True):
        """Initialize the pose feature extractor.
        
        Args:
            model_path: Path to the MediaPipe pose landmarker model file.
            target_fps: Target frame rate for output sequences. All videos will
                be resampled to this frame rate to ensure consistent temporal
                resolution regardless of source video frame rate. Default is 30 FPS.
            compute_density_map: Whether to compute the density map. Set to False
                for inference to skip the heuristic-based pseudo-label generation.
                Default is True.
        """
        self.model_path = model_path
        self.target_fps = target_fps
        self.compute_density_map_flag = compute_density_map
        self._ensure_model_exists()
    
    def _ensure_model_exists(self):
        """Check if the MediaPipe model exists, and download it if not."""
        if os.path.exists(self.model_path):
            return

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        urllib.request.urlretrieve(MEDIA_PIPE_MODEL_URL, self.model_path)
    
    def _resample_to_target_fps(self, angles_tensor, source_fps):
        """Resample the angle sequence to match the target frame rate.
        
        Uses linear interpolation to resample the sequence from the source
        frame rate to the target frame rate.
        
        Args:
            angles_tensor: Tensor of shape (T, 6) containing angle features.
            source_fps: The frame rate of the source video.
            
        Returns:
            Resampled tensor with shape (T', 6) where T' is adjusted to
            match the target frame rate.
        """
        if abs(source_fps - self.target_fps) < 0.01:
            # Frame rates are essentially the same, no resampling needed
            return angles_tensor
        
        num_source_frames = angles_tensor.shape[0]
        duration_seconds = num_source_frames / source_fps
        num_target_frames = int(round(duration_seconds * self.target_fps))
        
        # Interpolate each angle channel separately
        # Using torch.nn.functional.interpolate requires reshaping
        # Shape: (T, 6) -> (1, 6, T) for interpolation -> (T', 6)
        angles_for_interp = angles_tensor.T.unsqueeze(0) # (1, 6, T)
        resampled = torch.nn.functional.interpolate(
            angles_for_interp, 
            size=num_target_frames, 
            mode='linear', 
            align_corners=True
        )
        resampled = resampled.squeeze(0).T # (T', 6)
        
        return resampled
    
    def extract(self, video_path) -> PoseExtractionResult:
        """Extract pose angle features and density map from a video file.
        
        Computes the following 6 angle features per frame:
        - Left elbow angle (wrist-elbow-shoulder)
        - Right elbow angle (wrist-elbow-shoulder)
        - Left shoulder angle (elbow-shoulder-hip)
        - Right shoulder angle (elbow-shoulder-hip)
        - Left body angle (shoulder-hip-knee)
        - Right body angle (shoulder-hip-knee)
        
        Also computes a gaussian density map using signal processing to detect
        push-up positions (valleys in the elbow angle signal).
        
        The output sequence is resampled to the target frame rate (default 30 FPS)
        to ensure consistent temporal resolution regardless of the source video's
        frame rate.
        
        Args:
            video_path: Path to the video file.
            
        Returns:
            PoseExtractionResult
        """
        cap = cv2.VideoCapture(video_path)

        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if source_fps <= 0:
            source_fps = 30.0

        frame_duration_ms = 1000.0 / source_fps
        frame_idx = 0

        landmarker = self._create_landmarker()

        angles_list = []
        print(f"Detecting poses for {video_path}")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=frame_rgb
                )
                
                timestamp_ms = int(frame_idx * frame_duration_ms)
                
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                body_angles = calculate_angles(result)
                angles_list.append(body_angles)
                frame_idx += 1

        finally:
            landmarker.close()
            cap.release()

        raw_angles_tensor = torch.tensor(angles_list, dtype=torch.float32)
        
        raw_angles_tensor_resampled = self._resample_to_target_fps(raw_angles_tensor, source_fps)
        
        params = CounterParameters(
            smoothing_window=21,
            poly_order=3,
            min_prominence=0.11,
            min_distance=5,
            median_filter_size=3
        )
        counter = HeuristicPushupCounter(params)
        
        pushup_results = counter.count_pushups(raw_angles_tensor_resampled)

        if self.compute_density_map_flag:
            density_map = generate_gaussian_density_map(pushup_results.valleys, len(raw_angles_tensor_resampled))
        else:
            density_map = torch.zeros(len(raw_angles_tensor_resampled), dtype=torch.float32)

        return PoseExtractionResult(
            angle_sequence=raw_angles_tensor_resampled,
            density_map=density_map
        )

    def _create_landmarker(self):
        """Create and configure a MediaPipe pose landmarker.
        
        Returns:
            Configured PoseLandmarker instance ready for video processing.
        """
        base_options = mp_tasks.BaseOptions(model_asset_path=self.model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1
        )
        return vision.PoseLandmarker.create_from_options(options)
