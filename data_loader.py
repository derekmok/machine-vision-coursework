from typing import Any

import torch
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerResult
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

DEFAULT_MODEL_PATH = ".models/pose_landmarker.task"
DEFAULT_CACHE_DIR = ".landmark_cache"

# MediaPipe Pose landmark indices
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26

class VideoDataset(Dataset):
    """Dataset for loading videos from a folder. Labels from filename prefix."""

    def __init__(self, video_dir, transform=None, model_path=DEFAULT_MODEL_PATH, cache_dir=DEFAULT_CACHE_DIR):
        self.video_dir = video_dir
        self.transform = transform
        self.model_path = model_path
        self.cache_dir = cache_dir
        
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
        
        # Try to load from cache first
        if os.path.exists(cache_path):
            landmarks_sequence = torch.load(cache_path, weights_only=True)
        else:
            # Extract landmarks and save to cache
            video_path = os.path.join(self.video_dir, video_filename)
            landmarks_sequence = self._extract_landmarks(video_path)
            torch.save(landmarks_sequence, cache_path)

        label = self.labels[idx]

        if self.transform:
            landmarks_sequence = self.transform(landmarks_sequence)

        return landmarks_sequence, label, len(landmarks_sequence)

    @staticmethod
    def _compute_angle(a, b, c):
        """Compute the angle at vertex B given three 3D points A, B, C.
        
        Args:
            a: numpy array of shape (3,) for point A
            b: numpy array of shape (3,) for point B (vertex)
            c: numpy array of shape (3,) for point C
            
        Returns:
            Angle in degrees at vertex B.
        """
        ba = a - b
        bc = c - b
        
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        return np.degrees(angle)

    def _get_landmark_xyz(self, pose_landmarks, idx):
        """Extract (x, y, z) coordinates for a landmark by index."""
        lm = pose_landmarks[idx]
        return np.array([lm.x, lm.y, lm.z])

    def _extract_landmarks(self, path):
        """Extract pose angle features from a video using MediaPipe.
        
        Computes the following 6 angle features per frame:
        - Left elbow angle (wrist-elbow-shoulder)
        - Right elbow angle (wrist-elbow-shoulder)
        - Left shoulder angle (elbow-shoulder-hip)
        - Right shoulder angle (elbow-shoulder-hip)
        - Left body angle (shoulder-hip-knee)
        - Right body angle (shoulder-hip-knee)
        
        Args:
            path: Path to the video file.
            
        Returns:
            Tensor of shape (T, 6) containing angle features in degrees,
            where T is the number of frames.
            If no pose is detected in a frame, angles are set to 0.
        """
        cap = cv2.VideoCapture(path)
        angles_list = []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        
        frame_duration_ms = 1000.0 / fps
        frame_idx = 0

        landmarker = self._create_landmarker()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                mp_frame = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=frame_rgb
                )
                
                timestamp_ms = int(frame_idx * frame_duration_ms)
                
                result = landmarker.detect_for_video(mp_frame, timestamp_ms)

                body_angles = self._calculate_angles(result)
                angles_list.append(body_angles)
                frame_idx += 1

        finally:
            landmarker.close()
            cap.release()

        angles_tensor = torch.tensor(angles_list, dtype=torch.float32)

        return angles_tensor

    def _calculate_angles(self, result: PoseLandmarkerResult):
        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            pose = result.pose_landmarks[0]

            # Get landmark coordinates
            left_shoulder = self._get_landmark_xyz(pose, LEFT_SHOULDER)
            right_shoulder = self._get_landmark_xyz(pose, RIGHT_SHOULDER)
            left_elbow = self._get_landmark_xyz(pose, LEFT_ELBOW)
            right_elbow = self._get_landmark_xyz(pose, RIGHT_ELBOW)
            left_wrist = self._get_landmark_xyz(pose, LEFT_WRIST)
            right_wrist = self._get_landmark_xyz(pose, RIGHT_WRIST)
            left_hip = self._get_landmark_xyz(pose, LEFT_HIP)
            right_hip = self._get_landmark_xyz(pose, RIGHT_HIP)
            left_knee = self._get_landmark_xyz(pose, LEFT_KNEE)
            right_knee = self._get_landmark_xyz(pose, RIGHT_KNEE)

            # Compute angles
            left_elbow_angle = self._compute_angle(left_wrist, left_elbow, left_shoulder)
            right_elbow_angle = self._compute_angle(right_wrist, right_elbow, right_shoulder)
            left_shoulder_angle = self._compute_angle(left_elbow, left_shoulder, left_hip)
            right_shoulder_angle = self._compute_angle(right_elbow, right_shoulder, right_hip)
            left_body_angle = self._compute_angle(left_shoulder, left_hip, left_knee)
            right_body_angle = self._compute_angle(right_shoulder, right_hip, right_knee)

            frame_angles = [
                left_elbow_angle,
                right_elbow_angle,
                left_shoulder_angle,
                right_shoulder_angle,
                left_body_angle,
                right_body_angle
            ]
        else:
            frame_angles = [0.0] * 6

        return frame_angles

    def _create_landmarker(self):
        base_options = mp_tasks.BaseOptions(model_asset_path=self.model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1
        )
        return vision.PoseLandmarker.create_from_options(options)

    def _get_cache_path(self, video_filename):
        cache_filename = os.path.splitext(video_filename)[0] + '.pt'
        return os.path.join(self.cache_dir, cache_filename)
