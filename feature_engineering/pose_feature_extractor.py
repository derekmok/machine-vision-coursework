"""Pose feature extraction module for computing joint angles from videos.

This module provides functionality to extract pose landmarks from videos using
MediaPipe and compute joint angles (elbow, shoulder, and hip angles).
"""

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerResult
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

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


class PoseFeatureExtractor:
    """Extracts joint angle features from videos using MediaPipe pose estimation.
    
    This class processes video files and computes 6 joint angles per frame:
    - Left and right elbow angles (wrist-elbow-shoulder)
    - Left and right shoulder angles (elbow-shoulder-hip)
    - Left and right body angles (shoulder-hip-knee)
    """
    
    def __init__(self, model_path, target_fps=30.0):
        """Initialize the pose feature extractor.
        
        Args:
            model_path: Path to the MediaPipe pose landmarker model file.
            target_fps: Target frame rate for output sequences. All videos will
                be resampled to this frame rate to ensure consistent temporal
                resolution regardless of source video frame rate. Default is 30 FPS.
        """
        self.model_path = model_path
        self.target_fps = target_fps
    
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
        if num_source_frames < 2:
            return angles_tensor
        
        # Calculate the duration in seconds and the target number of frames
        duration_seconds = num_source_frames / source_fps
        num_target_frames = max(2, int(round(duration_seconds * self.target_fps)))
        
        # Interpolate each angle channel separately
        # Using torch.nn.functional.interpolate requires reshaping
        # Shape: (T, 6) -> (1, 6, T) for interpolation -> (T', 6)
        angles_for_interp = angles_tensor.T.unsqueeze(0)  # (1, 6, T)
        resampled = torch.nn.functional.interpolate(
            angles_for_interp, 
            size=num_target_frames, 
            mode='linear', 
            align_corners=True
        )
        resampled = resampled.squeeze(0).T  # (T', 6)
        
        return resampled
    
    def extract_joint_angles(self, video_path):
        """Extract pose angle features from a video file.
        
        Computes the following 6 angle features per frame:
        - Left elbow angle (wrist-elbow-shoulder)
        - Right elbow angle (wrist-elbow-shoulder)
        - Left shoulder angle (elbow-shoulder-hip)
        - Right shoulder angle (elbow-shoulder-hip)
        - Left body angle (shoulder-hip-knee)
        - Right body angle (shoulder-hip-knee)
        
        The output sequence is resampled to the target frame rate (default 30 FPS)
        to ensure consistent temporal resolution regardless of the source video's
        frame rate.
        
        Args:
            video_path: Path to the video file.
            
        Returns:
            Tensor of shape (T, 6) containing angle features normalized to 0-1,
            where T is the number of frames at the target frame rate.
            If no pose is detected in a frame, angles are set to 0.
        """
        cap = cv2.VideoCapture(video_path)
        angles_list = []
        
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if source_fps <= 0:
            source_fps = 30.0
        
        frame_duration_ms = 1000.0 / source_fps
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
        
        # Resample to target frame rate for consistent temporal resolution
        angles_tensor = self._resample_to_target_fps(angles_tensor, source_fps)
        
        # Apply smoothing to reduce noise from pose estimation
        angles_tensor = self._smooth_angles(angles_tensor)

        return angles_tensor

    def _smooth_angles(self, angles_tensor):
        """Apply two-stage smoothing to angle sequences.
        
        Stage 1: Median filter removes outliers/spikes from pose detection
                 failures or incorrect landmark positions.
        Stage 2: Savitzky-Golay filter smooths high-frequency jitter while
                 preserving signal peaks and timing (zero phase lag).
        
        Args:
            angles_tensor: Tensor of shape (T, 6) containing angle features.
            
        Returns:
            Smoothed tensor of shape (T, 6).
        """
        # Need at least window_length frames for Savitzky-Golay
        if angles_tensor.shape[0] < 5:
            return angles_tensor
        
        angles_np = angles_tensor.numpy()
        
        # Stage 1: Median filter to remove outliers (kernel_size=3)
        # Applied along the time axis (axis=0) for each angle channel
        angles_np = median_filter(angles_np, size=(3, 1))
        
        # Stage 2: Savitzky-Golay filter to smooth jitter (window=5, order=2)
        # Applied along the time axis (axis=0) for each angle channel
        angles_np = savgol_filter(angles_np, window_length=5, polyorder=2, axis=0)
        
        return torch.tensor(angles_np, dtype=torch.float32)

    @staticmethod
    def _compute_angle(a, b, c):
        """Compute the angle at vertex B given three 3D points A, B, C.
        
        Args:
            a: numpy array of shape (3,) for point A
            b: numpy array of shape (3,) for point B (vertex)
            c: numpy array of shape (3,) for point C
            
        Returns:
            Angle at vertex B, normalized to 0-1 range (0 = 0°, 1 = 180°).
        """
        ba = a - b
        bc = c - b
        
        cos_angle = (ba @ bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        # Normalize to 0-1 range (angles are 0-180 degrees)
        return np.degrees(angle) / 180.0

    def _get_landmark_xyz(self, pose_landmarks, idx):
        """Extract (x, y, z) coordinates for a landmark by index."""
        lm = pose_landmarks[idx]
        return np.array([lm.x, lm.y, lm.z])

    def _calculate_angles(self, result: PoseLandmarkerResult):
        """Calculate the 6 joint angles from a pose detection result.
        
        Args:
            result: MediaPipe pose landmarker result for a single frame.
            
        Returns:
            List of 6 float values representing the joint angles normalized to 0-1.
            Returns [0.0] * 6 if no pose is detected.
        """
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
