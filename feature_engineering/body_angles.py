import numpy as np
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerResult

from feature_engineering.constants import LEFT_ELBOW_ANGLE_IDX, RIGHT_ELBOW_ANGLE_IDX, LEFT_SHOULDER_ANGLE_IDX, \
    RIGHT_SHOULDER_ANGLE_IDX, LEFT_BODY_ANGLE_IDX, RIGHT_BODY_ANGLE_IDX

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

def _get_landmark_xyz(pose_landmarks: list[NormalizedLandmark], idx):
    landmark = pose_landmarks[idx]
    return np.array([landmark.x, landmark.y, landmark.z])


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


def calculate_angles(result: PoseLandmarkerResult):
    """Calculate the 6 joint angles from a pose detection result.

    Args:
        result: MediaPipe pose landmarker result for a single frame.

    Returns:
        List of 6 float values representing the joint angles normalized to 0-1.
        Returns [0.0] * 6 if no pose is detected.
    """
    frame_angles = [0.0] * 6
    if result.pose_landmarks and len(result.pose_landmarks) > 0:
        pose_landmarks = result.pose_landmarks[0]

        left_shoulder = _get_landmark_xyz(pose_landmarks, LEFT_SHOULDER)
        right_shoulder = _get_landmark_xyz(pose_landmarks, RIGHT_SHOULDER)
        left_elbow = _get_landmark_xyz(pose_landmarks, LEFT_ELBOW)
        right_elbow = _get_landmark_xyz(pose_landmarks, RIGHT_ELBOW)
        left_wrist = _get_landmark_xyz(pose_landmarks, LEFT_WRIST)
        right_wrist = _get_landmark_xyz(pose_landmarks, RIGHT_WRIST)
        left_hip = _get_landmark_xyz(pose_landmarks, LEFT_HIP)
        right_hip = _get_landmark_xyz(pose_landmarks, RIGHT_HIP)
        left_knee = _get_landmark_xyz(pose_landmarks, LEFT_KNEE)
        right_knee = _get_landmark_xyz(pose_landmarks, RIGHT_KNEE)

        left_elbow_angle = _compute_angle(left_wrist, left_elbow, left_shoulder)
        right_elbow_angle = _compute_angle(right_wrist, right_elbow, right_shoulder)
        left_shoulder_angle = _compute_angle(left_elbow, left_shoulder, left_hip)
        right_shoulder_angle = _compute_angle(right_elbow, right_shoulder, right_hip)
        left_body_angle = _compute_angle(left_shoulder, left_hip, left_knee)
        right_body_angle = _compute_angle(right_shoulder, right_hip, right_knee)

        frame_angles[LEFT_ELBOW_ANGLE_IDX] = left_elbow_angle
        frame_angles[RIGHT_ELBOW_ANGLE_IDX] = right_elbow_angle
        frame_angles[LEFT_SHOULDER_ANGLE_IDX] = left_shoulder_angle
        frame_angles[RIGHT_SHOULDER_ANGLE_IDX] = right_shoulder_angle
        frame_angles[LEFT_BODY_ANGLE_IDX] = left_body_angle
        frame_angles[RIGHT_BODY_ANGLE_IDX] = right_body_angle

    return frame_angles
