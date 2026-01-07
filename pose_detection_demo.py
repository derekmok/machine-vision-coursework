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

# %% [markdown]
# # MediaPipe Pose Detection Demo
#
# This notebook demonstrates MediaPipe pose detection on video frames:
# 1. Select 4 videos from the `video-data/` directory
# 2. Extract a single frame from each video
# 3. Run MediaPipe pose detection on each frame
# 4. Overlay the detected landmarks on the frames

# %%
import os
import urllib.request

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

# %% [markdown]
# ## Setup: Download MediaPipe Model (if needed)

# %%
MODEL_PATH = ".models/pose_landmarker_heavy.task"

def ensure_model_exists():
    """Download the MediaPipe pose landmarker model if not present."""
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
        print(f"Downloading model to {MODEL_PATH}...")
        urllib.request.urlretrieve(url, MODEL_PATH)
        print("Download complete.")
    else:
        print(f"Model already exists at {MODEL_PATH}")

ensure_model_exists()

# %% [markdown]
# ## Select 4 Videos from the Dataset

# %%
VIDEO_DIR = "video-data"

video_files = sorted([f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')])

# Pick videos with different labels (1, 2, 3, 4 push-ups) for variety
selected_videos = []
seen_labels = set()
for vf in video_files:
    label = vf.split('_')[0]
    if label not in seen_labels and len(selected_videos) < 4:
        selected_videos.append(vf)
        seen_labels.add(label)

print("Selected videos:")
for v in selected_videos:
    print(f"  {v}")

# %% [markdown]
# ## Extract One Frame from Each Video

# %%
def extract_frame(video_path, position=0.25):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    target_frame = int(total_frames * position)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None

frames = {}
for video_name in selected_videos:
    video_path = os.path.join(VIDEO_DIR, video_name)
    frame = extract_frame(video_path)
    if frame is not None:
        frames[video_name] = frame
        print(f"Extracted frame from {video_name}: shape {frame.shape}")

# %% [markdown]
# ## Run MediaPipe Pose Detection

# %%
base_options = mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_poses=1
)
landmarker = vision.PoseLandmarker.create_from_options(options)

pose_results = {}
for video_name, frame in frames.items():
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    result = landmarker.detect(mp_image)
    pose_results[video_name] = result
    
    num_poses = len(result.pose_landmarks) if result.pose_landmarks else 0
    print(f"{video_name}: detected {num_poses} pose(s)")

landmarker.close()

# %% [markdown]
# ## Overlay Landmarks on Frames

# %%
# MediaPipe pose landmark connections for skeleton drawing
# Based on the PoseLandmarker 33-point model
POSE_CONNECTIONS = [
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7), # Left eye
    (0, 4), (4, 5), (5, 6), (6, 8), # Right eye
    (9, 10), # Mouth
    # Torso
    (11, 12), # Shoulders
    (11, 23), (12, 24), # Shoulder to hip
    (23, 24), # Hips
    # Left arm
    (11, 13), (13, 15), # Shoulder to wrist
    (15, 17), (15, 19), (15, 21), (17, 19), # Wrist to fingers
    # Right arm
    (12, 14), (14, 16), # Shoulder to wrist
    (16, 18), (16, 20), (16, 22), (18, 20), # Wrist to fingers
    # Left leg
    (23, 25), (25, 27), # Hip to ankle
    (27, 29), (27, 31), (29, 31), # Ankle to foot
    # Right leg
    (24, 26), (26, 28), # Hip to ankle
    (28, 30), (28, 32), (30, 32), # Ankle to foot
]

def draw_landmarks(frame, pose_landmarks):
    annotated = frame.copy()
    h, w = annotated.shape[:2]
    
    if not pose_landmarks or len(pose_landmarks) == 0:
        return annotated
    
    landmarks = pose_landmarks[0]
    
    # Draw connections
    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            
            start_point = (int(start.x * w), int(start.y * h))
            end_point = (int(end.x * w), int(end.y * h))
            
            cv2.line(annotated, start_point, end_point, (0, 255, 0), 2)
    
    # Draw landmark points
    for landmark in landmarks:
        point = (int(landmark.x * w), int(landmark.y * h))
        cv2.circle(annotated, point, 4, (255, 0, 0), -1)
    
    return annotated

# %% [markdown]
# ## Visualize Results: 2x2 Grid of Annotated Frames

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for idx, (video_name, frame) in enumerate(frames.items()):
    if idx >= 4:
        break
    
    result = pose_results.get(video_name)
    annotated_frame = draw_landmarks(frame, result.pose_landmarks if result else None)
    
    ax = axes[idx]
    ax.imshow(annotated_frame)
    
    label = video_name.split('_')[0]
    ax.set_title(f"{video_name}\n({label} push-ups)", fontsize=10)
    ax.axis('off')

plt.suptitle("MediaPipe Pose Detection on Video Frames", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Landmark Legend
#
# The skeleton overlay shows:
# - **Red circles**: Individual landmark points (33 total)
# - **Green lines**: Connections between landmarks forming the skeleton
#
# Key landmarks used in push-up detection:
# - Shoulders (indices 11, 12)
# - Elbows (indices 13, 14)  
# - Wrists (indices 15, 16)
# - Hips (indices 23, 24)
# - Knees (indices 25, 26)
