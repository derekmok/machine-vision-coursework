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
# # Video Metadata Inspector
#
# This notebook inspects every video in the `video-data` folder and extracts:
# - **Framerate** (FPS)
# - **Number of frames**
# - **Length** (duration in seconds)
#
# It also computes summary statistics for each of these fields.

# %% [markdown]
# ## Setup and Imports

# %%
import os
import cv2
import pandas as pd
import numpy as np

# %% [markdown]
# ## Video Inspection Function

# %%
def get_video_metadata(video_path: str) -> dict:
    """
    Extract metadata from a video file.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Dictionary containing filename, framerate, frame_count, and duration
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return {
            'filename': os.path.basename(video_path),
            'framerate': None,
            'frame_count': None,
            'duration_seconds': None,
            'error': 'Could not open video'
        }
    
    # Extract metadata
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate duration
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    
    return {
        'filename': os.path.basename(video_path),
        'framerate': fps,
        'frame_count': frame_count,
        'duration_seconds': duration,
        'error': None
    }

# %% [markdown]
# ## Inspect All Videos

# %%
video_dir = './video-data'

# Get all video files
video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
video_files = [
    os.path.join(video_dir, f) 
    for f in os.listdir(video_dir) 
    if f.lower().endswith(video_extensions)
]

print(f"Found {len(video_files)} video files in '{video_dir}'")

# %%
# Extract metadata from all videos
metadata_list = []

for video_path in sorted(video_files):
    metadata = get_video_metadata(video_path)
    metadata_list.append(metadata)
    
# Create DataFrame
df = pd.DataFrame(metadata_list)

# %% [markdown]
# ## Video Metadata Table

# %%
# Display all video metadata
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("=" * 80)
print("VIDEO METADATA")
print("=" * 80)
df_display = df[['filename', 'framerate', 'frame_count', 'duration_seconds']].copy()
df_display['duration_seconds'] = df_display['duration_seconds'].round(2)
print(df_display.to_string(index=False))

# %% [markdown]
# ## Summary Statistics

# %%
# Filter out any videos with errors
df_valid = df[df['error'].isna()].copy()

print("=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

# Statistics for each field
stats_columns = ['framerate', 'frame_count', 'duration_seconds']
stats_rows = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']

summary_stats = df_valid[stats_columns].describe()
print("\n")
print(summary_stats.round(2).to_string())

# %%
# Additional summary information
print("\n" + "=" * 80)
print("ADDITIONAL INSIGHTS")
print("=" * 80)

print(f"\nTotal videos analyzed: {len(df_valid)}")
print(f"Videos with errors: {len(df) - len(df_valid)}")

print(f"\n--- Framerate ---")
print(f"  Unique framerates: {sorted(df_valid['framerate'].unique())}")
print(f"  Most common: {df_valid['framerate'].mode().values[0]:.2f} FPS")

print(f"\n--- Frame Count ---")
print(f"  Total frames (all videos): {df_valid['frame_count'].sum():,}")
print(f"  Shortest video: {df_valid.loc[df_valid['frame_count'].idxmin(), 'filename']} ({df_valid['frame_count'].min()} frames)")
print(f"  Longest video: {df_valid.loc[df_valid['frame_count'].idxmax(), 'filename']} ({df_valid['frame_count'].max()} frames)")

print(f"\n--- Duration ---")
print(f"  Total duration (all videos): {df_valid['duration_seconds'].sum():.2f} seconds ({df_valid['duration_seconds'].sum() / 60:.2f} minutes)")
print(f"  Shortest video: {df_valid.loc[df_valid['duration_seconds'].idxmin(), 'filename']} ({df_valid['duration_seconds'].min():.2f}s)")
print(f"  Longest video: {df_valid.loc[df_valid['duration_seconds'].idxmax(), 'filename']} ({df_valid['duration_seconds'].max():.2f}s)")

# %% [markdown]
# ## Grouped Statistics by Label

# %%
# Extract label from filename (first character before underscore)
df_valid['label'] = df_valid['filename'].apply(lambda x: x.split('_')[0])

print("=" * 80)
print("STATISTICS GROUPED BY LABEL (Pushup Count)")
print("=" * 80)

grouped = df_valid.groupby('label').agg({
    'filename': 'count',
    'framerate': ['mean', 'std'],
    'frame_count': ['mean', 'std', 'min', 'max'],
    'duration_seconds': ['mean', 'std', 'min', 'max']
}).round(2)

grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
grouped = grouped.rename(columns={'filename_count': 'video_count'})
print("\n")
print(grouped.to_string())
