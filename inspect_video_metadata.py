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

# %%
import os

import cv2
import pandas as pd


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
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
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

video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
video_files = [
    os.path.join(video_dir, f) 
    for f in os.listdir(video_dir) 
    if f.lower().endswith(video_extensions)
]

print(f"Found {len(video_files)} video files in '{video_dir}'")

# %%
metadata_list = []

for video_path in sorted(video_files):
    metadata = get_video_metadata(video_path)
    metadata_list.append(metadata)
    
df = pd.DataFrame(metadata_list)

# %% [markdown]
# ## Video Metadata Table

# %%
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

df_display = df[['filename', 'framerate', 'frame_count', 'duration_seconds']].copy()
df_display['duration_seconds'] = df_display['duration_seconds'].round(2)
df_display

# %% [markdown]
# ## Summary Statistics

# %%
df_valid = df[df['error'].isna()].copy()

stats_columns = ['framerate', 'frame_count', 'duration_seconds']

summary_stats_df = df_valid[stats_columns].describe().round(2)
summary_stats_df

# %%
unique_fps = sorted(df_valid['framerate'].unique())
most_common_fps = df_valid['framerate'].mode().values[0]
min_frame_idx = df_valid['frame_count'].idxmin()
max_frame_idx = df_valid['frame_count'].idxmax()
total_duration_sec = df_valid['duration_seconds'].sum()
total_duration_min = total_duration_sec / 60
min_duration_idx = df_valid['duration_seconds'].idxmin()
max_duration_idx = df_valid['duration_seconds'].idxmax()

insights_data = {
    'Category': [
        'Overview', 'Overview',
        'Framerate', 'Framerate',
        'Frame Count', 'Frame Count', 'Frame Count',
        'Duration', 'Duration', 'Duration'
    ],
    'Metric': [
        'Total videos analyzed',
        'Videos with errors',
        'Unique framerates',
        'Most common FPS',
        'Total frames (all videos)',
        'Shortest video',
        'Longest video',
        'Total duration',
        'Shortest video',
        'Longest video'
    ],
    'Value': [
        str(len(df_valid)),
        str(len(df) - len(df_valid)),
        ', '.join([f"{fps:.2f}" for fps in unique_fps]),
        f"{most_common_fps:.2f}",
        f"{df_valid['frame_count'].sum():,}",
        f"{df_valid.loc[min_frame_idx, 'filename']} ({df_valid['frame_count'].min()} frames)",
        f"{df_valid.loc[max_frame_idx, 'filename']} ({df_valid['frame_count'].max()} frames)",
        f"{total_duration_sec:.2f} seconds ({total_duration_min:.2f} minutes)",
        f"{df_valid.loc[min_duration_idx, 'filename']} ({df_valid['duration_seconds'].min():.2f}s)",
        f"{df_valid.loc[max_duration_idx, 'filename']} ({df_valid['duration_seconds'].max():.2f}s)"
    ]
}

insights_df = pd.DataFrame(insights_data)
insights_df

# %% [markdown]
# ## Grouped Statistics by Label

# %%
df_valid['label'] = df_valid['filename'].apply(lambda x: x.split('_')[0])

grouped_df = df_valid.groupby('label').agg({
    'filename': 'count',
    'framerate': ['mean', 'std'],
    'frame_count': ['mean', 'std', 'min', 'max'],
    'duration_seconds': ['mean', 'std', 'min', 'max']
}).round(2)

grouped_df.columns = ['_'.join(col).strip() for col in grouped_df.columns.values]
grouped_df = grouped_df.rename(columns={'filename_count': 'video_count'})
grouped_df
