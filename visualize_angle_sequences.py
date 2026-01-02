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
# # Visualizing Extracted Angle Sequences
#
# This notebook demonstrates the angle sequences extracted from push-up videos using `PoseFeatureExtractor`.
#
# The extracted angles are:
# 1. **Left Elbow Angle** (wrist-elbow-shoulder)
# 2. **Right Elbow Angle** (wrist-elbow-shoulder)
# 3. **Left Shoulder Angle** (elbow-shoulder-hip)
# 4. **Right Shoulder Angle** (elbow-shoulder-hip)
# 5. **Left Body Angle** (shoulder-hip-knee)
# 6. **Right Body Angle** (shoulder-hip-knee)

# %%
import torch
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import numpy as np

# %%
CACHE_DIR = ".landmark_cache"

ANGLE_NAMES = [
    "Left Elbow",
    "Right Elbow",
    "Left Shoulder",
    "Right Shoulder",
    "Left Body (Hip)",
    "Right Body (Hip)"
]

# %% [markdown]
# ## Load Cached Angle Sequences

# %%
# Get all cached files and sort by push-up count label
cache_files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.pt")))

# Group by label
files_by_label = {}
for f in cache_files:
    filename = os.path.basename(f)
    label = int(filename.split('_')[0])
    if label not in files_by_label:
        files_by_label[label] = []
    files_by_label[label].append(f)

print("Available labels and counts:")
for label in sorted(files_by_label.keys()):
    print(f"  Label {label}: {len(files_by_label[label])} videos")

# %%
# Select a few sample videos (one from each label category)
sample_files = []
for label in sorted(files_by_label.keys())[:4]:  # Take first 4 labels
    sample_files.append(files_by_label[label][0])

print("Selected samples:")
for f in sample_files:
    print(f"  {os.path.basename(f)}")

# %% [markdown]
# ## Plot All 6 Angles for Each Sample Video

# %%
fig, axes = plt.subplots(len(sample_files), 1, figsize=(14, 4 * len(sample_files)))

for idx, file_path in enumerate(sample_files):
    cache_data = torch.load(file_path, weights_only=True)
    angles, density_map = cache_data['angles'], cache_data['density_map']
    filename = os.path.basename(file_path)
    label = filename.split('_')[0]
    
    ax = axes[idx] if len(sample_files) > 1 else axes
    frames = np.arange(len(angles))
    
    for i, angle_name in enumerate(ANGLE_NAMES):
        ax.plot(frames, angles[:, i].numpy(), label=angle_name, alpha=0.8)
    
    ax.set_xlabel('Frame')
    ax.set_ylabel('Angle (normalized)')
    ax.set_title(f'Video: {filename} (Label: {label} push-ups)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Focus on Elbow Angles (Most Indicative of Push-up Motion)

# %%
fig, axes = plt.subplots(len(sample_files), 1, figsize=(14, 3 * len(sample_files)))

for idx, file_path in enumerate(sample_files):
    cache_data = torch.load(file_path, weights_only=True)
    angles, density_map = cache_data['angles'], cache_data['density_map']
    filename = os.path.basename(file_path)
    label = filename.split('_')[0]
    
    ax = axes[idx] if len(sample_files) > 1 else axes
    frames = np.arange(len(angles))
    
    # Plot left and right elbow angles
    ax.plot(frames, angles[:, 0].numpy(), label='Left Elbow', color='blue', linewidth=2)
    ax.plot(frames, angles[:, 1].numpy(), label='Right Elbow', color='red', linewidth=2)
    
    # Add horizontal reference lines
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='0.5 (90°)')
    
    ax.set_xlabel('Frame')
    ax.set_ylabel('Elbow Angle (normalized)')
    ax.set_title(f'Elbow Angles - {filename} ({label} push-ups)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Density Map Visualization
#
# The density map shows gaussian bumps centered at detected push-up positions (elbow angle valleys).

# %%
fig, axes = plt.subplots(len(sample_files), 1, figsize=(14, 4 * len(sample_files)))

for idx, file_path in enumerate(sample_files):
    cache_data = torch.load(file_path, weights_only=True)
    angles, density_map = cache_data['angles'], cache_data['density_map']
    filename = os.path.basename(file_path)
    label = filename.split('_')[0]
    
    ax = axes[idx] if len(sample_files) > 1 else axes
    frames = np.arange(len(angles))
    
    # Create twin axis for density map
    ax2 = ax.twinx()
    
    # Plot elbow angles (average of left and right)
    avg_elbow = (angles[:, 0] + angles[:, 1]) / 2
    ax.plot(frames, avg_elbow.numpy(), label='Avg Elbow Angle', color='steelblue', linewidth=1.5, alpha=0.7)
    
    # Plot density map as filled area
    ax2.fill_between(frames, density_map.numpy(), alpha=0.4, color='orange', label='Density Map')
    ax2.plot(frames, density_map.numpy(), color='darkorange', linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel('Frame')
    ax.set_ylabel('Elbow Angle (normalized)', color='steelblue')
    ax2.set_ylabel('Density', color='darkorange')
    ax.set_title(f'Elbow Angles & Density Map - {filename} ({label} push-ups)')
    ax.tick_params(axis='y', labelcolor='steelblue')
    ax2.tick_params(axis='y', labelcolor='darkorange')
    ax.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.1)
    
    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Compare Angle Patterns Across Different Push-up Counts

# %%
# Plot the angle with maximum variance (typically best for detecting push-ups)
fig, ax = plt.subplots(figsize=(14, 6))

colors = plt.cm.viridis(np.linspace(0, 1, len(sample_files)))

for idx, file_path in enumerate(sample_files):
    cache_data = torch.load(file_path, weights_only=True)
    angles, density_map = cache_data['angles'], cache_data['density_map']
    filename = os.path.basename(file_path)
    label = filename.split('_')[0]
    
    # Use the elbow with greater variance for this video
    left_var = angles[:, 0].var().item()
    right_var = angles[:, 1].var().item()
    elbow_idx = 0 if left_var > right_var else 1
    elbow_name = "Left" if elbow_idx == 0 else "Right"
    
    frames = np.arange(len(angles))
    ax.plot(frames, angles[:, elbow_idx].numpy(), 
            label=f'{label} push-ups ({elbow_name} elbow)', 
            color=colors[idx], linewidth=1.5, alpha=0.8)

ax.set_xlabel('Frame', fontsize=12)
ax.set_ylabel('Elbow Angle (normalized)', fontsize=12)
ax.set_title('Elbow Angle Patterns Across Different Push-up Counts', fontsize=14)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Subplots: Separate Angle Types

# %%
# Pick one sample video
sample_path = sample_files[1]  # 2 push-ups
cache_data = torch.load(sample_path, weights_only=True)
angles, density_map = cache_data['angles'], cache_data['density_map']
filename = os.path.basename(sample_path)

fig, axes = plt.subplots(3, 2, figsize=(14, 10))
frames = np.arange(len(angles))

for i, (ax, name) in enumerate(zip(axes.flat, ANGLE_NAMES)):
    ax.plot(frames, angles[:, i].numpy(), color='steelblue', linewidth=1.5)
    ax.fill_between(frames, angles[:, i].numpy(), alpha=0.3, color='steelblue')
    ax.set_title(name, fontsize=12)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Angle (normalized)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)

fig.suptitle(f'All Joint Angles - {filename}', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Statistics Summary

# %%
for file_path in sample_files:
    cache_data = torch.load(file_path, weights_only=True)
    angles, density_map = cache_data['angles'], cache_data['density_map']
    filename = os.path.basename(file_path)
    label = filename.split('_')[0]
    
    # Build statistics DataFrame
    stats_data = []
    for i, name in enumerate(ANGLE_NAMES):
        angle_data = angles[:, i]
        stats_data.append({
            'Angle Type': name,
            'Mean': angle_data.mean().item(),
            'Std': angle_data.std().item(),
            'Min': angle_data.min().item(),
            'Max': angle_data.max().item()
        })
    
    # Add density map statistics
    stats_data.append({
        'Angle Type': 'Density Map',
        'Mean': density_map.mean().item(),
        'Std': density_map.std().item(),
        'Min': density_map.min().item(),
        'Max': density_map.max().item()
    })
    
    df = pd.DataFrame(stats_data)
    df = df.set_index('Angle Type')
    
    print(f"\n{filename} ({label} push-ups) - {len(angles)} frames")
    display(df.round(3))

# %% [markdown]
# ## Verify Density Map Sums
#
# Check all samples to ensure the sum of the density map matches the labeled push-up count.

# %%
mismatches = []
tolerance = 0.1

for file_path in cache_files:
    cache_data = torch.load(file_path, weights_only=True)
    density_map = cache_data['density_map']
    filename = os.path.basename(file_path)
    label = int(filename.split('_')[0])
    
    density_sum = density_map.sum().item()
    diff = abs(density_sum - label)
    
    if diff > tolerance:
        mismatches.append({
            'Filename': filename,
            'Label': label,
            'Density Sum': density_sum,
            'Diff': diff,
            'Path': file_path
        })

df_mismatches = pd.DataFrame(mismatches)
if not df_mismatches.empty:
    print(f"Found {len(mismatches)} samples with density map sum mismatches (tolerance={tolerance}):")
    display(df_mismatches)
else:
    print(f"All {len(cache_files)} samples have density map sums matching their labels (within tolerance {tolerance}).")

# %% [markdown]
# ## Visualize Mismatch Samples (if any)

# %%
if mismatches:
    # Take up to 5 examples to visualize
    samples_to_plot = mismatches[:5]
    
    fig, axes = plt.subplots(len(samples_to_plot), 1, figsize=(14, 4 * len(samples_to_plot)))
    
    for idx, mismatch in enumerate(samples_to_plot):
        file_path = mismatch['Path']
        cache_data = torch.load(file_path, weights_only=True)
        angles, density_map = cache_data['angles'], cache_data['density_map']
        filename = mismatch['Filename']
        label = mismatch['Label']
        density_sum = mismatch['Density Sum']
        
        ax = axes[idx] if len(samples_to_plot) > 1 else axes
        frames = np.arange(len(angles))
        
        # Create twin axis for density map
        ax2 = ax.twinx()
        
        # Plot elbow angles (average of left and right)
        avg_elbow = (angles[:, 0] + angles[:, 1]) / 2
        ax.plot(frames, avg_elbow.numpy(), label='Avg Elbow Angle', color='steelblue', linewidth=1.5, alpha=0.7)
        
        # Plot density map as filled area
        ax2.fill_between(frames, density_map.numpy(), alpha=0.4, color='orange', label='Density Map')
        ax2.plot(frames, density_map.numpy(), color='darkorange', linewidth=1.5, alpha=0.8)
        
        ax.set_xlabel('Frame')
        ax.set_ylabel('Elbow Angle (normalized)', color='steelblue')
        ax2.set_ylabel('Density', color='darkorange')
        ax.set_title(f'MISMATCH - {filename} (Label: {label}, Sum: {density_sum:.2f})')
        ax.tick_params(axis='y', labelcolor='steelblue')
        ax2.tick_params(axis='y', labelcolor='darkorange')
        ax.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.1)
        
        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    plt.show()
else:
    print("No mismatches to visualize.")
