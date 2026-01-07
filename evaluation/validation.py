import os
from typing import Optional, TypedDict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.signal import find_peaks
from torch.utils.data import DataLoader

from feature_engineering.constants import TARGET_FPS
from feature_engineering.resampling_utils import resampled_index_to_original_frame
from neural_net.data_loader import VideoDataset
from neural_net.ensemble_model import EnsembleModel


def _find_density_peaks(density_map: np.ndarray) -> np.ndarray:
    """Find peaks in a density map.
    
    Args:
        density_map: 1D array of density values
        
    Returns:
        Array of peak indices
    """
    peak_height_threshold = np.max(density_map) * 0.1
    peaks, _ = find_peaks(
        density_map,
        height=peak_height_threshold,
        prominence=0.01,
        distance=5
    )
    return peaks


class ValidationResult(TypedDict):
    """Result dictionary for a single validation video."""
    video_filename: str
    true_count: int
    pred_count: float
    rounded_pred: int
    is_correct: bool
    density_map: np.ndarray
    angles: np.ndarray
    seq_len: int


def evaluate_on_validation_data(
    ensemble: EnsembleModel,
    validation_dir: str = "validation-data",
) -> Optional[list[ValidationResult]]:
    """Evaluate the trained ensemble on unseen validation data.
    
    Args:
        ensemble: EnsembleModel model
        validation_dir: Path to the validation data directory
        
    Returns:
        Dictionary with predictions, targets, and per-video results
    """
    if not os.path.exists(validation_dir):
        print(f"Validation data directory '{validation_dir}' does not exist.")
        print("Skipping validation evaluation.")
        return None
    
    val_dataset = VideoDataset.for_inference(validation_dir, cache_dir=".validation-cache")
    
    if len(val_dataset) == 0:
        print("No videos found in validation directory.")
        return None
    
    print(f"Found {len(val_dataset)} validation videos")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ensemble = ensemble.to(device)
    ensemble.eval()
    
    results = []
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    with torch.no_grad():
        for i, (sequences, labels) in enumerate(val_loader):
            video_filename = val_dataset.video_files[i]
            
            true_count = int(video_filename[0])
            
            sequences = sequences.to(device)
            seq_len = sequences.shape[1]
            
            pred_count, density_map = ensemble(sequences)
            pred_count = pred_count.cpu().squeeze().item()
            density_map = density_map.cpu().squeeze().numpy()
            rounded_pred = round(pred_count)
            
            is_correct = (rounded_pred == true_count)
            
            results.append({
                'video_filename': video_filename,
                'true_count': true_count,
                'pred_count': pred_count,
                'rounded_pred': rounded_pred,
                'is_correct': is_correct,
                'density_map': density_map,
                'angles': sequences.cpu().squeeze().numpy(),
                'seq_len': seq_len,
            })
    
    return results


def display_validation_results(results: Optional[list[ValidationResult]]) -> Optional[pd.DataFrame]:
    """Display validation results in a formatted table.
    
    Args:
        results: List of result dictionaries from evaluate_on_validation_data
    """
    if results is None:
        return
    
    df = pd.DataFrame([{
        'Video': r['video_filename'],
        'True Count': r['true_count'],
        'Predicted': f"{r['pred_count']:.2f}",
        'Rounded': r['rounded_pred'],
        'Result': '✓ Correct' if r['is_correct'] else '✗ Wrong',
    } for r in results])
    
    correct = sum(1 for r in results if r['is_correct'])
    total = len(results)
    accuracy = correct / total if total > 0 else 0
    
    print("=" * 60)
    print("VALIDATION RESULTS ON UNSEEN DATA")
    print("=" * 60)
    print()
    print(df.to_string(index=False))
    print()
    print("-" * 60)
    print(f"Accuracy: {correct}/{total} ({accuracy:.1%})")
    print("-" * 60)
    
    return df


def plot_validation_angle_sequences(results: Optional[list[ValidationResult]]) -> None:
    """Plot angle sequences for each validation video.
    
    Args:
        results: List of result dictionaries from evaluate_on_validation_data
    """
    if results is None or len(results) == 0:
        return
    
    n_videos = len(results)
    fig, axes = plt.subplots(n_videos, 1, figsize=(14, 4 * n_videos))
    if n_videos == 1:
        axes = [axes]
    
    angle_names = ['Left Elbow', 'Right Elbow', 'Left Shoulder', 'Right Shoulder', 'Left Hip', 'Right Hip']
    colors = plt.cm.tab10.colors
    
    for idx, (ax, result) in enumerate(zip(axes, results)):
        angles = result['angles']  # Shape: (seq_len, 6)
        seq_len = result['seq_len']
        frames = range(seq_len)
        
        for i, (angle_name, color) in enumerate(zip(angle_names, colors)):
            ax.plot(frames, angles[:seq_len, i], color=color, linewidth=1.5, alpha=0.8, label=angle_name)
        
        status = "✓" if result['is_correct'] else "✗"
        ax.set_title(f"{result['video_filename']} | True: {result['true_count']}, "
                     f"Pred: {result['pred_count']:.2f} → {result['rounded_pred']} {status}",
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Angle (normalized)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8, ncol=3)
    
    fig.suptitle('Validation Videos: Angle Sequences', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    plt.savefig('plots/validation_angle_sequences.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Angle sequences saved to 'plots/validation_angle_sequences.png'")


def plot_validation_density_maps(results: Optional[list[ValidationResult]]) -> None:
    """Plot density maps for each validation video.
    
    Args:
        results: List of result dictionaries from evaluate_on_validation_data
    """
    if results is None or len(results) == 0:
        return
    
    n_videos = len(results)
    fig, axes = plt.subplots(n_videos, 1, figsize=(14, 4 * n_videos))
    if n_videos == 1:
        axes = [axes]
    
    for idx, (ax, result) in enumerate(zip(axes, results)):
        density_map = result['density_map']
        frames = range(len(density_map))
        
        ax.plot(frames, density_map, color='steelblue', linewidth=1.5)
        ax.fill_between(frames, density_map, alpha=0.3, color='steelblue')
        
        # Detect and mark all peaks
        peaks = _find_density_peaks(density_map)
        
        # Mark each detected peak
        for i, peak_idx in enumerate(peaks):
            peak_val = density_map[peak_idx]
            ax.axvline(x=peak_idx, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
            ax.scatter([peak_idx], [peak_val], color='red', s=80, zorder=5)
            ax.annotate(f'{i+1}', (peak_idx, peak_val), textcoords="offset points", 
                        xytext=(0, 8), ha='center', fontsize=8, fontweight='bold', color='red')
        
        status = "✓" if result['is_correct'] else "✗"
        ax.set_title(f"{result['video_filename']} | True: {result['true_count']}, "
                     f"Pred: {result['pred_count']:.2f} → {result['rounded_pred']} {status} | "
                     f"Peaks: {len(peaks)} | Density Sum: {density_map.sum():.2f}",
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Predicted Density')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
    
    fig.suptitle('Validation Videos: Predicted Density Maps', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    os.makedirs('plots', exist_ok=True)
    
    plt.savefig('plots/validation_density_maps.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Density maps saved to 'plots/validation_density_maps.png'")


def plot_peak_frames(results: Optional[list[ValidationResult]], validation_dir: str = "validation-data") -> None:
    """Extract and plot video frames at all detected density map peaks.
    
    Uses scipy.signal.find_peaks to detect all local maxima in the density map
    and displays the corresponding video frames.
    
    Args:
        results: List of result dictionaries from evaluate_on_validation_data
        validation_dir: Path to the validation data directory
    """
    if results is None or len(results) == 0:
        return
    
    for result in results:
        video_path = os.path.join(validation_dir, result['video_filename'])
        density_map = result['density_map']
        
        # Detect all peaks in the density map
        peaks = _find_density_peaks(density_map)
        
        if len(peaks) == 0:
            print(f"No peaks detected in {result['video_filename']}")
            continue
        
        n_peaks = len(peaks)
        
        n_cols = min(n_peaks, 5)
        n_rows = (n_peaks + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        if n_peaks == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        cap = cv2.VideoCapture(video_path)
        
        # Get source video FPS to convert resampled indices back to original frame indices
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if source_fps <= 0:
            source_fps = TARGET_FPS  # Fallback to target FPS if unable to read
        
        for i, peak_idx in enumerate(peaks):
            row, col = divmod(i, n_cols)
            ax = axes[row, col]
            
            original_frame_idx = resampled_index_to_original_frame(peak_idx, source_fps)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, original_frame_idx)
            ret, frame = cap.read()
            
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ax.imshow(frame_rgb)
            else:
                ax.text(0.5, 0.5, f'Could not read frame {peak_idx}', 
                        ha='center', va='center', fontsize=12)
            
            peak_height = density_map[peak_idx]
            ax.set_title(f'Peak {i+1}: Frame {original_frame_idx}\nDensity: {peak_height:.3f}',
                         fontsize=9, fontweight='bold')
            ax.axis('off')
        
        cap.release()
        
        for i in range(n_peaks, n_rows * n_cols):
            row, col = divmod(i, n_cols)
            axes[row, col].axis('off')
        
        status = "✓" if result['is_correct'] else "✗"
        fig.suptitle(f"{result['video_filename']} | {n_peaks} peaks detected\n"
                     f"True: {result['true_count']} | Pred: {result['rounded_pred']} {status}",
                     fontsize=12, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        
        os.makedirs('plots', exist_ok=True)
        
        save_name = f"plots/validation_peaks_{result['video_filename'].replace('.mp4', '')}.png"
        plt.savefig(save_name, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Peak frames saved to '{save_name}'")
