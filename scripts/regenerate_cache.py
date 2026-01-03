"""Script to regenerate the landmark cache with updated pose feature extraction.

This script processes all video files in the video-data directory and regenerates
the cached joint angle features. Use this after modifying the PoseFeatureExtractor
to ensure all cached features are consistent with the new extraction logic.
"""

import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import torch

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from feature_engineering.pose_feature_extractor import PoseFeatureExtractor


# Paths relative to the script's location
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

DEFAULT_VIDEO_DIR = str(_PROJECT_ROOT / "video-data")
DEFAULT_CACHE_DIR = str(_PROJECT_ROOT / ".landmark_cache")
DEFAULT_MODEL_PATH = str(_PROJECT_ROOT / ".models" / "pose_landmarker.task")
DEFAULT_TARGET_FPS = 30.0


def regenerate_cache(
    video_dir: str = DEFAULT_VIDEO_DIR,
    cache_dir: str = DEFAULT_CACHE_DIR,
    model_path: str = DEFAULT_MODEL_PATH,
    target_fps: float = DEFAULT_TARGET_FPS,
    force: bool = False,
):
    """Regenerate the landmark cache for all videos.
    
    Args:
        video_dir: Directory containing video files.
        cache_dir: Directory to store cached features.
        model_path: Path to the MediaPipe pose landmarker model.
        target_fps: Target frame rate for output sequences.
        force: If True, regenerate all cache files. If False, skip existing ones.
    """
    video_dir = Path(video_dir)
    cache_dir = Path(cache_dir)
    
    if not video_dir.exists():
        print(f"Error: Video directory '{video_dir}' does not exist.")
        sys.exit(1)
    
    if not Path(model_path).exists():
        print(f"Error: Model file '{model_path}' does not exist.")
        sys.exit(1)
    
    # Create cache directory if it doesn't exist
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all video files
    video_extensions = ('.mp4', '.avi', '.mov')
    video_files = [f for f in video_dir.iterdir() if f.suffix.lower() in video_extensions]
    
    if not video_files:
        print(f"No video files found in '{video_dir}'.")
        return
    
    print(f"Found {len(video_files)} video files in '{video_dir}'")
    print(f"Cache directory: '{cache_dir}'")
    print(f"Target FPS: {target_fps}")
    print(f"Force regenerate: {force}")
    print()
    
    # Initialize the feature extractor
    extractor = PoseFeatureExtractor(model_path, target_fps=target_fps)
    
    # Process each video
    processed = 0
    skipped = 0
    failed = 0
    
    for video_path in tqdm(video_files, desc="Processing videos"):
        cache_filename = video_path.stem + '.pt'
        cache_path = cache_dir / cache_filename
        
        # Skip if cache exists and force is False
        if cache_path.exists() and not force:
            skipped += 1
            continue
        
        try:
            # Extract joint angles and density map
            angles_tensor, density_map = extractor.extract_features(str(video_path))
            
            # Save to cache as a dict containing both tensors
            cache_data = {
                'angles': angles_tensor,
                'density_map': density_map,
            }
            torch.save(cache_data, cache_path)
            processed += 1
            
        except Exception as e:
            print(f"\nError processing '{video_path.name}': {e}")
            failed += 1
    
    print()
    print(f"Processing complete:")
    print(f"  Processed: {processed}")
    print(f"  Skipped (already cached): {skipped}")
    print(f"  Failed: {failed}")


def clear_cache(cache_dir: str = DEFAULT_CACHE_DIR):
    """Clear all cached files.
    
    Args:
        cache_dir: Directory containing cached feature files.
    """
    cache_dir = Path(cache_dir)
    
    if not cache_dir.exists():
        print(f"Cache directory '{cache_dir}' does not exist.")
        return
    
    cache_files = list(cache_dir.glob('*.pt'))
    
    if not cache_files:
        print(f"No cache files found in '{cache_dir}'.")
        return
    
    print(f"Deleting {len(cache_files)} cache files...")
    for cache_file in cache_files:
        cache_file.unlink()
    
    print("Cache cleared.")


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate the landmark cache with updated pose feature extraction."
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Regenerate command
    regen_parser = subparsers.add_parser('regenerate', help='Regenerate cache files')
    regen_parser.add_argument(
        '--video-dir', '-v',
        default=DEFAULT_VIDEO_DIR,
        help=f"Directory containing video files (default: {DEFAULT_VIDEO_DIR})"
    )
    regen_parser.add_argument(
        '--cache-dir', '-c',
        default=DEFAULT_CACHE_DIR,
        help=f"Directory to store cached features (default: {DEFAULT_CACHE_DIR})"
    )
    regen_parser.add_argument(
        '--model-path', '-m',
        default=DEFAULT_MODEL_PATH,
        help=f"Path to the MediaPipe pose landmarker model (default: {DEFAULT_MODEL_PATH})"
    )
    regen_parser.add_argument(
        '--target-fps', '-f',
        type=float,
        default=DEFAULT_TARGET_FPS,
        help=f"Target frame rate for output sequences (default: {DEFAULT_TARGET_FPS})"
    )
    regen_parser.add_argument(
        '--force',
        action='store_true',
        help="Force regeneration of all cache files, even if they already exist"
    )
    
    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear all cache files')
    clear_parser.add_argument(
        '--cache-dir', '-c',
        default=DEFAULT_CACHE_DIR,
        help=f"Directory containing cached feature files (default: {DEFAULT_CACHE_DIR})"
    )
    
    args = parser.parse_args()
    
    if args.command == 'regenerate':
        regenerate_cache(
            video_dir=args.video_dir,
            cache_dir=args.cache_dir,
            model_path=args.model_path,
            target_fps=args.target_fps,
            force=args.force,
        )
    elif args.command == 'clear':
        clear_cache(cache_dir=args.cache_dir)
    else:
        # Default behavior: regenerate with --force
        regenerate_cache(force=True)


if __name__ == '__main__':
    main()
