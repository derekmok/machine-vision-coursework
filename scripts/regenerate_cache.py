"""Script to regenerate the feature cache with updated pose feature extraction.

This script processes all video files in the video-data directory and regenerates
the cached joint angle features. Use this after modifying the PoseFeatureExtractor
to ensure all cached features are consistent with the new extraction logic.
"""

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from neural_net.data_loader import VideoDataset


# Paths relative to the script's location
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

DEFAULT_VIDEO_DIR = str(_PROJECT_ROOT / "video-data")
DEFAULT_CACHE_DIR = str(_PROJECT_ROOT / ".feature_cache")
DEFAULT_MODEL_PATH = str(_PROJECT_ROOT / ".models" / "pose_landmarker.task")


def regenerate_cache(
    video_dir: str = DEFAULT_VIDEO_DIR,
    cache_dir: str = DEFAULT_CACHE_DIR,
    model_path: str = DEFAULT_MODEL_PATH,
    force: bool = False,
):
    """Regenerate the landmark cache for all videos using VideoDataset.
    
    Args:
        video_dir: Directory containing video files.
        cache_dir: Directory to store cached features.
        model_path: Path to the MediaPipe pose landmarker model.
        force: If True, regenerate all cache files. If False, skip existing ones.
    """
    video_dir_path = Path(video_dir)
    if not video_dir_path.exists():
        print(f"Error: Video directory '{video_dir}' does not exist.")
        sys.exit(1)
    
    print(f"Cache directory: '{cache_dir}'")
    print(f"Force regenerate: {force}")
    print()
    
    # Use VideoDataset to handle the heavy lifting
    dataset = VideoDataset(
        video_dir=video_dir,
        feature_processor=lambda features : (),
        media_pipe_model_path=model_path,
        cache_dir=cache_dir
    )
    
    print(f"Found {len(dataset)} video files in '{video_dir}'")
    
    # Process each video
    processed = 0
    skipped = 0
    failed = 0
    
    for i in tqdm(range(len(dataset)), desc="Processing videos"):
        video_filename = dataset.video_files[i]
        cache_path = Path(dataset._get_cache_path(video_filename))
        
        # If force is True, remove existing cache to trigger re-extraction in __getitem__
        if force and cache_path.exists():
            cache_path.unlink()
            
        if cache_path.exists():
            skipped += 1
            continue
            
        try:
            # Accessing __getitem__ triggers extraction if cache is missing
            _ = dataset[i]
            processed += 1
            
        except Exception as e:
            print(f"\nError processing '{video_filename}': {e}")
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
            force=args.force,
        )
    elif args.command == 'clear':
        clear_cache(cache_dir=args.cache_dir)
    else:
        # Default behavior: regenerate with --force
        regenerate_cache(force=True)


if __name__ == '__main__':
    main()
