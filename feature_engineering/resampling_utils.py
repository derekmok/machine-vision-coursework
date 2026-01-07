from feature_engineering.constants import TARGET_FPS


def resampled_index_to_original_frame(
    resampled_idx: int,
    source_fps: float,
) -> int:
    """Convert a resampled sequence index back to the original video frame index.
    
    During feature extraction, video frames are resampled to TARGET_FPS. This function
    maps an index in the resampled sequence back to the corresponding frame index
    in the original video.
    
    Args:
        resampled_idx: Index in the resampled sequence (at TARGET_FPS).
        source_fps: Original video's frame rate.

    Returns:
        The corresponding frame index in the original video.
    """
    fps_ratio = source_fps / TARGET_FPS
    original_frame_idx = int(round(resampled_idx * fps_ratio))

    return original_frame_idx
