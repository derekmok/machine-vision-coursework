import torch

class Compose:
    """Compose multiple transforms together.
    
    Applies a sequence of transforms in order.
    
    Args:
        transforms: List of transform objects to apply.
    """
    
    def __init__(self, transforms: list):
        self.transforms = transforms
    
    def __call__(self, landmarks: torch.Tensor) -> torch.Tensor:
        """Apply all transforms in sequence.
        
        Args:
            landmarks: Tensor of shape (T, 24) containing landmark coordinates.
            
        Returns:
            Transformed landmarks tensor.
        """
        for transform in self.transforms:
            landmarks = transform(landmarks)
        return landmarks

class RandomSequenceRepeat:
    """Repeat landmark sequences to generate training examples with higher counts.
    
    This label-aware transform concatenates a sequence with itself one or more
    times to create a training example representing more repetitions of the
    exercise. For example, a sequence with 2 push-ups repeated twice becomes
    a sequence representing 4 push-ups.
    
    Note: This transform takes (landmarks, label) as input and returns
    (new_landmarks, new_label), unlike regular transforms that only modify
    landmarks.
    
    Args:
        p: Probability of applying the repeat. Default is 0.5.
        max_count: Maximum allowed count after repetition. Default is 10.
        max_repeats: Maximum number of times to repeat (not including original).
                     Default is 3 (so sequence can be 1x to 4x original).
    """
    
    def __init__(self, p: float = 0.5, max_count: int = 10, max_repeats: int = 3):
        if not 0 <= p <= 1:
            raise ValueError(f"Probability p must be in [0, 1], got {p}")
        if max_count <= 0:
            raise ValueError(f"max_count must be positive, got {max_count}")
        if max_repeats <= 0:
            raise ValueError(f"max_repeats must be positive, got {max_repeats}")
        self.p = p
        self.max_count = max_count
        self.max_repeats = max_repeats
    
    def __call__(self, landmarks: torch.Tensor, label: int) -> tuple:
        """Apply random sequence repetition.
        
        Args:
            landmarks: Tensor of shape (T, 24) containing landmark coordinates.
            label: Integer count (e.g., number of push-ups).
            
        Returns:
            Tuple of (repeated_landmarks, new_label) where:
            - repeated_landmarks: Tensor of shape (T * num_repeats, 24)
            - new_label: Integer = label * num_repeats
        """
        if torch.rand(1).item() >= self.p:
            return landmarks, label
        
        if label <= 0:
            return landmarks, label
        
        # Calculate maximum allowed repeats based on max_count constraint
        max_allowed_repeats = self.max_count // label
        
        if max_allowed_repeats <= 1:
            # Can't repeat without exceeding max_count
            return landmarks, label
        
        # Cap at max_repeats parameter (total copies, not additional copies)
        max_allowed_repeats = min(max_allowed_repeats, self.max_repeats + 1)
        
        # Randomly choose number of total copies (at least 2 for a repeat to happen)
        num_copies = torch.randint(2, max_allowed_repeats + 1, (1,)).item()
        
        # Concatenate the sequence
        repeated_landmarks = torch.cat([landmarks] * num_copies, dim=0)
        new_label = label * num_copies
        
        return repeated_landmarks, new_label

class RandomHorizontalFlipLandmarks:
    """Randomly flip landmark sequences horizontally.
    
    This transform performs a horizontal flip on pose landmarks by:
    1. Mirroring x-coordinates around the center (0.5 for normalized coordinates)
    2. Swapping left and right landmark pairs to maintain anatomical consistency
    
    The landmark layout is assumed to be (T, 24) where 24 = 8 landmarks × 3 coords:
    - Indices 0-2: Left Shoulder (x, y, z)
    - Indices 3-5: Right Shoulder (x, y, z)
    - Indices 6-8: Left Elbow (x, y, z)
    - Indices 9-11: Right Elbow (x, y, z)
    - Indices 12-14: Left Wrist (x, y, z)
    - Indices 15-17: Right Wrist (x, y, z)
    - Indices 18-20: Left Hip (x, y, z)
    - Indices 21-23: Right Hip (x, y, z)
    
    Args:
        p: Probability of applying the flip. Default is 0.5.
    """
    
    # Pairs of (left_start_idx, right_start_idx) for each landmark pair
    # Each landmark has 3 values (x, y, z), so we swap in groups of 3
    LANDMARK_PAIRS = [
        (0, 3),    # Left/Right Shoulder
        (6, 9),    # Left/Right Elbow
        (12, 15),  # Left/Right Wrist
        (18, 21),  # Left/Right Hip
    ]
    
    # X-coordinate indices (every 3rd value starting at 0)
    X_INDICES = [0, 3, 6, 9, 12, 15, 18, 21]
    
    def __init__(self, p: float = 0.5):
        if not 0 <= p <= 1:
            raise ValueError(f"Probability p must be in [0, 1], got {p}")
        self.p = p
    
    def __call__(self, landmarks: torch.Tensor) -> torch.Tensor:
        """Apply random horizontal flip to landmarks.
        
        Args:
            landmarks: Tensor of shape (T, 24) containing landmark coordinates.
            
        Returns:
            Flipped landmarks tensor of the same shape, or original if not flipped.
        """
        if torch.rand(1).item() >= self.p:
            return landmarks
        
        # Clone to avoid modifying original tensor
        flipped = landmarks.clone()
        
        # 1. Mirror x-coordinates around center (0.5)
        # For normalized coordinates in [0, 1], flipped_x = 1 - x
        for x_idx in self.X_INDICES:
            flipped[:, x_idx] = 1.0 - flipped[:, x_idx]
        
        # 2. Swap left and right landmark pairs
        for left_start, right_start in self.LANDMARK_PAIRS:
            # Swap all 3 coordinates (x, y, z) for each pair
            temp = flipped[:, left_start:left_start+3].clone()
            flipped[:, left_start:left_start+3] = flipped[:, right_start:right_start+3]
            flipped[:, right_start:right_start+3] = temp
        
        return flipped

class RandomTemporalJitter:
    """Randomly subsample frames from the landmark sequence.
    
    This transform simulates temporal variations by randomly dropping frames
    from the sequence, effectively creating a faster or jittered version of
    the motion.
    
    Args:
        p: Probability of applying the jitter. Default is 0.5.
        drop_ratio: Tuple of (min_ratio, max_ratio) for the fraction of frames
                    to keep. Default is (0.7, 1.0), meaning 70-100% of frames
                    are kept.
    """
    
    def __init__(self, p: float = 0.5, drop_ratio: tuple = (0.7, 1.0)):
        if not 0 <= p <= 1:
            raise ValueError(f"Probability p must be in [0, 1], got {p}")
        if not (0 < drop_ratio[0] <= drop_ratio[1] <= 1.0):
            raise ValueError(f"drop_ratio must satisfy 0 < min <= max <= 1, got {drop_ratio}")
        self.p = p
        self.drop_ratio = drop_ratio
    
    def __call__(self, landmarks: torch.Tensor) -> torch.Tensor:
        """Apply random temporal jitter to landmarks.
        
        Args:
            landmarks: Tensor of shape (T, 24) containing landmark coordinates.
            
        Returns:
            Jittered landmarks tensor of shape (T', 24) where T' <= T.
        """
        if torch.rand(1).item() >= self.p:
            return landmarks
        
        T = landmarks.shape[0]
        if T <= 2:  # Don't jitter very short sequences
            return landmarks
        
        # Randomly determine how many frames to keep
        keep_ratio = torch.empty(1).uniform_(self.drop_ratio[0], self.drop_ratio[1]).item()
        num_keep = max(2, int(T * keep_ratio))  # Keep at least 2 frames
        
        # Randomly select frame indices (sorted to maintain temporal order)
        indices = torch.randperm(T)[:num_keep].sort().values
        
        return landmarks[indices]

class RandomNoise:
    """Add random Gaussian noise to landmark coordinates.
    
    This transform adds small random perturbations to simulate detection
    noise and make the model more robust to imprecise landmark detection.
    
    Args:
        p: Probability of applying noise. Default is 0.5.
        std: Standard deviation of the Gaussian noise. Default is 0.01.
             For normalized coordinates in [0, 1], 0.01 is about 1% noise.
    """
    
    def __init__(self, p: float = 0.5, std: float = 0.01):
        if not 0 <= p <= 1:
            raise ValueError(f"Probability p must be in [0, 1], got {p}")
        if std < 0:
            raise ValueError(f"std must be non-negative, got {std}")
        self.p = p
        self.std = std
    
    def __call__(self, landmarks: torch.Tensor) -> torch.Tensor:
        """Apply random Gaussian noise to landmarks.
        
        Args:
            landmarks: Tensor of shape (T, 24) containing landmark coordinates.
            
        Returns:
            Noisy landmarks tensor of the same shape.
        """
        if torch.rand(1).item() >= self.p:
            return landmarks
        
        # Add Gaussian noise
        noise = torch.randn_like(landmarks) * self.std
        return landmarks + noise