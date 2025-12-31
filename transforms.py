import torch
import torch.nn.functional as F

class Compose:
    """Compose multiple transforms together.
    
    Applies a sequence of transforms in order.
    
    Args:
        transforms: List of transform objects to apply.
    """
    
    def __init__(self, transforms: list):
        self.transforms = transforms
    
    def __call__(self, landmarks: torch.Tensor, label: int) -> tuple[torch.Tensor, int]:
        """Apply all transforms in sequence.
        
        Args:
            landmarks: Tensor of shape (T, 6) containing angle features.
            label: Integer count.
            
        Returns:
            Tuple of (transformed_landmarks, label).
        """
        for transform in self.transforms:
            landmarks, label = transform(landmarks, label)
        return landmarks, label


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
    
    This transform performs a horizontal flip on pose angle features by:
    1. Swapping left and right angle values to maintain anatomical consistency.
    
    The angle layout is assumed to be (T, 6):
    - Index 0: Left Elbow Angle
    - Index 1: Right Elbow Angle
    - Index 2: Left Shoulder Angle
    - Index 3: Right Shoulder Angle
    - Index 4: Left Body Angle
    - Index 5: Right Body Angle
    
    Args:
        p: Probability of applying the flip. Default is 0.5.
    """
    
    # Pairs of (left_idx, right_idx) for each angle pair
    ANGLE_PAIRS = [
        (0, 1),    # Left/Right Elbow
        (2, 3),    # Left/Right Shoulder
        (4, 5),    # Left/Right Body
    ]
    
    def __init__(self, p: float = 0.5):
        if not 0 <= p <= 1:
            raise ValueError(f"Probability p must be in [0, 1], got {p}")
        self.p = p
    
    def __call__(self, landmarks: torch.Tensor, label: int) -> tuple[torch.Tensor, int]:
        """Apply random horizontal flip to landmarks.
        
        Args:
            landmarks: Tensor of shape (T, 6) containing angle features.
            label: Integer count.
            
        Returns:
            Tuple of (flipped_landmarks, label).
        """
        if torch.rand(1).item() >= self.p:
            return landmarks, label
        
        # Clone to avoid modifying original tensor
        flipped = landmarks.clone()
        
        # Swap left and right angle pairs
        # Note: Angles are scalars (degrees), so mirroring the image just swaps 
        # the angle values between left and right sides. No value inversion needed.
        for left_idx, right_idx in self.ANGLE_PAIRS:
            temp = flipped[:, left_idx].clone()
            flipped[:, left_idx] = flipped[:, right_idx]
            flipped[:, right_idx] = temp
        
        return flipped, label


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
    
    def __call__(self, landmarks: torch.Tensor, label: int) -> tuple[torch.Tensor, int]:
        """Apply random temporal jitter to landmarks.
        
        Args:
            landmarks: Tensor of shape (T, 6) containing angle features.
            label: Integer count.
            
        Returns:
            Tuple of (jittered_landmarks, label).
        """
        if torch.rand(1).item() >= self.p:
            return landmarks, label
        
        T = landmarks.shape[0]
        if T <= 2:  # Don't jitter very short sequences
            return landmarks, label
        
        # Randomly determine how many frames to keep
        keep_ratio = torch.empty(1).uniform_(self.drop_ratio[0], self.drop_ratio[1]).item()
        num_keep = max(2, int(T * keep_ratio))  # Keep at least 2 frames
        
        # Randomly select frame indices (sorted to maintain temporal order)
        indices = torch.randperm(T)[:num_keep].sort().values
        
        return landmarks[indices], label


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
    
    def __call__(self, landmarks: torch.Tensor, label: int) -> tuple[torch.Tensor, int]:
        """Apply random Gaussian noise to landmarks.
        
        Args:
            landmarks: Tensor of shape (T, 6) containing angle features.
            label: Integer count.
            
        Returns:
            Tuple of (noisy_landmarks, label).
        """
        if torch.rand(1).item() >= self.p:
            return landmarks, label
        
        # Add Gaussian noise
        noise = torch.randn_like(landmarks) * self.std
        return landmarks + noise, label


class RandomTimeWarp:
    """Resample temporal sequence to simulate speed variations.
    
    This transform resamples the sequence to a new length between 0.8x and 1.2x
    of the original, simulating videos recorded at different speeds or exercises
    performed at varying paces. Uses linear interpolation to maintain smooth
    angle transitions.
    
    Args:
        p: Probability of applying the warp. Default is 0.5.
        scale_range: Tuple of (min_scale, max_scale) for the resampling factor.
                     Default is (0.8, 1.2).
    """
    
    def __init__(self, p: float = 0.5, scale_range: tuple = (0.8, 1.2)):
        if not 0 <= p <= 1:
            raise ValueError(f"Probability p must be in [0, 1], got {p}")
        if not (0 < scale_range[0] <= scale_range[1]):
            raise ValueError(f"scale_range must satisfy 0 < min <= max, got {scale_range}")
        self.p = p
        self.scale_range = scale_range
    
    def __call__(self, landmarks: torch.Tensor, label: int) -> tuple[torch.Tensor, int]:
        """Apply random time warp to landmarks.
        
        Args:
            landmarks: Tensor of shape (T, 6) containing angle features.
            label: Integer count.
            
        Returns:
            Tuple of (warped_landmarks, label).
        """
        if torch.rand(1).item() >= self.p:
            return landmarks, label
        
        T = landmarks.shape[0]
        if T <= 2:  # Don't warp very short sequences
            return landmarks, label
        
        # Sample scale factor uniformly from scale_range
        scale = torch.empty(1).uniform_(self.scale_range[0], self.scale_range[1]).item()
        T_new = max(2, int(T * scale))
        
        # Reshape for F.interpolate: (T, 6) -> (1, 6, T)
        x = landmarks.T.unsqueeze(0).float()
        
        # Interpolate using linear mode
        x_warped = F.interpolate(x, size=T_new, mode='linear', align_corners=True)
        
        # Reshape back: (1, 6, T_new) -> (T_new, 6)
        return x_warped.squeeze(0).T, label


class RandomSequenceReverse:
    """Randomly reverse the temporal order of the angle sequence.
    
    This transform flips the sequence along the time axis, effectively
    making a push-up motion appear to go in reverse. This is a valid
    augmentation because:
    1. The biomechanics of a push-up (descend then ascend) reversed 
       (ascend then descend) still represents valid motion dynamics.
    2. It doubles the effective training data variety.
    
    Args:
        p: Probability of applying the reverse. Default is 0.5.
    """
    
    def __init__(self, p: float = 0.5):
        if not 0 <= p <= 1:
            raise ValueError(f"Probability p must be in [0, 1], got {p}")
        self.p = p
    
    def __call__(self, landmarks: torch.Tensor, label: int) -> tuple[torch.Tensor, int]:
        """Apply random sequence reversal.
        
        Args:
            landmarks: Tensor of shape (T, 6) containing angle features.
            label: Integer count.
            
        Returns:
            Tuple of (reversed_landmarks, label).
        """
        if torch.rand(1).item() >= self.p:
            return landmarks, label
        
        # Reverse along the time dimension (dim=0)
        return torch.flip(landmarks, dims=[0]), label


class RandomScaling:
    """Randomly scale the angle values by a factor.
    
    This transform applies a multiplicative scaling to the normalized angle
    values, simulating variations in range of motion. For example, someone
    with greater flexibility might achieve larger joint angles during push-ups.
    
    Since angles are normalized between 0 and 1, the result is clamped to
    stay within [0, 1].
    
    Args:
        p: Probability of applying the scaling. Default is 0.5.
        scale_range: Tuple of (min_scale, max_scale) for the scaling factor.
                     Default is (0.8, 1.2) for +/- 20% variation.
    """
    
    def __init__(self, p: float = 0.5, scale_range: tuple = (0.8, 1.2)):
        if not 0 <= p <= 1:
            raise ValueError(f"Probability p must be in [0, 1], got {p}")
        if not (0 < scale_range[0] <= scale_range[1]):
            raise ValueError(f"scale_range must satisfy 0 < min <= max, got {scale_range}")
        self.p = p
        self.scale_range = scale_range
    
    def __call__(self, landmarks: torch.Tensor, label: int) -> tuple[torch.Tensor, int]:
        """Apply random scaling to angle values.
        
        Args:
            landmarks: Tensor of shape (T, 6) containing normalized angle features
                       in the range [0, 1].
            label: Integer count.
            
        Returns:
            Tuple of (scaled_landmarks, label).
        """
        if torch.rand(1).item() >= self.p:
            return landmarks, label
        
        # Sample scale factor uniformly from scale_range
        scale = torch.empty(1).uniform_(self.scale_range[0], self.scale_range[1]).item()
        
        # Apply scaling and clamp to valid range
        scaled = landmarks * scale
        return torch.clamp(scaled, 0.0, 1.0), label
