import torch
import torch.nn.functional as F

from feature_engineering.constants import (
    LEFT_ELBOW_ANGLE_IDX,
    RIGHT_ELBOW_ANGLE_IDX,
    LEFT_SHOULDER_ANGLE_IDX,
    RIGHT_SHOULDER_ANGLE_IDX,
    LEFT_BODY_ANGLE_IDX,
    RIGHT_BODY_ANGLE_IDX,
)


class Compose:
    """Compose multiple transforms together.
    
    Applies a sequence of transforms in order.
    
    Args:
        transforms: List of transform objects to apply.
    """
    
    def __init__(self, transforms: list):
        self.transforms = transforms
    
    def __call__(self, landmarks: torch.Tensor, density_map: torch.Tensor, label: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Apply all transforms in sequence.
        
        Args:
            landmarks: Tensor of shape (T, 6) containing angle features.
            density_map: Tensor of shape (T,) containing gaussian density map.
            label: Integer count.
            
        Returns:
            Tuple of (transformed_landmarks, transformed_density_map, label).
        """
        for transform in self.transforms:
            landmarks, density_map, label = transform(landmarks, density_map, label)
        return landmarks, density_map, label


class RandomSequenceRepeat:
    """Repeat landmark sequences to generate training examples with higher counts.
    
    This transform concatenates a sequence with itself one or more times to create
    a training example representing more pushups. For example, a sequence with 2
    pushups repeated twice becomes a sequence representing 4 pushups.
    
    Args:
        p: Probability of applying the repeat. Default is 0.5.
        max_count: Maximum allowed count after repetition. Default is 10.
        max_repeat_factor: Maximum multiplication factor. Default is 4 (sequence can be 1x to 4x).
    """
    
    def __init__(self, p: float = 0.5, max_count: int = 10, max_repeat_factor: int = 4):
        self.p = p
        self.max_count = max_count
        self.max_repeat_factor = max_repeat_factor
    
    def __call__(self, landmarks: torch.Tensor, density_map: torch.Tensor, label: int) -> tuple:
        """Apply random sequence repetition.
        
        Args:
            landmarks: Tensor of shape (T, 6) containing landmark coordinates.
            density_map: Tensor of shape (T,) containing gaussian density map.
            label: Integer count (e.g., number of push-ups).
            
        Returns:
            Tuple of (repeated_landmarks, repeated_density_map, new_label) where:
            - repeated_landmarks: Tensor of shape (T * factor, 6)
            - repeated_density_map: Tensor of shape (T * factor,)
            - new_label: Integer = label * factor
        """
        if torch.rand(1).item() >= self.p:
            return landmarks, density_map, label
        
        if label <= 0:
            return landmarks, density_map, label
        
        max_allowed_factor = self.max_count // label
        
        if max_allowed_factor < 2:
            return landmarks, density_map, label
        
        max_allowed_factor = min(max_allowed_factor, self.max_repeat_factor)
        
        factor = torch.randint(2, max_allowed_factor + 1, (1,)).item()
        
        repeated_landmarks = landmarks.repeat(factor, 1)
        repeated_density_map = density_map.repeat(factor)
        new_label = label * factor
        
        return repeated_landmarks, repeated_density_map, new_label


class RandomHorizontalFlipLandmarks:
    """Randomly flip landmark sequences horizontally.
    
    This transform performs a horizontal flip on pose angle features by swapping
     left and right angle values to maintain anatomical consistency.

    Args:
        p: Probability of applying the flip. Default is 0.5.
    """
    
    _ANGLE_PAIRS = [
        (LEFT_ELBOW_ANGLE_IDX, RIGHT_ELBOW_ANGLE_IDX),
        (LEFT_SHOULDER_ANGLE_IDX, RIGHT_SHOULDER_ANGLE_IDX),
        (LEFT_BODY_ANGLE_IDX, RIGHT_BODY_ANGLE_IDX)
    ]
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, landmarks: torch.Tensor, density_map: torch.Tensor, label: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Apply random horizontal flip to landmarks.
        
        Args:
            landmarks: Tensor of shape (T, 6) containing angle features.
            density_map: Tensor of shape (T,) containing gaussian density map.
            label: Integer count.
            
        Returns:
            Tuple of (flipped_landmarks, density_map, label).
        """
        if torch.rand(1).item() >= self.p:
            return landmarks, density_map, label
        
        flipped = landmarks.clone()
        
        for left_idx, right_idx in self._ANGLE_PAIRS:
            temp = flipped[:, left_idx].clone()
            flipped[:, left_idx] = flipped[:, right_idx]
            flipped[:, right_idx] = temp
        
        return flipped, density_map, label


class RandomNoise:
    """Add random Gaussian noise to the joint angles.
    
    This transform adds gaussian noise to the joint angles to simulate
    detection noise and to make the model more robust to imprecise
    pose detection.
    
    Args:
        p: Probability of applying noise. Default is 0.5.
        std: Standard deviation of the Gaussian noise. Default is 0.01.
    """
    
    def __init__(self, p: float = 0.5, std: float = 0.01):
        self.p = p
        self.std = std
    
    def __call__(self, landmarks: torch.Tensor, density_map: torch.Tensor, label: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Apply random Gaussian noise to joint angles.
        
        Args:
            landmarks: Tensor of shape (T, 6) containing angle features.
            density_map: Tensor of shape (T,) containing gaussian density map.
            label: Integer count.
            
        Returns:
            Tuple of (noisy_landmarks, density_map, label).
        """
        if torch.rand(1).item() >= self.p:
            return landmarks, density_map, label
        
        noise = torch.empty_like(landmarks).normal_(std=self.std)
        return landmarks + noise, density_map, label


class RandomTimeWarp:
    """Resample temporal sequence to simulate speed variations.
    
    This transform resamples the sequence to a new length of the original,
    simulating pushups performed at varying paces. Uses linear interpolation
    to resample angle values
    
    Args:
        p: Probability of applying the warp. Default is 0.5.
        scale_range: Tuple of (min_scale, max_scale) for the resampling factor.
                     Default is (0.7, 1.4).
    """
    
    def __init__(self, p: float = 0.5, scale_range: tuple = (0.7, 1.4)):
        self.p = p
        self.scale_range = scale_range
    
    def __call__(self, landmarks: torch.Tensor, density_map: torch.Tensor, label: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Apply random time warp to landmarks.
        
        Args:
            landmarks: Tensor of shape (T, 6) containing angle features.
            density_map: Tensor of shape (T,) containing gaussian density map.
            label: Integer count.
            
        Returns:
            Tuple of (warped_landmarks, warped_density_map, label).
        """
        if torch.rand(1).item() >= self.p:
            return landmarks, density_map, label
        
        num_frames = landmarks.shape[0]

        scale = torch.empty(1).uniform_(self.scale_range[0], self.scale_range[1]).item()
        num_frames_new = int(num_frames * scale)
        
        # Reshape for F.interpolate: (T, 6) -> (1, 6, T)
        sequence = landmarks.T.unsqueeze(0).float()
        
        sequence_warped = F.interpolate(sequence, size=num_frames_new, mode='linear', align_corners=True)
        
        # Also warp density_map: (T,) -> (1, 1, T) -> interpolate -> (T_new,)
        density_map_reshaped = density_map.unsqueeze(0).unsqueeze(0).float()
        density_map_warped = F.interpolate(density_map_reshaped, size=num_frames_new, mode='linear', align_corners=True)
        
        # Reshape back: (1, 6, T_new) -> (T_new, 6), (1, 1, T_new) -> (T_new,)
        sequence_result = sequence_warped.squeeze(0).T
        density_map_result = density_map_warped.squeeze(0).squeeze(0)
        
        # Renormalize density map sum to match original sum since interpolation changes
        # the area under the curve
        current_sum = density_map_result.sum()
        original_sum = density_map.sum()
        if current_sum > 0:
            density_map_result *= (original_sum / current_sum)

        return sequence_result, density_map_result, label


class RandomSequenceReverse:
    """Randomly reverse the temporal order of the angle sequence.
    
    This transform flips the sequence along the time axis, effectively
    making a pushup motion appear to go in reverse. This is a valid
    augmentation because the biomechanics of a push-up (descend then
    ascend) reversed (ascend then descend) still represents a valid
    pushup.

    Args:
        p: Probability of applying the reverse. Default is 0.5.
    """
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, landmarks: torch.Tensor, density_map: torch.Tensor, label: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Apply random sequence reversal.
        
        Args:
            landmarks: Tensor of shape (T, 6) containing angle features.
            density_map: Tensor of shape (T,) containing gaussian density map.
            label: Integer count.
            
        Returns:
            Tuple of (reversed_landmarks, reversed_density_map, label).
        """
        if torch.rand(1).item() >= self.p:
            return landmarks, density_map, label
        
        return torch.flip(landmarks, dims=[0]), torch.flip(density_map, dims=[0]), label


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
                     Default is (0.9, 1.1) for +/- 10% variation.
    """
    
    def __init__(self, p: float = 0.5, scale_range: tuple = (0.9, 1.1)):
        self.p = p
        self.scale_range = scale_range
    
    def __call__(self, landmarks: torch.Tensor, density_map: torch.Tensor, label: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Apply random scaling to angle values.
        
        Args:
            landmarks: Tensor of shape (T, 6) containing normalized angle features
                       in the range [0, 1].
            density_map: Tensor of shape (T,) containing gaussian density map.
            label: Integer count.
            
        Returns:
            Tuple of (scaled_landmarks, density_map, label).
        """
        if torch.rand(1).item() >= self.p:
            return landmarks, density_map, label
        
        scale = torch.empty(1).uniform_(self.scale_range[0], self.scale_range[1]).item()
        
        scaled = landmarks * scale
        return torch.clamp(scaled, 0.0, 1.0), density_map, label


class RandomDropout:
    """Randomly set individual angle values to zero.
    
    This transform randomly zeros out individual angle values in the sequence.
    
    This simulates real-world pose estimation failures where individual joints
    may be missed due to occlusion, motion blur, or low confidence detections.
    
    By learning to count from incomplete/corrupted sequences, the model becomes
    more robust to failures in the pose detection model.
    
    Args:
        p: Probability of applying dropout. Default is 0.5.
        dropout_rate: The fraction of individual angle values to zero out.
                      Default is 0.1 (10% dropout).
    """
    
    def __init__(self, p: float = 0.5, dropout_rate: float = 0.1):
        self.p = p
        self.dropout_rate = dropout_rate
    
    def __call__(self, landmarks: torch.Tensor, density_map: torch.Tensor, label: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Apply random dropout to landmarks.
        
        Args:
            landmarks: Tensor of shape (T, 6) containing angle features.
            density_map: Tensor of shape (T,) containing gaussian density map.
            label: Integer count.
            
        Returns:
            Tuple of (dropped_landmarks, density_map, label).
        """
        if torch.rand(1).item() >= self.p:
            return landmarks, density_map, label
        
        keep_mask = torch.rand_like(landmarks) >= self.dropout_rate
        
        return landmarks * keep_mask.float(), density_map, label
