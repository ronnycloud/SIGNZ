"""
Data Augmentation for Sign Language Sequences
==============================================
Applies spatial and temporal augmentations to 454-dim feature sequences during training:
  1. Spatial Noise (Gaussian noise on landmarks)
  2. Scale Variation (random rescaling of landmark coordinates)
  3. Mirroring / Hand-Swapping (swap left and right hand coordinates)
  4. Speed / Temporal Jitter Variation (resampling frames along time axis)
"""

import numpy as np
from typing import Dict, Any, Optional


class SequenceAugmentor:
    """Applies randomized augmentations to landmark sequence features (T, D)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
        self.spatial_noise = self.config.get("spatial_noise", True)
        self.spatial_noise_std = self.config.get("spatial_noise_std", 0.005)
        self.scale_variation = self.config.get("scale_variation", True)
        self.scale_range = tuple(self.config.get("scale_range", (0.9, 1.1)))
        self.mirror = self.config.get("mirror", True)
        self.speed_variation = self.config.get("speed_variation", True)
        self.speed_range = tuple(self.config.get("speed_range", (0.8, 1.2)))

    def apply_spatial_noise(self, seq: np.ndarray) -> np.ndarray:
        """Add small Gaussian noise to spatial landmark coordinates."""
        if not self.spatial_noise:
            return seq
        noise = np.random.normal(0, self.spatial_noise_std, size=seq.shape).astype(np.float32)
        # Apply noise only to landmark/velocity coordinates, not hand flags
        seq_aug = seq.copy()
        seq_aug[:, :-2] += noise[:, :-2]
        return seq_aug

    def apply_scale_variation(self, seq: np.ndarray) -> np.ndarray:
        """Rescale landmark coordinates by a random factor."""
        if not self.scale_variation:
            return seq
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        seq_aug = seq.copy()
        # Scale landmark and velocity positions
        seq_aug[:, :-2] *= scale
        return seq_aug

    def apply_mirror(self, seq: np.ndarray) -> np.ndarray:
        """Swap left hand and right hand features (horizontal mirror reflection)."""
        if not self.mirror or np.random.rand() < 0.5:
            return seq

        seq_aug = seq.copy()
        # In 454-dim vector:
        # left_hand(63), right_hand(63), pose(100), left_vel(63), right_vel(63), pose_vel(100), flags(2)
        # Swap LH (0:63) and RH (63:126)
        lh = seq_aug[:, 0:63].copy()
        rh = seq_aug[:, 63:126].copy()
        seq_aug[:, 0:63] = rh
        seq_aug[:, 63:126] = lh

        # Swap LH vel (226:289) and RH vel (289:352)
        lh_vel = seq_aug[:, 226:289].copy()
        rh_vel = seq_aug[:, 289:352].copy()
        seq_aug[:, 226:289] = rh_vel
        seq_aug[:, 289:352] = lh_vel

        # Swap hand presence flags [lh_present, rh_present]
        flags_lh = seq_aug[:, 452].copy()
        flags_rh = seq_aug[:, 453].copy()
        seq_aug[:, 452] = flags_rh
        seq_aug[:, 453] = flags_lh

        # Negate X coordinates for hand landmarks
        lh_coords = seq_aug[:, 0:63].reshape(-1, 21, 3)
        lh_coords[:, :, 0] *= -1
        seq_aug[:, 0:63] = lh_coords.reshape(-1, 63)

        rh_coords = seq_aug[:, 63:126].reshape(-1, 21, 3)
        rh_coords[:, :, 0] *= -1
        seq_aug[:, 63:126] = rh_coords.reshape(-1, 63)

        return seq_aug

    def apply_speed_variation(self, seq: np.ndarray) -> np.ndarray:
        """Resample sequence along time dimension (speed up or slow down)."""
        if not self.speed_variation:
            return seq

        T, D = seq.shape
        if T < 4:
            return seq

        speed_factor = np.random.uniform(self.speed_range[0], self.speed_range[1])
        new_T = max(4, int(T / speed_factor))

        old_indices = np.linspace(0, T - 1, num=T)
        new_indices = np.linspace(0, T - 1, num=new_T)

        resampled = np.zeros((new_T, D), dtype=np.float32)
        for d in range(D):
            resampled[:, d] = np.interp(new_indices, old_indices, seq[:, d])

        return resampled

    def __call__(self, seq: np.ndarray) -> np.ndarray:
        """Apply active augmentations to sequence."""
        if not self.enabled:
            return seq

        seq = self.apply_mirror(seq)
        seq = self.apply_scale_variation(seq)
        seq = self.apply_spatial_noise(seq)
        seq = self.apply_speed_variation(seq)
        return seq
