"""
Landmark Normalizer
====================
Applies geometric normalization to raw MediaPipe landmarks so that the model
is invariant to:
    1. **Translation** — wrist-centering removes absolute screen position.
    2. **Scale** — dividing by palm size removes distance-to-camera effects.
    3. **Missing data** — linear interpolation fills brief occlusions.

These three transforms are essential for signer-independent recognition:
without them, a model trained on one person at one distance fails on another.
"""

import numpy as np
from typing import Optional


class LandmarkNormalizer:
    """Normalizes hand and pose landmarks for translation/scale invariance."""

    # Wrist landmark index in each 21-landmark hand
    WRIST_IDX = 0
    # Middle finger MCP — used with wrist to compute palm size
    MCP_IDX = 9

    def __init__(self, interpolation_window: int = 3):
        """
        Parameters
        ----------
        interpolation_window : int
            Number of past frames to use for linear interpolation
            when landmarks are missing (zeros).
        """
        self.interpolation_window = interpolation_window
        # Buffers for temporal interpolation
        self._left_hand_buffer = []
        self._right_hand_buffer = []
        self._pose_buffer = []

    def normalize_hand(self, hand: np.ndarray) -> np.ndarray:
        """Normalize a single hand (63-dim vector → 63-dim normalized).

        Steps:
            1. Reshape to (21, 3)
            2. Subtract wrist position (translation invariance)
            3. Divide by palm size (scale invariance)
            4. Flatten back to (63,)

        If the hand is all zeros (not detected), returns zeros.
        """
        if np.allclose(hand, 0):
            return hand.copy()

        landmarks = hand.reshape(21, 3)

        # Step 1: Wrist-centering
        wrist = landmarks[self.WRIST_IDX].copy()
        centered = landmarks - wrist

        # Step 2: Scale normalization using palm size
        palm_size = np.linalg.norm(
            centered[self.MCP_IDX] - centered[self.WRIST_IDX]
        )
        if palm_size > 1e-6:
            centered = centered / palm_size

        return centered.flatten().astype(np.float32)

    def normalize_pose(self, pose: np.ndarray) -> np.ndarray:
        """Normalize upper-body pose (100-dim → 100-dim normalized).

        Steps:
            1. Reshape to (25, 4)  — (x, y, z, visibility)
            2. Compute shoulder midpoint as origin
            3. Compute shoulder width as scale
            4. Normalize x, y, z; preserve visibility
            5. Flatten back to (100,)
        """
        if np.allclose(pose, 0):
            return pose.copy()

        landmarks = pose.reshape(25, 4)
        coords = landmarks[:, :3].copy()
        visibility = landmarks[:, 3:4].copy()

        # Shoulder landmarks in upper-body subset:
        # Index 11 = left shoulder, Index 12 = right shoulder
        # (these are indices 11, 12 in the full 33-landmark pose,
        #  which map to indices 11, 12 in our upper-body subset of 0–24)
        left_shoulder = coords[11]
        right_shoulder = coords[12]
        midpoint = (left_shoulder + right_shoulder) / 2.0

        # Translation invariance
        coords = coords - midpoint

        # Scale invariance using shoulder width
        shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)
        if shoulder_width > 1e-6:
            coords = coords / shoulder_width

        normalized = np.concatenate([coords, visibility], axis=1)
        return normalized.flatten().astype(np.float32)

    def normalize_frame(
        self,
        left_hand: np.ndarray,
        right_hand: np.ndarray,
        pose: np.ndarray,
    ) -> np.ndarray:
        """Normalize all modalities and concatenate.

        Returns (226,) normalized feature vector.
        """
        lh = self.normalize_hand(left_hand)
        rh = self.normalize_hand(right_hand)
        p = self.normalize_pose(pose)
        return np.concatenate([lh, rh, p])

    def interpolate_missing(
        self,
        landmarks: np.ndarray,
        buffer: list,
    ) -> np.ndarray:
        """Fill missing (zero) landmarks via linear interpolation from buffer.

        If the current frame has zero landmarks but previous frames had valid
        data, linearly extrapolate from the last valid frames.
        """
        is_missing = np.allclose(landmarks, 0)

        if not is_missing:
            buffer.append(landmarks.copy())
            if len(buffer) > self.interpolation_window:
                buffer.pop(0)
            return landmarks

        # Attempt interpolation from buffer
        if len(buffer) >= 1:
            interpolated = buffer[-1].copy()
            if len(buffer) >= 2:
                # Linear extrapolation: last + (last - second_last)
                delta = buffer[-1] - buffer[-2]
                interpolated = buffer[-1] + delta
            buffer.append(interpolated)
            if len(buffer) > self.interpolation_window:
                buffer.pop(0)
            return interpolated

        return landmarks  # No buffer data, return zeros

    def normalize_with_interpolation(
        self,
        left_hand: np.ndarray,
        right_hand: np.ndarray,
        pose: np.ndarray,
    ) -> np.ndarray:
        """Normalize with missing-landmark interpolation.

        For real-time use — maintains internal state for temporal smoothing.
        """
        lh = self.interpolate_missing(left_hand, self._left_hand_buffer)
        rh = self.interpolate_missing(right_hand, self._right_hand_buffer)
        p = self.interpolate_missing(pose, self._pose_buffer)

        return self.normalize_frame(lh, rh, p)

    def reset(self):
        """Clear interpolation buffers (call between gesture sequences)."""
        self._left_hand_buffer.clear()
        self._right_hand_buffer.clear()
        self._pose_buffer.clear()
