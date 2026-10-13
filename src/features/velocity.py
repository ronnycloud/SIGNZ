"""
Velocity Feature Computer
==========================
Computes inter-frame velocity (first-order temporal derivatives) for all
landmark coordinates. Velocity features are critical for sign language
recognition because many signs differ only in motion trajectory while
sharing the same start and end hand shapes.

Example:
    "YES" (fist nodding up-down) vs "NO" (fist shaking left-right)
    → identical hand shape, distinguished only by velocity direction.

The velocity vector has the same dimensionality as the base landmarks (226),
doubling the total feature size when concatenated.
"""

import numpy as np
from typing import Optional


class VelocityComputer:
    """Computes frame-to-frame velocity of landmark coordinates.

    Maintains a single-frame buffer for streaming (real-time) computation.
    Also supports batch computation for offline dataset processing.
    """

    def __init__(self):
        self._prev_frame: Optional[np.ndarray] = None

    def compute_velocity(self, current: np.ndarray) -> np.ndarray:
        """Compute velocity for a single frame in a stream.

        Parameters
        ----------
        current : np.ndarray
            Landmark vector for the current frame (226 dims).

        Returns
        -------
        np.ndarray
            Velocity vector (226 dims). Zero for the first frame.
        """
        if self._prev_frame is None:
            velocity = np.zeros_like(current, dtype=np.float32)
        else:
            velocity = (current - self._prev_frame).astype(np.float32)

        self._prev_frame = current.copy()
        return velocity

    @staticmethod
    def compute_batch_velocity(sequence: np.ndarray) -> np.ndarray:
        """Compute velocity for an entire sequence at once.

        Parameters
        ----------
        sequence : np.ndarray
            Shape (T, D) where T = number of frames, D = feature dims.

        Returns
        -------
        np.ndarray
            Shape (T, D) velocity. First frame velocity is zero.
        """
        velocity = np.zeros_like(sequence, dtype=np.float32)
        if len(sequence) > 1:
            velocity[1:] = sequence[1:] - sequence[:-1]
        return velocity

    @staticmethod
    def compute_acceleration(sequence: np.ndarray) -> np.ndarray:
        """Compute second-order derivative (acceleration) for a sequence.

        Not used by default in the main pipeline, but available for
        ablation studies or more advanced motion encoding.

        Parameters
        ----------
        sequence : np.ndarray
            Shape (T, D).

        Returns
        -------
        np.ndarray
            Shape (T, D). First two frames are zero.
        """
        velocity = VelocityComputer.compute_batch_velocity(sequence)
        acceleration = np.zeros_like(velocity, dtype=np.float32)
        if len(velocity) > 1:
            acceleration[1:] = velocity[1:] - velocity[:-1]
        return acceleration

    def build_full_feature(
        self,
        landmarks: np.ndarray,
        hand_flags: np.ndarray,
    ) -> np.ndarray:
        """Build the complete 454-dim feature vector for one frame.

        Concatenates: landmarks (226) + velocity (226) + hand_flags (2)

        Parameters
        ----------
        landmarks : np.ndarray
            Normalized landmark vector (226 dims).
        hand_flags : np.ndarray
            Hand presence flags [left_present, right_present] (2 dims).

        Returns
        -------
        np.ndarray
            Full feature vector (454 dims).
        """
        velocity = self.compute_velocity(landmarks)
        return np.concatenate([landmarks, velocity, hand_flags]).astype(np.float32)

    @staticmethod
    def build_batch_features(
        landmarks_seq: np.ndarray,
        hand_flags_seq: np.ndarray,
    ) -> np.ndarray:
        """Build full features for an entire sequence (batch/offline mode).

        Parameters
        ----------
        landmarks_seq : np.ndarray
            Shape (T, 226) — normalized landmarks.
        hand_flags_seq : np.ndarray
            Shape (T, 2) — hand presence flags per frame.

        Returns
        -------
        np.ndarray
            Shape (T, 454) — full feature vectors.
        """
        velocity = VelocityComputer.compute_batch_velocity(landmarks_seq)
        return np.concatenate(
            [landmarks_seq, velocity, hand_flags_seq], axis=1
        ).astype(np.float32)

    def reset(self):
        """Clear the velocity buffer (call between sequences)."""
        self._prev_frame = None
