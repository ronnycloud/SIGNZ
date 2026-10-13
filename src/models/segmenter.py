"""
Adaptive Gesture Segmenter
===========================
Detects gesture boundaries in a continuous stream of keypoint features,
replacing the fixed 30-frame sliding window with learned boundary detection.

Architecture:
    1D Conv (local pattern extraction) → Bidirectional GRU (temporal context)
    → per-frame binary classifier: P(boundary | frame_t)

Boundary frames are detected via peak-finding on the probability signal.
Between consecutive peaks, each segment is classified independently.

Fallback: When no trained segmenter model is available, an energy-based
heuristic uses the L2 norm of velocity features to detect pauses between
signs (velocity drops below a threshold for several consecutive frames).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
from scipy.signal import find_peaks


class BoundaryDetector(nn.Module):
    """Neural boundary detector: Conv1D → BiGRU → binary classifier.

    Lightweight by design (~50K params) so it doesn't bottleneck real-time
    inference. Runs on the same feature stream as the main classifier.
    """

    def __init__(
        self,
        input_dim: int = 454,
        conv_channels: int = 64,
        conv_kernel: int = 5,
        gru_hidden: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()

        # 1D convolution for local pattern extraction
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, conv_channels, kernel_size=conv_kernel, padding=conv_kernel // 2),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Bidirectional GRU for temporal context
        self.gru = nn.GRU(
            input_size=conv_channels,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Per-frame binary classifier
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, T, input_dim)

        Returns
        -------
        (batch, T) boundary probabilities in [0, 1]
        """
        # Conv1D expects (batch, channels, time)
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)  # (B, T, conv_channels)
        h, _ = self.gru(h)  # (B, T, gru_hidden*2)
        logits = self.classifier(h).squeeze(-1)  # (B, T)
        return torch.sigmoid(logits)


class GestureSegmenter:
    """Segments continuous feature streams into individual gesture clips.

    Supports two modes:
    1. Neural mode — uses a trained BoundaryDetector
    2. Energy fallback — uses velocity magnitude heuristic
    """

    def __init__(
        self,
        boundary_threshold: float = 0.5,
        min_gesture_frames: int = 10,
        max_gesture_frames: int = 90,
        energy_threshold: float = 0.02,
        energy_patience: int = 5,
        use_neural: bool = False,
        model_path: Optional[str] = None,
        device: str = "cpu",
    ):
        self.boundary_threshold = boundary_threshold
        self.min_gesture_frames = min_gesture_frames
        self.max_gesture_frames = max_gesture_frames
        self.energy_threshold = energy_threshold
        self.energy_patience = energy_patience
        self.use_neural = use_neural
        self.device = device

        # Neural boundary detector
        self.detector: Optional[BoundaryDetector] = None
        if use_neural and model_path:
            self.detector = BoundaryDetector()
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            self.detector.load_state_dict(checkpoint["model_state_dict"])
            self.detector.eval()
            self.detector.to(device)

        # State for streaming segmentation
        self._buffer: List[np.ndarray] = []
        self._is_signing = False
        self._idle_count = 0

    def segment_batch(
        self,
        features: np.ndarray,
    ) -> List[np.ndarray]:
        """Segment a pre-recorded feature sequence (offline mode).

        Parameters
        ----------
        features : np.ndarray, shape (T, feature_dim)

        Returns
        -------
        List of segments, each shape (T_i, feature_dim)
        """
        if self.use_neural and self.detector is not None:
            return self._neural_segment(features)
        return self._energy_segment(features)

    def _neural_segment(self, features: np.ndarray) -> List[np.ndarray]:
        """Segment using the trained boundary detector."""
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            boundary_probs = self.detector(x).squeeze(0).cpu().numpy()

        # Find peaks in boundary probability
        peaks, _ = find_peaks(
            boundary_probs,
            height=self.boundary_threshold,
            distance=self.min_gesture_frames,
        )

        # Add start and end as boundaries
        boundaries = [0] + peaks.tolist() + [len(features)]

        segments = []
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i + 1]
            length = end - start
            if self.min_gesture_frames <= length <= self.max_gesture_frames:
                segments.append(features[start:end])

        return segments

    def _energy_segment(self, features: np.ndarray) -> List[np.ndarray]:
        """Segment using velocity energy heuristic (fallback).

        Algorithm:
            1. Compute velocity magnitude per frame (L2 norm of velocity dims)
            2. Smooth with a moving average
            3. Mark frames where energy < threshold as "idle"
            4. Find contiguous "active" regions → gesture segments
        """
        T = len(features)
        if T < self.min_gesture_frames:
            return []

        # Velocity features are at indices 226:452 in the 454-dim vector
        velocity = features[:, 226:452]  # (T, 226)
        energy = np.linalg.norm(velocity, axis=1)  # (T,)

        # Smooth energy with moving average (window=5)
        kernel = np.ones(5) / 5
        energy_smooth = np.convolve(energy, kernel, mode="same")

        # Binary activity signal
        is_active = energy_smooth > self.energy_threshold

        # Find contiguous active regions
        segments = []
        start = None
        for i in range(T):
            if is_active[i] and start is None:
                start = i
            elif not is_active[i] and start is not None:
                length = i - start
                if self.min_gesture_frames <= length <= self.max_gesture_frames:
                    segments.append(features[start:i])
                start = None

        # Handle segment extending to end
        if start is not None:
            length = T - start
            if self.min_gesture_frames <= length <= self.max_gesture_frames:
                segments.append(features[start:T])

        return segments

    def update_stream(self, frame_features: np.ndarray) -> Optional[np.ndarray]:
        """Process one frame in streaming mode.

        Call this once per frame during real-time inference. When a complete
        gesture is detected (signing stops), the accumulated segment is
        returned and the buffer is cleared.

        Parameters
        ----------
        frame_features : np.ndarray, shape (454,)

        Returns
        -------
        np.ndarray or None
            Complete gesture segment if boundary detected, else None.
        """
        # Compute frame energy from velocity features
        velocity = frame_features[226:452]
        energy = np.linalg.norm(velocity)

        if energy > self.energy_threshold:
            # Active frame
            self._is_signing = True
            self._idle_count = 0
            self._buffer.append(frame_features)

            # Enforce max length
            if len(self._buffer) >= self.max_gesture_frames:
                segment = np.array(self._buffer, dtype=np.float32)
                self._buffer.clear()
                self._is_signing = False
                return segment
        else:
            # Idle frame
            if self._is_signing:
                self._idle_count += 1
                self._buffer.append(frame_features)

                if self._idle_count >= self.energy_patience:
                    # End of gesture detected
                    if len(self._buffer) >= self.min_gesture_frames:
                        segment = np.array(self._buffer, dtype=np.float32)
                        self._buffer.clear()
                        self._is_signing = False
                        self._idle_count = 0
                        return segment
                    else:
                        # Too short — discard
                        self._buffer.clear()
                        self._is_signing = False
                        self._idle_count = 0

        return None

    @property
    def is_signing(self) -> bool:
        """Whether the signer is currently in a gesture."""
        return self._is_signing

    @property
    def buffer_length(self) -> int:
        """Number of frames currently in the buffer."""
        return len(self._buffer)

    def reset(self):
        """Clear the streaming buffer."""
        self._buffer.clear()
        self._is_signing = False
        self._idle_count = 0
