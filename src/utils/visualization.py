"""
Visualization Utilities
========================
Drawing, annotation, and probability visualization for the real-time
detection interface. Replaces the drawing functions from the old utils.py
with a cleaner, more configurable implementation.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


# Colour palette — visually distinct, accessible colours
PALETTE = [
    (245, 117, 16),   # Orange
    (117, 245, 16),   # Green
    (16, 117, 245),   # Blue
    (245, 16, 117),   # Pink
    (16, 245, 245),   # Cyan
    (200, 80, 200),   # Purple
    (255, 200, 50),   # Gold
    (100, 200, 100),  # Lime
    (50, 50, 255),    # Red-blue
    (200, 200, 200),  # Grey
]


class Visualizer:
    """Handles all on-screen visualization for the real-time pipeline."""

    def __init__(
        self,
        show_landmarks: bool = True,
        show_probabilities: bool = True,
        show_sentence: bool = True,
        show_fps: bool = True,
    ):
        self.show_landmarks = show_landmarks
        self.show_probabilities = show_probabilities
        self.show_sentence = show_sentence
        self.show_fps = show_fps

    def draw_header(
        self,
        image: np.ndarray,
        sentence: str,
        width: int = 640,
    ) -> np.ndarray:
        """Draw the sentence output bar at the top of the frame."""
        if not self.show_sentence:
            return image
        cv2.rectangle(image, (0, 0), (width, 45), (30, 30, 30), -1)
        cv2.putText(
            image, sentence, (10, 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA,
        )
        return image

    def draw_fps(self, image: np.ndarray, fps: float) -> np.ndarray:
        """Draw FPS counter in the top-right corner."""
        if not self.show_fps:
            return image
        text = f"FPS: {fps:.1f}"
        h, w = image.shape[:2]
        cv2.putText(
            image, text, (w - 150, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA,
        )
        return image

    def draw_confidence(
        self,
        image: np.ndarray,
        sign_label: str,
        confidence: float,
    ) -> np.ndarray:
        """Draw the currently detected sign and confidence score."""
        h, w = image.shape[:2]
        text = f"{sign_label}: {confidence:.1%}"
        color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255)
        cv2.putText(
            image, text, (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA,
        )
        return image

    def draw_segmentation_status(
        self,
        image: np.ndarray,
        is_signing: bool,
        buffer_length: int,
    ) -> np.ndarray:
        """Draw gesture segmentation status indicator."""
        h, w = image.shape[:2]
        color = (0, 255, 0) if is_signing else (128, 128, 128)
        label = "SIGNING" if is_signing else "IDLE"
        cv2.circle(image, (w - 30, h - 30), 12, color, -1)
        cv2.putText(
            image, f"{label} ({buffer_length}f)", (w - 200, h - 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )
        return image


def prob_viz(
    probabilities: np.ndarray,
    actions: np.ndarray,
    input_frame: np.ndarray,
    colors: Optional[List[Tuple[int, int, int]]] = None,
    top_k: int = 5,
) -> np.ndarray:
    """Draw horizontal probability bars for the top-K classes.

    Enhanced version of the original prob_viz with:
    - Dynamic colour assignment
    - Top-K filtering (doesn't clutter screen with 200 bars)
    - Percentage labels
    """
    output = input_frame.copy()
    if colors is None:
        colors = PALETTE

    # Get top-K indices
    top_indices = np.argsort(probabilities)[::-1][:top_k]

    y_start = 60
    bar_height = 30
    max_bar_width = 250

    for rank, idx in enumerate(top_indices):
        prob = probabilities[idx]
        label = actions[idx] if idx < len(actions) else f"class_{idx}"
        color = colors[rank % len(colors)]

        y = y_start + rank * (bar_height + 5)
        bar_width = int(prob * max_bar_width)

        # Background bar
        cv2.rectangle(output, (0, y), (max_bar_width, y + bar_height), (50, 50, 50), -1)
        # Probability bar
        cv2.rectangle(output, (0, y), (bar_width, y + bar_height), color, -1)
        # Label
        cv2.putText(
            output, f"{label} ({prob:.1%})", (5, y + bar_height - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )

    return output
