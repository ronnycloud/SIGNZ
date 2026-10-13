"""
Feature Extractor
==================
Handles MediaPipe Holistic landmark extraction from video frames.

Provides two interfaces:
    1. FeatureExtractor class — used by app.py and the real-time Flask pipeline.
       Returns per-modality arrays (left_hand, right_hand, pose) plus presence flags.
    2. extract_keypoints() function — used by the WLASL data pipeline for simpler
       batch processing. Returns a single flattened 258-dim vector.
"""

import cv2
import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


class FeatureExtractor:
    """MediaPipe Holistic wrapper used by the real-time Flask pipeline.

    Usage (context manager):
        with FeatureExtractor() as extractor:
            results = extractor.process_frame(frame)
    """

    # Upper-body pose landmark indices (0..24 out of 33 total).
    # We keep only 25 upper-body landmarks for pose — legs add noise.
    UPPER_BODY_INDICES = list(range(25))

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
    ):
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_complexity = model_complexity
        self._holistic = None

    # -- Context manager support ------------------------------------------
    def __enter__(self):
        self._holistic = mp_holistic.Holistic(
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            model_complexity=self.model_complexity,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._holistic is not None:
            self._holistic.close()
            self._holistic = None
        return False

    # -- Core processing ---------------------------------------------------
    def process_frame(self, frame: np.ndarray) -> dict:
        """Run MediaPipe on a BGR frame and return separated landmark arrays.

        Returns
        -------
        dict with keys:
            left_hand  : np.ndarray (63,)   21 landmarks × 3 coords
            right_hand : np.ndarray (63,)
            pose       : np.ndarray (100,)  25 landmarks × 4 (x,y,z,visibility)
            left_present  : bool
            right_present : bool
            _raw_results  : mediapipe results object (for draw_landmarks)
        """
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self._holistic.process(image)

        # Left hand (21 × 3 = 63)
        if results.left_hand_landmarks:
            lh = np.array(
                [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark],
                dtype=np.float32,
            ).flatten()
            left_present = True
        else:
            lh = np.zeros(21 * 3, dtype=np.float32)
            left_present = False

        # Right hand (21 × 3 = 63)
        if results.right_hand_landmarks:
            rh = np.array(
                [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark],
                dtype=np.float32,
            ).flatten()
            right_present = True
        else:
            rh = np.zeros(21 * 3, dtype=np.float32)
            right_present = False

        # Pose — upper body only (25 × 4 = 100)
        if results.pose_landmarks:
            all_pose = results.pose_landmarks.landmark
            pose = np.array(
                [[all_pose[i].x, all_pose[i].y, all_pose[i].z, all_pose[i].visibility]
                 for i in self.UPPER_BODY_INDICES],
                dtype=np.float32,
            ).flatten()
        else:
            pose = np.zeros(25 * 4, dtype=np.float32)

        return {
            "left_hand": lh,
            "right_hand": rh,
            "pose": pose,
            "left_present": left_present,
            "right_present": right_present,
            "_raw_results": results,
        }

    # -- Visualization -----------------------------------------------------
    def draw_landmarks(self, frame: np.ndarray, results_dict: dict) -> np.ndarray:
        """Draw MediaPipe landmarks on the frame for visualization."""
        results = results_dict["_raw_results"]
        annotated = frame.copy()

        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                mp_drawing_styles.get_default_pose_landmarks_style(),
            )

        return annotated


# ---------------------------------------------------------------------------
# Standalone extraction function used by the WLASL data pipeline
# ---------------------------------------------------------------------------

def extract_keypoints(results) -> np.ndarray:
    """Extract a flat 258-dim vector from raw MediaPipe Holistic results.

    Layout: pose(33×4=132) + left_hand(21×3=63) + right_hand(21×3=63) = 258

    This is a simpler interface used during batch dataset processing
    (src/data/wlasl_pipeline.py). Face landmarks are dropped entirely.
    """
    # Pose: all 33 landmarks × (x, y, z, visibility)
    if results.pose_landmarks:
        pose = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark],
            dtype=np.float32,
        ).flatten()
    else:
        pose = np.zeros(33 * 4, dtype=np.float32)

    # Left hand (21 × 3)
    if results.left_hand_landmarks:
        lh = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark],
            dtype=np.float32,
        ).flatten()
    else:
        lh = np.zeros(21 * 3, dtype=np.float32)

    # Right hand (21 × 3)
    if results.right_hand_landmarks:
        rh = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark],
            dtype=np.float32,
        ).flatten()
    else:
        rh = np.zeros(21 * 3, dtype=np.float32)

    # Total: 132 + 63 + 63 = 258
    return np.concatenate([pose, lh, rh])
