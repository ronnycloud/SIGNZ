"""
SignVision - Real-Time Action Detection & Subtitle System
=========================================================
Runs real-time webcam inference using the trained PyTorch MTAN-SLR model
(checkpoints/best_model.pt) with MediaPipe Holistic landmarks, top-5 confidence
display, and real-time NLP grammar correction for natural English subtitles.
"""

import os
import sys
import json
import time
import cv2
import numpy as np
import mediapipe as mp
import torch
import yaml

from utils import mediapipe_detection, draw_styled_landmarks
from src.features.normalizer import LandmarkNormalizer
from src.features.velocity import VelocityComputer
from src.models import build_model_from_config
from src.data.vocabulary import ACTION_IDS, get_display_name
from src.nlp.grammar_corrector import GrammarCorrector


def load_model_and_classes(model_path="checkpoints/best_model.pt", config_path="configs/model_config.yaml"):
    """Loads the trained PyTorch model and action class list."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint '{model_path}' not found. Please run 'python train.py' first.")

    with open(config_path, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    checkpoint = torch.load(model_path, map_location=device)
    actions = checkpoint.get("actions", checkpoint.get("classes", list(ACTION_IDS)))
    active_model = checkpoint.get("active_model", model_cfg.get("active_model", "bilstm_attention"))
    model_cfg["active_model"] = active_model

    model = build_model_from_config(model_cfg, num_classes=len(actions))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, actions, device


def prob_viz_top5(probs, actions, input_frame, top_k=5):
    """Draw confidence bars for top-k predicted classes in an overlay."""
    output_frame = input_frame.copy()
    top_indices = np.argsort(probs)[::-1][:top_k]

    h, w = output_frame.shape[:2]
    panel = output_frame.copy()
    cv2.rectangle(panel, (w - 240, 50), (w - 10, 50 + top_k * 36 + 12), (15, 23, 42), -1)
    cv2.addWeighted(panel, 0.75, output_frame, 0.25, 0, output_frame)

    for i, idx in enumerate(top_indices):
        prob = float(probs[idx])
        action_name = get_display_name(actions[idx]) if idx < len(actions) else "Unknown"
        bar_len = int(prob * 150)
        y = 75 + i * 36

        # Probability bar
        bar_color = (16, 185, 129) if i == 0 and prob > 0.35 else (148, 163, 184)
        cv2.rectangle(output_frame, (w - 230, y - 14), (w - 230 + bar_len, y - 2), bar_color, -1)

        # Label and percentage
        text = f"{action_name[:12]}: {prob:.0%}"
        cv2.putText(
            output_frame,
            text,
            (w - 230, y + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return output_frame


def run_realtime(model_path="checkpoints/best_model.pt", threshold=0.35):
    """Main real-time detection loop."""
    print("=" * 60)
    print("SignVision Real-Time Recognition (100 ASL Vocabulary)")
    print(f"Loading model from '{model_path}'...")

    model, actions, device = load_model_and_classes(model_path)
    print(f"Model loaded successfully on {device} with {len(actions)} classes.")

    normalizer = LandmarkNormalizer()
    velocity_computer = VelocityComputer()
    grammar = GrammarCorrector()

    buffer = []
    sentence_signs = []
    formatted_sentence = ""
    predictions_history = []
    max_buf_len = 60
    last_sign_time = time.time()
    sentence_finalized = False

    cap = None
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[Error] Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    mp_holistic = mp.solutions.holistic

    print("Controls: 'q' = Quit | 'r' = Clear sentence")
    print("=" * 60)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            now = time.time()
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)

            # Check hands presence
            left_present = results.left_hand_landmarks is not None
            right_present = results.right_hand_landmarks is not None
            hands_present = left_present or right_present

            # Extract normalized features
            lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark], dtype=np.float32).flatten() if left_present else np.zeros(63, dtype=np.float32)
            rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark], dtype=np.float32).flatten() if right_present else np.zeros(63, dtype=np.float32)
            if results.pose_landmarks:
                pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark[:25]], dtype=np.float32).flatten()
            else:
                pose = np.zeros(100, dtype=np.float32)

            norm_frame = normalizer.normalize_frame(lh, rh, pose)
            flags = np.array([float(left_present), float(right_present)], dtype=np.float32)
            feat_454 = velocity_computer.build_full_feature(norm_frame, flags)

            buffer.append(feat_454)
            if len(buffer) > max_buf_len:
                buffer.pop(0)

            # Run inference if hands are active and buffer is ready
            if hands_present and len(buffer) >= 10:
                seq_tensor = torch.tensor(np.array(buffer), dtype=torch.float32).unsqueeze(0).to(device)
                mask = torch.ones(1, len(buffer), dtype=torch.bool, device=device)

                with torch.no_grad():
                    logits = model(seq_tensor, mask=mask)
                    probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

                best_idx = int(np.argmax(probs))
                best_prob = float(probs[best_idx])
                predictions_history.append(best_idx)

                if len(predictions_history) > 10:
                    predictions_history = predictions_history[-10:]

                # Require consensus across recent frames
                from collections import Counter
                mode_idx, mode_count = Counter(predictions_history[-5:]).most_common(1)[0]
                if mode_idx == best_idx and mode_count >= 3 and best_prob > threshold:
                    sign_label = actions[best_idx]
                    if sentence_finalized:
                        sentence_signs = []
                        sentence_finalized = False

                    if not sentence_signs or sentence_signs[-1] != sign_label:
                        sentence_signs.append(sign_label)
                        last_sign_time = now
                        if len(sentence_signs) > 8:
                            sentence_signs = sentence_signs[-8:]
                        formatted_sentence = grammar.correct(sentence_signs)

                image = prob_viz_top5(probs, actions, image)

            # Auto-finalize sentence when hands are lowered or paused for > 1.3s
            if sentence_signs and not sentence_finalized:
                pause_limit = 1.0 if not hands_present else 1.5
                if (now - last_sign_time) >= pause_limit:
                    sentence_finalized = True
                    formatted_sentence = grammar.correct(sentence_signs)

            # Draw Subtitle Banner on top of frame
            h, w = image.shape[:2]
            overlay = image.copy()
            cv2.rectangle(overlay, (0, 0), (w, 48), (15, 23, 42), -1)
            cv2.addWeighted(overlay, 0.85, image, 0.15, 0, image)

            display_text = formatted_sentence if formatted_sentence else "Listening for signs..."
            text_color = (16, 185, 129) if sentence_finalized else (255, 255, 255)
            cv2.putText(image, display_text, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2, cv2.LINE_AA)

            cv2.imshow("SignVision - Real-Time Recognition", image)

            key = cv2.waitKey(10) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                sentence_signs = []
                formatted_sentence = ""
                sentence_finalized = False
                buffer = []

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_realtime()
