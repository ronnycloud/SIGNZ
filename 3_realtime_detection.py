"""
SignVision - Real-Time Action Detection & Subtitle System
=========================================================
Runs real-time webcam inference using the trained Keras LSTM model (action.h5)
with MediaPipe Holistic landmarks, top-5 confidence display, and real-time
NLP grammar correction for natural English subtitles.
"""

import os
import sys
import json
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

from utils import mediapipe_detection, draw_styled_landmarks, extract_keypoints
from src.data.vocabulary import ACTION_IDS, get_display_name
from src.nlp.grammar_corrector import GrammarCorrector


def load_actions(labels_path="action_labels.json"):
    """Load action labels saved during training or fallback to standard vocabulary."""
    if os.path.exists(labels_path):
        try:
            with open(labels_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, list) and len(loaded) > 0:
                    return np.array(loaded)
        except Exception:
            pass
    return np.array(ACTION_IDS)


def get_dynamic_colors(num_classes):
    """Generate distinct visually appealing BGR colors for visualization."""
    np.random.seed(42)
    colors = []
    for _ in range(num_classes):
        colors.append((
            int(np.random.randint(50, 240)),
            int(np.random.randint(80, 255)),
            int(np.random.randint(50, 240)),
        ))
    return colors


def prob_viz_top5(res, actions, input_frame, top_k=5):
    """Draw confidence bars for top-k predicted classes in an overlay."""
    output_frame = input_frame.copy()
    top_indices = np.argsort(res)[::-1][:top_k]

    h, w = output_frame.shape[:2]
    # Transparent dark side panel for top predictions
    panel = output_frame.copy()
    cv2.rectangle(panel, (w - 230, 50), (w - 10, 50 + top_k * 36 + 10), (15, 23, 42), -1)
    cv2.addWeighted(panel, 0.75, output_frame, 0.25, 0, output_frame)

    for i, idx in enumerate(top_indices):
        prob = float(res[idx])
        action_name = get_display_name(actions[idx]) if idx < len(actions) else "Unknown"
        bar_len = int(prob * 140)
        y = 75 + i * 36

        # Probability bar
        bar_color = (16, 185, 129) if i == 0 and prob > 0.4 else (148, 163, 184)
        cv2.rectangle(output_frame, (w - 220, y - 14), (w - 220 + bar_len, y - 2), bar_color, -1)

        # Label and percentage
        text = f"{action_name[:12]}: {prob:.0%}"
        cv2.putText(
            output_frame,
            text,
            (w - 220, y + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return output_frame


def run_realtime(model_path="action.h5", threshold=0.5):
    """Main real-time detection loop."""
    if not os.path.exists(model_path):
        print(f"[Error] Model '{model_path}' not found. Please train first with 2_train_model.py or train.py.")
        return

    actions = load_actions()
    print(f"[Inference] Loaded {len(actions)} classes. Loading model '{model_path}'...")
    model = load_model(model_path)
    grammar = GrammarCorrector()

    sequence = []
    sentence_signs = []
    formatted_sentence = ""
    predictions = []

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

    print("=" * 60)
    print("SignVision Real-Time Recognition Started")
    print("Controls: 'q' = Quit | 'r' = Clear sentence")
    print("=" * 60)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)

            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                best_idx = int(np.argmax(res))
                best_prob = float(res[best_idx])
                predictions.append(best_idx)

                # Temporal consistency check (at least 8 out of last 10 frames agree)
                if len(predictions) > 10:
                    predictions = predictions[-10:]

                if predictions.count(best_idx) >= 8 and best_prob > threshold:
                    sign_label = actions[best_idx]
                    if not sentence_signs or sentence_signs[-1] != sign_label:
                        sentence_signs.append(sign_label)
                        if len(sentence_signs) > 6:
                            sentence_signs = sentence_signs[-6:]
                        formatted_sentence = grammar.correct(sentence_signs)

                image = prob_viz_top5(res, actions, image)

            # Subtitle Banner (bottom or top)
            h, w = image.shape[:2]
            cv2.rectangle(image, (0, 0), (w, 45), (15, 23, 42), -1)
            display_text = formatted_sentence if formatted_sentence else "Listening for signs..."
            cv2.putText(
                image,
                display_text,
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("SignVision - Real-Time Recognition", image)

            key = cv2.waitKey(10) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                sentence_signs = []
                formatted_sentence = ""

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_realtime()
