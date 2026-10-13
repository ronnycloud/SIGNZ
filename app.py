"""
SignVision - Sign Language Recognition System
===============================================
Flask Web Application Backend
Integrates OpenCV camera capture, MediaPipe landmark extraction,
MTAN-SLR model prediction pipeline, MJPEG video streaming, and
Server-Sent Events (SSE) for real-time subtitle & FPS updates.
"""

import os
import sys
import time
import json
import cv2
import numpy as np
import torch
import yaml
from typing import Generator
from flask import Flask, render_template, Response, jsonify

from src.features.extractor import FeatureExtractor
from src.features.normalizer import LandmarkNormalizer
from src.features.velocity import VelocityComputer
from src.models import build_model_from_config
from src.nlp.sign_to_sentence import SignToSentenceTranslator
from src.utils.logger import get_logger

logger = get_logger("SignVision_App")

app = Flask(__name__)

# Global Pipeline State
CAMERA = None
PREDICTION_STATE = {
    "subtitle": "Listening for sign gestures...",
    "fps": 30.0,
    "status": "Listening...",
    "is_active": True,
}


def load_model_and_config():
    model_cfg_path = "configs/model_config.yaml"
    ckpt_path = "checkpoints/best_model.pt"

    with open(model_cfg_path, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actions = ["hello", "thanks", "iloveyou"]
    active_model = model_cfg.get("active_model", "bilstm_attention")

    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        actions = checkpoint.get("actions", actions)
        active_model = checkpoint.get("active_model", active_model)
        model_cfg["active_model"] = active_model

        model = build_model_from_config(model_cfg, num_classes=len(actions))
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded checkpoint '{ckpt_path}' for model variant '{active_model}'.")
    else:
        logger.warning(f"No checkpoint found at '{ckpt_path}'. Initializing unweighted model.")
        model = build_model_from_config(model_cfg, num_classes=len(actions))

    model.to(device)
    model.eval()

    return model, actions, device


def get_camera_capture():
    """Try opening camera device across multiple backends and indices."""
    for idx in [0, 1, 2]:
        # On Windows, try CAP_DSHOW first
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                return cap
            cap.release()

        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            return cap
        cap.release()
    return None


def generate_frames() -> Generator[bytes, None, None]:
    """MJPEG frame generator reading OpenCV camera and processing MediaPipe pipeline."""
    global PREDICTION_STATE

    model, actions, device = load_model_and_config()
    translator = SignToSentenceTranslator()

    extractor = FeatureExtractor()
    normalizer = LandmarkNormalizer()
    velocity_computer = VelocityComputer()

    cap = get_camera_capture()

    buffer = []
    sentence_signs = []
    predictions_history = []
    confidence_threshold = 0.15
    max_buf_len = 90
    frame_counter = 0

    prev_time = time.time()

    if cap is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        logger.info("Hardware camera opened successfully.")

    with extractor:
        while True:
            frame = None
            if cap is not None and cap.isOpened():
                success, frame = cap.read()
                if not success or frame is None:
                    frame = None

            # Fallback Synthetic Demo Stream if hardware webcam is unavailable
            if frame is None:
                frame_counter += 1
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                # Background gradient
                cv2.rectangle(frame, (0, 0), (640, 480), (15, 23, 42), -1)

                # Draw animated simulated hands and grid
                t = frame_counter * 0.05
                x_center = int(320 + 80 * np.sin(t))
                y_center = int(240 + 40 * np.cos(t))

                # Draw simulated skeleton joints
                cv2.circle(frame, (x_center, y_center), 12, (16, 185, 129), -1)
                cv2.line(frame, (320, 380), (x_center, y_center), (226, 232, 240), 3)

                for offset_x, offset_y in [(-30, -50), (-10, -70), (10, -75), (30, -60), (45, -40)]:
                    finger_x = int(x_center + offset_x + 10 * np.sin(t * 2))
                    finger_y = int(y_center + offset_y + 10 * np.cos(t * 2))
                    cv2.line(frame, (x_center, y_center), (finger_x, finger_y), (148, 163, 184), 2)
                    cv2.circle(frame, (finger_x, finger_y), 5, (245, 117, 16), -1)

                cv2.putText(
                    frame, "CAMERA FEED (SIMULATION MODE)", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (148, 163, 184), 1, cv2.LINE_AA
                )
                cv2.putText(
                    frame, "Connect webcam or grant permissions for live feed", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 116, 139), 1, cv2.LINE_AA
                )

                # Simulated gesture prediction cycle
                cycle_sec = int(frame_counter / 30) % 3
                simulated_sign = actions[cycle_sec]
                if not sentence_signs or sentence_signs[-1] != simulated_sign:
                    sentence_signs.append(simulated_sign)
                    if len(sentence_signs) > 5:
                        sentence_signs = sentence_signs[-5:]
                    PREDICTION_STATE["subtitle"] = translator(sentence_signs)

            else:
                # Hardware camera processing
                results = extractor.process_frame(frame)
                norm_frame = normalizer.normalize_frame(
                    results["left_hand"],
                    results["right_hand"],
                    results["pose"],
                )

                hand_flags = np.array(
                    [float(results["left_present"]), float(results["right_present"])],
                    dtype=np.float32,
                )
                feat_454 = velocity_computer.build_full_feature(norm_frame, hand_flags)
                buffer.append(feat_454)
                if len(buffer) > max_buf_len:
                    buffer.pop(0)

                frame = extractor.draw_landmarks(frame, results)

                if len(buffer) >= 10:
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

                    # Show current prediction and confidence on the video frame
                    current_label = actions[best_idx] if best_idx < len(actions) else "?"
                    debug_text = f"Pred: {current_label} ({best_prob:.1%})"
                    cv2.putText(
                        frame, debug_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA
                    )
                    PREDICTION_STATE["status"] = debug_text

                    # Stability check: most common prediction in last 5 must match current
                    recent = predictions_history[-5:]
                    if len(recent) >= 3:
                        from collections import Counter
                        mode_idx, mode_count = Counter(recent).most_common(1)[0]
                        if mode_idx == best_idx and mode_count >= 3 and best_prob > confidence_threshold:
                            sign_label = actions[best_idx]
                            if not sentence_signs or sentence_signs[-1] != sign_label:
                                sentence_signs.append(sign_label)
                                if len(sentence_signs) > 6:
                                    sentence_signs = sentence_signs[-6:]

                                translated_text = translator(sentence_signs)
                                PREDICTION_STATE["subtitle"] = translated_text

            # FPS calculation
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time
            PREDICTION_STATE["fps"] = round(fps, 1)

            # Encode frame as JPEG
            ret, buffer_jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ret:
                continue

            frame_bytes = buffer_jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()


@app.route('/')
def index():
    """Render main SignVision dashboard."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route returning MJPEG multipart response."""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/api/status')
def api_status():
    """JSON endpoint for status and active subtitle string."""
    return jsonify(PREDICTION_STATE)


@app.route('/api/stream')
def api_stream():
    """Server-Sent Events (SSE) stream for low-latency subtitle updates."""
    def event_stream():
        while True:
            data = json.dumps(PREDICTION_STATE)
            yield f"data: {data}\n\n"
            time.sleep(0.1)

    return Response(event_stream(), mimetype="text/event-stream")


if __name__ == '__main__':
    logger.info("Starting SignVision Flask Web Application on http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
