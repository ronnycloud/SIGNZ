"""
SignVision - Action Landmark Data Collection Tool
=================================================
Interactive webcam collection script for sign language action recognition.
Supports recording sequences of MediaPipe keypoints for 100 ASL words across
10 categories with 5 video sequences per word (30 frames each).

Dual Storage:
  - MP_Data/<action>/<sequence>/<frame_num>.npy (1662-dim landmark arrays)
  - dataset/MP_Data/<action>/<sequence>.npy (30, 258-dim sequence arrays)

Controls during collection:
  - 'q' : Save progress and quit
  - 's' : Skip current word
  - 'r' : Redo current video sequence
  - 'p' / SPACE : Pause / resume collection
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np
import mediapipe as mp

from utils import mediapipe_detection, draw_styled_landmarks, extract_keypoints as extract_keypoints_1662
from src.features.extractor import extract_keypoints as extract_keypoints_258
from src.data.vocabulary import (
    VOCABULARY_CATEGORIES,
    ACTION_IDS,
    ACTION_MAP,
    get_display_name,
    get_category,
    normalize_action_id,
)

# Default Storage Paths
LEGACY_DATA_PATH = os.path.join("MP_Data")
MODERN_DATA_PATH = os.path.join("dataset", "MP_Data")

# Actions: all 100 curated sign words across 10 categories
actions = np.array(ACTION_IDS)

# 5 videos per word as specified
no_sequences = 5

# 30 frames per video sequence
sequence_length = 30


def create_folders(action_list, num_seqs=no_sequences, legacy_path=LEGACY_DATA_PATH, modern_path=MODERN_DATA_PATH):
    """Ensure destination directories exist for both legacy and modern formats."""
    for action in action_list:
        # Legacy frame-by-frame folders
        for sequence in range(num_seqs):
            os.makedirs(os.path.join(legacy_path, action, str(sequence)), exist_ok=True)
        # Modern single-file sequence folder
        os.makedirs(os.path.join(modern_path, action), exist_ok=True)


def is_sequence_complete(action, sequence_idx, seq_len=sequence_length, legacy_path=LEGACY_DATA_PATH, modern_path=MODERN_DATA_PATH):
    """Check if a sequence was already fully recorded."""
    legacy_complete = False
    legacy_folder = os.path.join(legacy_path, action, str(sequence_idx))
    if os.path.isdir(legacy_folder):
        files = [f for f in os.listdir(legacy_folder) if f.endswith(".npy")]
        if len(files) >= seq_len:
            legacy_complete = True

    modern_complete = os.path.exists(os.path.join(modern_path, action, f"{sequence_idx}.npy"))
    return legacy_complete or modern_complete


def draw_ui_overlay(
    image,
    action,
    category,
    sequence_idx,
    total_seqs,
    action_idx,
    total_actions,
    frame_num=None,
    total_frames=sequence_length,
    countdown=None,
    is_paused=False,
):
    """Draws a clean, high-contrast overlay on the OpenCV feed."""
    h, w = image.shape[:2]

    # Top dark glass bar (height 70px)
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, 75), (15, 23, 42), -1)
    # Bottom status bar (height 40px)
    cv2.rectangle(overlay, (0, h - 45), (w, h), (15, 23, 42), -1)
    cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)

    display_name = get_display_name(action)

    # Word Title & Category Badge
    cv2.putText(
        image,
        f"{display_name.upper()}",
        (20, 36),
        cv2.FONT_HERSHEY_DUPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f"[{category}]",
        (20, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (56, 189, 248),  # Sky blue
        1,
        cv2.LINE_AA,
    )

    # Video progress & Global progress (top right)
    vid_text = f"Video: {sequence_idx + 1}/{total_seqs}"
    cv2.putText(
        image,
        vid_text,
        (w - 220, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (16, 185, 129),  # Emerald green
        2,
        cv2.LINE_AA,
    )
    pct = (action_idx / max(total_actions, 1)) * 100
    prog_text = f"Sign {action_idx + 1}/{total_actions} ({pct:.0f}%)"
    cv2.putText(
        image,
        prog_text,
        (w - 220, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (203, 213, 225),
        1,
        cv2.LINE_AA,
    )

    # Bottom Instructions Bar
    ctrl_text = "Controls: 'q'=Quit | 's'=Skip | 'r'=Redo Video | 'p'=Pause"
    cv2.putText(
        image,
        ctrl_text,
        (20, h - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (148, 163, 184),
        1,
        cv2.LINE_AA,
    )

    # Countdown Overlay
    if countdown is not None:
        cd_bg = image.copy()
        cv2.rectangle(cd_bg, (w // 2 - 200, h // 2 - 70), (w // 2 + 200, h // 2 + 70), (15, 23, 42), -1)
        cv2.addWeighted(cd_bg, 0.85, image, 0.15, 0, image)

        if countdown > 0:
            cd_text = f"GET READY: {countdown}"
            cd_color = (245, 158, 11)  # Amber
        else:
            cd_text = "RECORDING NOW!"
            cd_color = (16, 185, 129)  # Green

        text_size = cv2.getTextSize(cd_text, cv2.FONT_HERSHEY_DUPLEX, 1.1, 2)[0]
        tx = (w - text_size[0]) // 2
        cv2.putText(image, cd_text, (tx, h // 2 + 10), cv2.FONT_HERSHEY_DUPLEX, 1.1, cd_color, 2, cv2.LINE_AA)

    # Frame Progress Bar during recording
    elif frame_num is not None:
        bar_w = int((w - 40) * (frame_num + 1) / total_frames)
        cv2.rectangle(image, (20, h - 55), (w - 20, h - 47), (51, 65, 85), -1)
        cv2.rectangle(image, (20, h - 55), (20 + bar_w, h - 47), (16, 185, 129), -1)
        frame_str = f"Recording: Frame {frame_num + 1}/{total_frames}"
        cv2.putText(
            image,
            frame_str,
            (w - 240, h - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (16, 185, 129),
            1,
            cv2.LINE_AA,
        )

    # Paused Banner
    if is_paused:
        pause_box = image.copy()
        cv2.rectangle(pause_box, (w // 2 - 150, h // 2 - 40), (w // 2 + 150, h // 2 + 40), (30, 41, 59), -1)
        cv2.addWeighted(pause_box, 0.9, image, 0.1, 0, image)
        cv2.putText(
            image,
            "PAUSED - Press 'p' to Resume",
            (w // 2 - 140, h // 2 + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def collect_data(
    action_list=None,
    num_seqs=no_sequences,
    seq_len=sequence_length,
    camera_idx=0,
    resume=True,
    start_from=None,
    legacy_path=LEGACY_DATA_PATH,
    modern_path=MODERN_DATA_PATH,
):
    """Main collection loop with MediaPipe Holistic."""
    action_list = list(action_list or actions)

    # Filter starting position if requested
    if start_from:
        start_id = normalize_action_id(start_from)
        if start_id in action_list:
            idx = action_list.index(start_id)
            action_list = action_list[idx:]
            print(f"[Collect] Resuming from action: '{start_id}' ({len(action_list)} remaining)")

    create_folders(action_list, num_seqs, legacy_path, modern_path)

    # Open Camera with Windows DirectShow fallback
    cap = None
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(camera_idx)

    if not cap.isOpened():
        print(f"[Error] Could not open camera {camera_idx}. Check your webcam connection.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    mp_holistic = mp.solutions.holistic

    total_actions = len(action_list)
    print("=" * 60)
    print(f"SignVision Data Collector - {total_actions} Words × {num_seqs} Videos × {seq_len} Frames")
    print(f"Legacy Output Path : {legacy_path}")
    print(f"Modern Output Path : {modern_path}")
    print("Controls: 'q'=Quit | 's'=Skip Word | 'r'=Redo Video | 'p'=Pause")
    print("=" * 60)

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
    ) as holistic:
        for action_idx, action in enumerate(action_list):
            category = get_category(action)
            display_name = get_display_name(action)

            seq_idx = 0
            while seq_idx < num_seqs:
                # Check if already collected and resume enabled
                if resume and is_sequence_complete(action, seq_idx, seq_len, legacy_path, modern_path):
                    print(f"  [Skip] {action} (video {seq_idx + 1}/{num_seqs}) already exists.")
                    seq_idx += 1
                    continue

                # 1. Countdown Phase (Give user 2.5s to prepare hand pose)
                start_countdown = time.time()
                skip_action = False
                quit_program = False

                while time.time() - start_countdown < 2.5:
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    elapsed = time.time() - start_countdown
                    countdown_val = max(1, 3 - int(elapsed))

                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)

                    draw_ui_overlay(
                        image=image,
                        action=action,
                        category=category,
                        sequence_idx=seq_idx,
                        total_seqs=num_seqs,
                        action_idx=action_idx,
                        total_actions=total_actions,
                        countdown=countdown_val,
                    )
                    cv2.imshow("SignVision - Data Collection", image)

                    key = cv2.waitKey(10) & 0xFF
                    if key == ord("q"):
                        quit_program = True
                        break
                    elif key == ord("s"):
                        skip_action = True
                        break
                    elif key == ord("p") or key == 32:  # Pause
                        paused = True
                        while paused:
                            p_ret, p_frame = cap.read()
                            if p_ret:
                                draw_ui_overlay(
                                    p_frame, action, category, seq_idx, num_seqs,
                                    action_idx, total_actions, is_paused=True
                                )
                                cv2.imshow("SignVision - Data Collection", p_frame)
                            p_key = cv2.waitKey(30) & 0xFF
                            if p_key in (ord("p"), 32):
                                paused = False
                            elif p_key == ord("q"):
                                quit_program = True
                                paused = False

                if quit_program:
                    print("\n[Collect] Collection terminated by user.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return

                if skip_action:
                    print(f"\n[Collect] Skipping action: '{action}'")
                    break

                # 2. Frame Recording Phase (30 frames)
                frame_window_1662 = []
                frame_window_258 = []
                redo_video = False

                for frame_num in range(seq_len):
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)

                    # Extract both feature representations
                    kp_1662 = extract_keypoints_1662(results)
                    kp_258 = extract_keypoints_258(results)

                    frame_window_1662.append(kp_1662)
                    frame_window_258.append(kp_258)

                    # Save legacy frame-by-frame
                    npy_path = os.path.join(legacy_path, action, str(seq_idx), str(frame_num))
                    np.save(npy_path, kp_1662)

                    draw_ui_overlay(
                        image=image,
                        action=action,
                        category=category,
                        sequence_idx=seq_idx,
                        total_seqs=num_seqs,
                        action_idx=action_idx,
                        total_actions=total_actions,
                        frame_num=frame_num,
                        total_frames=seq_len,
                    )
                    cv2.imshow("SignVision - Data Collection", image)

                    key = cv2.waitKey(10) & 0xFF
                    if key == ord("q"):
                        quit_program = True
                        break
                    elif key == ord("r"):
                        print(f"  [Redo] Redoing video {seq_idx + 1} for {action}...")
                        redo_video = True
                        break
                    elif key == ord("s"):
                        skip_action = True
                        break

                if quit_program:
                    print("\n[Collect] Collection terminated by user.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return

                if skip_action:
                    break

                if redo_video:
                    continue

                # 3. Save Modern Sequence File (shape: 30, 258)
                if len(frame_window_258) == seq_len:
                    modern_file = os.path.join(modern_path, action, f"{seq_idx}.npy")
                    np.save(modern_file, np.array(frame_window_258, dtype=np.float32))
                    print(f"  [Saved] {display_name} -> Video {seq_idx + 1}/{num_seqs} recorded ({seq_len} frames)")
                    seq_idx += 1
                else:
                    print(f"  [Warning] Incomplete video ({len(frame_window_258)}/{seq_len} frames). Redoing...")

    cap.release()
    cv2.destroyAllWindows()
    print("\n[Collect] Data collection finished successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SignVision 100-Word Data Collection Tool")
    parser.add_argument("--action", type=str, default=None, help="Record only a specific action (e.g. 'bathroom')")
    parser.add_argument("--category", type=str, default=None, help="Record only a specific category (e.g. 'Food', 'Home')")
    parser.add_argument("--start_from", type=str, default=None, help="Start recording from a specific action name")
    parser.add_argument("--num_videos", type=int, default=no_sequences, help="Number of videos per action (default: 5)")
    parser.add_argument("--sequence_length", type=int, default=sequence_length, help="Frames per video (default: 30)")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing recorded sequences")
    args = parser.parse_args()

    # Determine actions to record
    target_actions = list(ACTION_IDS)
    if args.action:
        norm = normalize_action_id(args.action)
        if norm in ACTION_IDS:
            target_actions = [norm]
        else:
            print(f"[Error] Unknown action '{args.action}'. Available actions: {len(ACTION_IDS)}")
            sys.exit(1)
    elif args.category:
        cat_match = None
        for cat in VOCABULARY_CATEGORIES:
            if cat.lower() == args.category.lower():
                cat_match = cat
                break
        if cat_match:
            target_actions = [item["id"] for item in VOCABULARY_CATEGORIES[cat_match]]
            print(f"[Collect] Category '{cat_match}': {len(target_actions)} words selected.")
        else:
            print(f"[Error] Unknown category '{args.category}'. Available: {list(VOCABULARY_CATEGORIES.keys())}")
            sys.exit(1)

    collect_data(
        action_list=target_actions,
        num_seqs=args.num_videos,
        seq_len=args.sequence_length,
        camera_idx=args.camera,
        resume=not args.overwrite,
        start_from=args.start_from,
    )
