"""
SignVision - Standalone Action Landmark Data Collection Tool
=============================================================
100% STANDALONE script. Can be copied to any machine or directory alone!
Zero dependencies on internal project files. Only requires:
    pip install opencv-python mediapipe numpy

Features:
  - 100 ASL words across 10 categories
  - Default: 5 video sequences per word, 30 frames per video
  - Dual storage:
      MP_Data/<action>/<sequence>/<frame_num>.npy (1662-dim arrays)
      dataset/MP_Data/<action>/<sequence>.npy (30, 258-dim arrays)
  - Interactive OpenCV GUI with 3-2-1 countdown, live progress bar
  - Resumable: automatically skips already recorded sequences
  - Keyboard Controls:
      'q' : Save progress and exit
      's' : Skip current word
      'r' : Redo current video sequence
      'p' / SPACE : Pause / resume

CLI Options:
  python 1_collect_data.py --action bathroom
  python 1_collect_data.py --category Food
  python 1_collect_data.py --start_from teacher
  python 1_collect_data.py --num_videos 5
  python 1_collect_data.py --camera 0
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np
import mediapipe as mp

# =============================================================================
# 1. 100-WORD VOCABULARY REGISTRY (EMBEDDED)
# =============================================================================

VOCABULARY_CATEGORIES = {
    "Common Signs": [
        {"id": "fine", "display": "Fine"},
        {"id": "finish", "display": "Finish"},
        {"id": "forget", "display": "Forget"},
        {"id": "go", "display": "Go"},
        {"id": "hello", "display": "Hello"},
        {"id": "help", "display": "Help"},
        {"id": "dont_want", "display": "Don't Want"},
        {"id": "like", "display": "Like"},
        {"id": "more", "display": "More"},
        {"id": "need", "display": "Need"},
        {"id": "wrong", "display": "Wrong"},
        {"id": "no", "display": "No"},
        {"id": "not", "display": "Not"},
        {"id": "please", "display": "Please"},
        {"id": "right", "display": "Right"},
        {"id": "thank_you", "display": "Thank-You"},
        {"id": "want", "display": "Want"},
        {"id": "yes", "display": "Yes"},
    ],
    "Home": [
        {"id": "bathroom", "display": "Bathroom"},
        {"id": "car", "display": "Car"},
        {"id": "home", "display": "Home"},
        {"id": "kitchen", "display": "Kitchen"},
        {"id": "living_room", "display": "Living Room"},
        {"id": "phone", "display": "Phone"},
    ],
    "Food": [
        {"id": "cheese", "display": "Cheese"},
        {"id": "cook", "display": "Cook"},
        {"id": "cookies", "display": "Cookies"},
        {"id": "drink", "display": "Drink"},
        {"id": "eat", "display": "Eat / Food"},
        {"id": "hamburger", "display": "Hamburger"},
        {"id": "milk", "display": "Milk"},
        {"id": "sandwich", "display": "Sandwich"},
    ],
    "People": [
        {"id": "me", "display": "I / Me"},
        {"id": "my", "display": "My / Mine"},
        {"id": "you", "display": "You"},
        {"id": "your", "display": "Your"},
        {"id": "boy", "display": "Boy"},
        {"id": "girl", "display": "Girl"},
        {"id": "man", "display": "Man"},
        {"id": "woman", "display": "Woman"},
        {"id": "deaf", "display": "Deaf"},
        {"id": "hearing", "display": "Hearing"},
    ],
    "Questions": [
        {"id": "how", "display": "How"},
        {"id": "what", "display": "What"},
        {"id": "when", "display": "When"},
        {"id": "where", "display": "Where"},
        {"id": "who", "display": "Who"},
        {"id": "why", "display": "Why"},
    ],
    "School": [
        {"id": "asl", "display": "ASL"},
        {"id": "book", "display": "Book"},
        {"id": "class", "display": "Class"},
        {"id": "know", "display": "Know"},
        {"id": "learn", "display": "Learn"},
        {"id": "sentence", "display": "Sentence"},
        {"id": "sign", "display": "Sign"},
        {"id": "student", "display": "Student"},
        {"id": "teacher", "display": "Teacher"},
    ],
    "Time": [
        {"id": "afternoon", "display": "Afternoon"},
        {"id": "day", "display": "Day"},
        {"id": "hour", "display": "Hour"},
        {"id": "minute", "display": "Minute"},
        {"id": "month", "display": "Month"},
        {"id": "morning", "display": "Morning"},
        {"id": "night", "display": "Night"},
        {"id": "noon", "display": "Noon"},
        {"id": "time", "display": "Time"},
        {"id": "week", "display": "Week"},
        {"id": "year", "display": "Year"},
    ],
    "Family": [
        {"id": "baby", "display": "Baby"},
        {"id": "brother", "display": "Brother"},
        {"id": "children", "display": "Children"},
        {"id": "family", "display": "Family"},
        {"id": "father", "display": "Father"},
        {"id": "grandfather", "display": "Grandfather"},
        {"id": "grandmother", "display": "Grandmother"},
        {"id": "husband", "display": "Husband"},
        {"id": "mother", "display": "Mother"},
        {"id": "sister", "display": "Sister"},
        {"id": "wife", "display": "Wife"},
    ],
    "Colors & Animals": [
        {"id": "dog", "display": "Dog"},
        {"id": "cat", "display": "Cat"},
        {"id": "blue", "display": "Blue"},
        {"id": "green", "display": "Green"},
        {"id": "orange", "display": "Orange"},
        {"id": "pink", "display": "Pink"},
        {"id": "purple", "display": "Purple"},
        {"id": "red", "display": "Red"},
        {"id": "yellow", "display": "Yellow"},
    ],
    "Descriptions & Conversational": [
        {"id": "so_so", "display": "So-So"},
        {"id": "good", "display": "Good"},
        {"id": "happy", "display": "Happy"},
        {"id": "sad", "display": "Sad"},
        {"id": "bad", "display": "Bad"},
        {"id": "chat", "display": "Chat"},
        {"id": "signing", "display": "Signing"},
        {"id": "look", "display": "Look / See / Watch"},
        {"id": "pay_attention", "display": "Pay Attention"},
        {"id": "dont", "display": "Not / Don't / Doesn't"},
        {"id": "not_yet", "display": "Not Yet"},
        {"id": "bye", "display": "Bye"},
    ],
}

ACTION_IDS = [
    item["id"]
    for cat_list in VOCABULARY_CATEGORIES.values()
    for item in cat_list
]

ACTION_MAP = {}
for category, items in VOCABULARY_CATEGORIES.items():
    for item in items:
        ACTION_MAP[item["id"]] = {
            "id": item["id"],
            "display": item["display"],
            "category": category,
        }

GLOSS_ALIASES = {
    "eat_food": "eat",
    "food": "eat",
    "i_me": "me",
    "i": "me",
    "my_mine": "my",
    "mine": "my",
    "thanks": "thank_you",
    "thankyou": "thank_you",
    "look_see_watch": "look",
    "see": "look",
    "watch": "look",
    "not_dont_doesnt": "dont",
    "doesnt": "dont",
    "not_want": "dont_want",
}


def get_display_name(action_id: str) -> str:
    norm = action_id.lower().strip()
    norm = GLOSS_ALIASES.get(norm, norm)
    if norm in ACTION_MAP:
        return ACTION_MAP[norm]["display"]
    return action_id.replace("_", " ").title()


def get_category(action_id: str) -> str:
    norm = action_id.lower().strip()
    norm = GLOSS_ALIASES.get(norm, norm)
    if norm in ACTION_MAP:
        return ACTION_MAP[norm]["category"]
    return "General"


def normalize_action_id(token: str) -> str:
    t = str(token).lower().strip().replace("-", "_").replace(" ", "_").replace("/", "_")
    return GLOSS_ALIASES.get(t, t)


# =============================================================================
# 2. MEDIAPIPE & FEATURE EXTRACTION HELPERS (EMBEDDED)
# =============================================================================

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def mediapipe_detection(image, model):
    """Processes BGR OpenCV frame through MediaPipe Holistic."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_styled_landmarks(image, results):
    """Draws styled landmarks with high-visibility colors on image."""
    # Face mesh tessellation
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
        )
    # Pose
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
        )
    # Left hand
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
        )
    # Right hand
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )


def extract_keypoints_1662(results):
    """1662-dim vector: pose(33*4) + face(468*3) + left_hand(21*3) + right_hand(21*3)."""
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh]).astype(np.float32)


def extract_keypoints_258(results):
    """258-dim vector: pose(33*4) + left_hand(21*3) + right_hand(21*3) (used by MTAN-SLR)."""
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, lh, rh]).astype(np.float32)


# =============================================================================
# 3. DIRECTORY SETUP & CHECKS
# =============================================================================

LEGACY_DATA_PATH = os.path.join("MP_Data")
MODERN_DATA_PATH = os.path.join("dataset", "MP_Data")

actions = np.array(ACTION_IDS)
no_sequences = 5
sequence_length = 30


def create_folders(action_list, num_seqs=no_sequences, legacy_path=LEGACY_DATA_PATH, modern_path=MODERN_DATA_PATH):
    for action in action_list:
        for sequence in range(num_seqs):
            os.makedirs(os.path.join(legacy_path, action, str(sequence)), exist_ok=True)
        os.makedirs(os.path.join(modern_path, action), exist_ok=True)


def is_sequence_complete(action, sequence_idx, seq_len=sequence_length, legacy_path=LEGACY_DATA_PATH, modern_path=MODERN_DATA_PATH):
    legacy_folder = os.path.join(legacy_path, action, str(sequence_idx))
    legacy_done = False
    if os.path.isdir(legacy_folder):
        files = [f for f in os.listdir(legacy_folder) if f.endswith(".npy")]
        if len(files) >= seq_len:
            legacy_done = True
    modern_done = os.path.exists(os.path.join(modern_path, action, f"{sequence_idx}.npy"))
    return legacy_done or modern_done


# =============================================================================
# 4. OPENCV USER INTERFACE OVERLAY
# =============================================================================

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
    h, w = image.shape[:2]

    # Top dark glass bar (height 75px)
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, 75), (15, 23, 42), -1)
    # Bottom bar (height 45px)
    cv2.rectangle(overlay, (0, h - 45), (w, h), (15, 23, 42), -1)
    cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)

    display_name = get_display_name(action)

    # Word Title & Category Badge
    cv2.putText(image, f"{display_name.upper()}", (20, 36), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, f"[{category}]", (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (56, 189, 248), 1, cv2.LINE_AA)

    # Video progress & Global progress (top right)
    vid_text = f"Video: {sequence_idx + 1}/{total_seqs}"
    cv2.putText(image, vid_text, (w - 220, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (16, 185, 129), 2, cv2.LINE_AA)
    pct = (action_idx / max(total_actions, 1)) * 100
    prog_text = f"Sign {action_idx + 1}/{total_actions} ({pct:.0f}%)"
    cv2.putText(image, prog_text, (w - 220, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (203, 213, 225), 1, cv2.LINE_AA)

    # Bottom Instructions Bar
    ctrl_text = "Controls: 'q'=Quit | 's'=Skip | 'r'=Redo Video | 'p'=Pause"
    cv2.putText(image, ctrl_text, (20, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (148, 163, 184), 1, cv2.LINE_AA)

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
        cv2.putText(image, frame_str, (w - 240, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (16, 185, 129), 1, cv2.LINE_AA)

    # Paused Banner
    if is_paused:
        pause_box = image.copy()
        cv2.rectangle(pause_box, (w // 2 - 150, h // 2 - 40), (w // 2 + 150, h // 2 + 40), (30, 41, 59), -1)
        cv2.addWeighted(pause_box, 0.9, image, 0.1, 0, image)
        cv2.putText(image, "PAUSED - Press 'p' to Resume", (w // 2 - 140, h // 2 + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


# =============================================================================
# 5. CORE COLLECTION LOOP
# =============================================================================

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
    action_list = list(action_list or actions)

    if start_from:
        start_id = normalize_action_id(start_from)
        if start_id in action_list:
            idx = action_list.index(start_id)
            action_list = action_list[idx:]
            print(f"[Collect] Resuming from action: '{start_id}' ({len(action_list)} remaining)")

    create_folders(action_list, num_seqs, legacy_path, modern_path)

    # Try Windows DirectShow first, then standard OpenCV capture
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

    total_actions = len(action_list)
    print("=" * 65)
    print(f"SignVision Standalone Collector - {total_actions} Words × {num_seqs} Videos × {seq_len} Frames")
    print(f"Legacy Output Path : {os.path.abspath(legacy_path)}")
    print(f"Modern Output Path : {os.path.abspath(modern_path)}")
    print("Controls: 'q'=Quit | 's'=Skip Word | 'r'=Redo Video | 'p'=Pause")
    print("=" * 65)

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
                if resume and is_sequence_complete(action, seq_idx, seq_len, legacy_path, modern_path):
                    print(f"  [Skip] {action} (video {seq_idx + 1}/{num_seqs}) already exists.")
                    seq_idx += 1
                    continue

                # Countdown Phase (2.5 seconds preparation)
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
                    elif key in (ord("p"), 32):  # Pause
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

                # Frame Recording Phase (30 frames)
                frame_window_1662 = []
                frame_window_258 = []
                redo_video = False

                for frame_num in range(seq_len):
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)

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

                # Save Modern Sequence File (shape: 30, 258)
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
    parser = argparse.ArgumentParser(description="SignVision Standalone 100-Word Data Collection Tool")
    parser.add_argument("--action", type=str, default=None, help="Record only a specific action (e.g. 'bathroom')")
    parser.add_argument("--category", type=str, default=None, help="Record only a specific category (e.g. 'Food', 'Home')")
    parser.add_argument("--start_from", type=str, default=None, help="Start recording from a specific action name")
    parser.add_argument("--num_videos", type=int, default=no_sequences, help="Number of videos per action (default: 5)")
    parser.add_argument("--sequence_length", type=int, default=sequence_length, help="Frames per video (default: 30)")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing recorded sequences")
    args = parser.parse_args()

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
