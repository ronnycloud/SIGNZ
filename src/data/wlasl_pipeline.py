"""
WLASL Dataset Pipeline
=======================
Automates the full workflow of converting the WLASL video dataset into
MediaPipe landmark sequences (.npy files) organized by gloss label.

Pipeline steps:
    1. Parse WLASL JSON metadata
    2. Locate or download video files
    3. Run MediaPipe Holistic on each video
    4. Save per-video landmark arrays to  MP_Data/<gloss>/<video_id>.npy
    5. Validate outputs and remove corrupt samples
    6. Generate label_map.json and dataset_stats.json

Usage:
    python -m src.data.wlasl_pipeline --json_path WLASL/start_kit/WLASL_v0.3.json \
           --video_dir WLASL/start_kit/videos --output_dir dataset/MP_Data --num_classes 100
"""

import os
import sys
import json
import argparse
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

# Allow running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.features.extractor import extract_keypoints

mp_holistic = mp.solutions.holistic


# --------------------------------------------------------------------------
# 1. Parse WLASL JSON
# --------------------------------------------------------------------------

def parse_wlasl_json(json_path: str, num_classes: int = 100) -> list:
    """Parse the WLASL metadata JSON and return entries for the top N classes.

    The WLASL JSON is a list of dicts, each with:
        - gloss: str          (the word label)
        - instances: list     (each has video_id, split, signer_id, ...)

    Classes in the JSON are ordered by decreasing number of samples, so
    taking the first `num_classes` gives the best-represented words.

    Returns
    -------
    list of dict
        Each dict has keys: gloss, video_id, split, url (if present).
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Take only the first num_classes glosses
    selected = data[:num_classes]

    entries = []
    for gloss_entry in selected:
        gloss = gloss_entry["gloss"]
        for instance in gloss_entry.get("instances", []):
            entries.append({
                "gloss": gloss,
                "video_id": instance.get("video_id", ""),
                "split": instance.get("split", ""),
                "signer_id": instance.get("signer_id", ""),
                "url": instance.get("url", ""),
                "frame_start": instance.get("frame_start", -1),
                "frame_end": instance.get("frame_end", -1),
            })

    print(f"[WLASL] Parsed {len(entries)} instances across {len(selected)} glosses.")
    return entries


# --------------------------------------------------------------------------
# 2. Locate video files
# --------------------------------------------------------------------------

def find_video_path(video_id: str, video_dir: str, url: str = "") -> str:
    """Return the full path to a WLASL video file given its ID and optional URL.

    Checks:
        - <video_dir>/<video_id>.<ext>
        - <video_dir>/<yt_id>.<ext>
        - <video_dir>/../raw_videos/<video_id>.<ext>
        - <video_dir>/../raw_videos/<yt_id>.<ext>
    """
    yt_id = url[-11:] if ("youtube" in url or "youtu.be" in url) and len(url) >= 11 else ""

    search_dirs = [video_dir]
    raw_dir = os.path.join(os.path.dirname(video_dir.rstrip("\\/")), "raw_videos")
    if os.path.exists(raw_dir):
        search_dirs.append(raw_dir)

    name_stems = [video_id]
    if yt_id and yt_id not in name_stems:
        name_stems.append(yt_id)

    for sdir in search_dirs:
        for stem in name_stems:
            for ext in [".mp4", ".avi", ".mkv", ".webm"]:
                candidate = os.path.join(sdir, f"{stem}{ext}")
                if os.path.exists(candidate):
                    return candidate

    return os.path.join(video_dir, f"{video_id}.mp4")


# --------------------------------------------------------------------------
# 3. Extract landmarks from a single video
# --------------------------------------------------------------------------

def process_single_video(
    video_path: str,
    holistic_model,
    frame_start: int = -1,
    frame_end: int = -1,
    min_frames: int = 5,
) -> np.ndarray:
    """Extract MediaPipe landmarks from a video file.

    Parameters
    ----------
    video_path : str
        Path to the video file.
    holistic_model
        An active MediaPipe Holistic instance.
    frame_start, frame_end : int
        If both >= 0, only process frames in [frame_start, frame_end].
    min_frames : int
        Discard videos shorter than this.

    Returns
    -------
    np.ndarray or None
        Shape (T, 258) where T is the number of frames. Returns None if
        the video cannot be read or has too few valid frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    frames_data = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Apply frame range trimming if specified
        if frame_start >= 0 and frame_end >= 0:
            if frame_idx < frame_start:
                frame_idx += 1
                continue
            if frame_idx > frame_end:
                break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic_model.process(image)

        keypoints = extract_keypoints(results)
        frames_data.append(keypoints)
        frame_idx += 1

    cap.release()

    if len(frames_data) < min_frames:
        return None

    return np.array(frames_data, dtype=np.float32)


# --------------------------------------------------------------------------
# 4. Run full pipeline
# --------------------------------------------------------------------------

def run_wlasl_pipeline(
    json_path: str,
    video_dir: str,
    output_dir: str,
    num_classes: int = 100,
    min_frames: int = 5,
):
    """Main entry point: parse JSON → process videos → save .npy files.

    Folder structure created:
        output_dir/
        ├── <gloss_1>/
        │   ├── <video_id_a>.npy
        │   └── <video_id_b>.npy
        ├── <gloss_2>/
        └── ...
    """
    entries = parse_wlasl_json(json_path, num_classes=num_classes)

    os.makedirs(output_dir, exist_ok=True)

    # Track statistics
    stats = {
        "total_entries": len(entries),
        "processed": 0,
        "skipped_missing": 0,
        "skipped_short": 0,
        "skipped_corrupt": 0,
    }
    label_map = {}

    print(f"[WLASL] Processing {len(entries)} videos -> {output_dir}")

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:

        for entry in tqdm(entries, desc="Extracting landmarks"):
            gloss = entry["gloss"]
            video_id = entry["video_id"]

            # Create gloss directory
            gloss_dir = os.path.join(output_dir, gloss)
            os.makedirs(gloss_dir, exist_ok=True)

            # Track label
            if gloss not in label_map:
                label_map[gloss] = len(label_map)

            # Output path
            out_path = os.path.join(gloss_dir, f"{video_id}.npy")
            if os.path.exists(out_path):
                stats["processed"] += 1
                continue  # Resume support — skip already processed

            # Locate video file
            video_path = find_video_path(video_id, video_dir, url=entry.get("url", ""))
            if not os.path.exists(video_path):
                stats["skipped_missing"] += 1
                continue

            # Process
            try:
                landmarks = process_single_video(
                    video_path,
                    holistic,
                    frame_start=entry.get("frame_start", -1),
                    frame_end=entry.get("frame_end", -1),
                    min_frames=min_frames,
                )
            except Exception as e:
                print(f"  [ERROR] {video_id}: {e}")
                stats["skipped_corrupt"] += 1
                continue

            if landmarks is None:
                stats["skipped_short"] += 1
                continue

            np.save(out_path, landmarks)
            stats["processed"] += 1

    # ---- Save metadata --------------------------------------------------
    label_map_path = os.path.join(output_dir, "label_map.json")
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)

    stats_path = os.path.join(output_dir, "dataset_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*60}")
    print(f"WLASL Pipeline Complete")
    print(f"{'='*60}")
    print(f"  Processed:       {stats['processed']}")
    print(f"  Missing videos:  {stats['skipped_missing']}")
    print(f"  Too short:       {stats['skipped_short']}")
    print(f"  Corrupt:         {stats['skipped_corrupt']}")
    print(f"  Label map saved: {label_map_path}")
    print(f"  Stats saved:     {stats_path}")
    print(f"{'='*60}")


# --------------------------------------------------------------------------
# 5. Validation — scan MP_Data and remove corrupt .npy files
# --------------------------------------------------------------------------

def validate_dataset(data_dir: str, expected_dim: int = 258, min_frames: int = 5):
    """Scan all .npy files and remove those that are corrupt or degenerate.

    Checks:
        - File can be loaded with np.load
        - Shape is (T, expected_dim) with T >= min_frames
        - No NaN or Inf values
        - Not all-zero (completely failed detection)
    """
    removed = 0
    checked = 0

    for gloss in sorted(os.listdir(data_dir)):
        gloss_dir = os.path.join(data_dir, gloss)
        if not os.path.isdir(gloss_dir):
            continue

        for fname in os.listdir(gloss_dir):
            if not fname.endswith(".npy"):
                continue

            fpath = os.path.join(gloss_dir, fname)
            checked += 1

            try:
                arr = np.load(fpath)
                if arr.ndim != 2 or arr.shape[1] != expected_dim:
                    raise ValueError(f"Bad shape {arr.shape}")
                if arr.shape[0] < min_frames:
                    raise ValueError(f"Too short ({arr.shape[0]} frames)")
                if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                    raise ValueError("Contains NaN/Inf")
                if np.allclose(arr, 0):
                    raise ValueError("All zeros")
            except Exception as e:
                print(f"  [REMOVED] {fpath}: {e}")
                os.remove(fpath)
                removed += 1

    print(f"Validation complete: checked {checked}, removed {removed}")


# --------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WLASL → MediaPipe landmark pipeline")
    subparsers = parser.add_subparsers(dest="command")

    # ---- process command ----
    proc = subparsers.add_parser("process", help="Process WLASL videos into .npy landmarks")
    proc.add_argument("--json_path", required=True, help="Path to WLASL_v0.3.json")
    proc.add_argument("--video_dir", required=True, help="Directory containing WLASL video files")
    proc.add_argument("--output_dir", default="dataset/MP_Data", help="Output directory for .npy files")
    proc.add_argument("--num_classes", type=int, default=100, help="Number of gloss classes to process (default: 100)")
    proc.add_argument("--min_frames", type=int, default=5, help="Minimum frames per video (default: 5)")

    # ---- validate command ----
    val = subparsers.add_parser("validate", help="Validate and clean the processed dataset")
    val.add_argument("--data_dir", default="dataset/MP_Data", help="Directory to validate")
    val.add_argument("--expected_dim", type=int, default=258, help="Expected feature dimension")
    val.add_argument("--min_frames", type=int, default=5, help="Minimum frames per sample")

    args = parser.parse_args()

    if args.command == "process":
        run_wlasl_pipeline(
            json_path=args.json_path,
            video_dir=args.video_dir,
            output_dir=args.output_dir,
            num_classes=args.num_classes,
            min_frames=args.min_frames,
        )
    elif args.command == "validate":
        validate_dataset(
            data_dir=args.data_dir,
            expected_dim=args.expected_dim,
            min_frames=args.min_frames,
        )
    else:
        parser.print_help()
