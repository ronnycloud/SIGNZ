"""
Fast Concurrent WLASL Video Downloader
=======================================
Downloads videos for the top N glosses in WLASL using concurrent worker threads.
Supports YouTube videos (via yt-dlp) and host direct links.

Usage:
    python video_downloader.py --num_classes 100 --workers 4
"""

import os
import sys
import json
import time
import argparse
import urllib.request
import subprocess
import concurrent.futures

# Ensure local venv/Scripts is in PATH for yt-dlp
scripts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "venv", "Scripts"))
if os.path.exists(scripts_dir) and scripts_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = scripts_dir + os.pathsep + os.environ.get("PATH", "")

YOUTUBE_DOWNLOADER = "yt-dlp"


def download_single_video(instance: dict, saveto: str = "raw_videos"):
    video_url = instance.get("url", "")
    video_id = instance.get("video_id", "")
    if not video_url:
        return False

    os.makedirs(saveto, exist_ok=True)

    is_yt = "youtube" in video_url or "youtu.be" in video_url
    yt_id = video_url[-11:] if is_yt and len(video_url) >= 11 else ""

    # Check if already exists
    if is_yt:
        for ext in [".mp4", ".mkv", ".webm"]:
            if os.path.exists(os.path.join(saveto, f"{yt_id}{ext}")) or os.path.exists(os.path.join(saveto, f"{video_id}{ext}")):
                return True
    else:
        for ext in [".mp4", ".swf", ".avi", ".webm"]:
            if os.path.exists(os.path.join(saveto, f"{video_id}{ext}")):
                return True

    if is_yt:
        # Download via yt-dlp
        out_template = os.path.join(saveto, "%(id)s.%(ext)s")
        cmd = [YOUTUBE_DOWNLOADER, "--no-warning", "-o", out_template, video_url]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if res.returncode == 0:
                print(f"  [YT OK] {video_id} ({yt_id})")
                return True
            else:
                # Video unavailable/private/deleted
                return False
        except Exception:
            return False
    else:
        # Direct URL download
        dest_path = os.path.join(saveto, f"{video_id}.mp4")
        req = urllib.request.Request(video_url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
        })
        try:
            with urllib.request.urlopen(req, timeout=5) as response, open(dest_path, 'wb') as out_file:
                out_file.write(response.read())
            print(f"  [HOST OK] {video_id}")
            return True
        except Exception:
            return False


def download_wlasl_videos(indexfile: str = "WLASL_v0.3.json", num_classes: int = 100, max_workers: int = 4):
    with open(indexfile, "r", encoding="utf-8") as f:
        content = json.load(f)

    selected = content[:num_classes]
    all_instances = []
    for entry in selected:
        for inst in entry.get("instances", []):
            all_instances.append(inst)

    print(f"[WLASL Downloader] Downloading {len(all_instances)} video instances across top {len(selected)} classes with {max_workers} threads...")

    success = 0
    failed = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_single_video, inst): inst for inst in all_instances}
        for future in concurrent.futures.as_completed(futures):
            if future.result():
                success += 1
            else:
                failed += 1

    print(f"\n[WLASL Downloader Finished] Total: {len(all_instances)} | Success/Existing: {success} | Failed/Unavailable: {failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download WLASL videos concurrently")
    parser.add_argument("--json_path", default="WLASL_v0.3.json", help="Path to WLASL JSON file")
    parser.add_argument("--num_classes", type=int, default=100, help="Number of top gloss classes")
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent downloader threads")
    args = parser.parse_args()

    download_wlasl_videos(indexfile=args.json_path, num_classes=args.num_classes, max_workers=args.workers)
