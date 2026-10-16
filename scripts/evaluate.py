"""
Evaluation Script for MTAN-SLR
==============================
Re-creates the exact train/val/test split used by train.py (same seed and
ratios) and reports top-1 / top-5 accuracy for each split, plus CPU latency
for the PyTorch model, the ONNX model and MediaPipe landmark extraction.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --skip_latency
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import glob
import json
import time

import cv2
import numpy as np
import torch
import yaml

from src.data.dataset import create_dataloaders
from src.models import build_model_from_config


def _bench(fn, n_runs: int = 200, warmup: int = 20) -> dict:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return {"mean_ms": round(float(np.mean(times)), 2), "p95_ms": round(float(np.percentile(times, 95)), 2)}


@torch.no_grad()
def _accuracy(model, loader) -> dict:
    n = top1 = top5 = 0
    for x, y, mask in loader:
        k = min(5, model(x[:1], mask=mask[:1]).shape[-1])
        pred = model(x, mask=mask).topk(k, dim=-1).indices
        top1 += (pred[:, 0] == y).sum().item()
        top5 += (pred == y[:, None]).any(-1).sum().item()
        n += y.numel()
    return {"n": n, "top1": round(100 * top1 / max(n, 1), 2), "top5": round(100 * top5 / max(n, 1), 2)}


def evaluate(checkpoint_path: str, onnx_path: str, skip_latency: bool) -> dict:
    model_cfg = yaml.safe_load(open("configs/model_config.yaml", encoding="utf-8"))
    train_cfg = yaml.safe_load(open("configs/training_config.yaml", encoding="utf-8"))

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_cfg["active_model"] = ckpt.get("active_model", model_cfg.get("active_model"))
    model = build_model_from_config(model_cfg, num_classes=ckpt["num_classes"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    d = train_cfg["data"]
    train_loader, val_loader, test_loader, dataset = create_dataloaders(
        data_dir=d["data_path"],
        max_frames=d["max_sequence_length"],
        batch_size=32,
        train_split=d["train_split"],
        val_split=d["val_split"],
        test_split=d["test_split"],
        num_workers=0,
        pin_memory=False,
        augmentor=None,  # evaluate on clean, un-augmented data
        seed=train_cfg.get("seed", 42),
    )
    if list(dataset.classes) != list(ckpt.get("classes", dataset.classes)):
        raise RuntimeError("Class order in dataset does not match the checkpoint.")

    results = {name: _accuracy(model, loader) for name, loader in
               [("train", train_loader), ("val", val_loader), ("test", test_loader)]}
    results["classes_with_data"] = sum(
        1 for c in dataset.classes if glob.glob(os.path.join(d["data_path"], c, "*.npy")))

    if skip_latency:
        return results

    x = torch.randn(1, 30, 454)
    mask = torch.ones(1, 30, dtype=torch.bool)
    with torch.no_grad():
        results["pytorch_cpu"] = _bench(lambda: model(x, mask=mask))

    if os.path.exists(onnx_path):
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            feeds = {"input_features": x.numpy(), "mask": mask.numpy()}
            if sess.run(None, feeds)[0].shape[-1] != ckpt["num_classes"]:
                results["onnx_cpu"] = "stale model.onnx - re-run scripts/export_onnx.py"
            else:
                results["onnx_cpu"] = _bench(lambda: sess.run(None, feeds))
        except ImportError:
            results["onnx_cpu"] = "onnxruntime not installed"

    videos = sorted(glob.glob("WLASL/start_kit/raw_videos/*.mp4"))
    if videos:
        from src.features.extractor import FeatureExtractor
        frames, cap = [], cv2.VideoCapture(videos[0])
        while len(frames) < 120:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.resize(frame, (640, 480)))
        if len(frames) > 20:
            with FeatureExtractor() as fe:
                it = iter(frames * 10)
                results["mediapipe_640x480"] = _bench(lambda: fe.process_frame(next(it)),
                                                      n_runs=len(frames) - 10, warmup=10)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MTAN-SLR accuracy and latency")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--onnx", default="checkpoints/model.onnx")
    parser.add_argument("--skip_latency", action="store_true")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.checkpoint, args.onnx, args.skip_latency), indent=2))
