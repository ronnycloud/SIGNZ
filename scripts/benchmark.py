"""
MTAN-SLR Performance Benchmarking Script
=========================================
Measures inference latency (mean, std, min, max ms), FPS throughput,
and model size across PyTorch and ONNX execution modes.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import time
import yaml
import numpy as np
import torch

from src.models import build_model_from_config
from src.utils.logger import get_logger
from src.utils.metrics import MetricsCalculator

logger = get_logger("MTAN_Benchmark")


def benchmark_system(
    checkpoint_path: str = "checkpoints/best_model.pt",
    model_config_path: str = "configs/model_config.yaml",
    n_runs: int = 100,
    seq_len: int = 30,
):
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found at '{checkpoint_path}'. Run train.py first.")
        return

    with open(model_config_path, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    actions = checkpoint.get("actions", ["hello", "thanks", "iloveyou"])
    active_model = checkpoint.get("active_model", model_cfg.get("active_model", "bilstm_attention"))

    model_cfg["active_model"] = active_model
    model = build_model_from_config(model_cfg, num_classes=len(actions))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dummy_input = torch.randn(1, seq_len, 454).to(device)
    dummy_mask = torch.ones(1, seq_len, dtype=torch.bool).to(device)

    logger.info(f"Running benchmark for '{active_model}' on {device} ({n_runs} runs)...")

    # PyTorch latency benchmark
    def run_pt():
        with torch.no_grad():
            _ = model(dummy_input, mask=dummy_mask)

    pt_latency = MetricsCalculator.measure_latency(run_pt, n_runs=n_runs)

    file_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)

    logger.info("=" * 60)
    logger.info("MTAN-SLR BENCHMARK RESULTS")
    logger.info("=" * 60)
    logger.info(f"Model Variant:      {active_model}")
    logger.info(f"Execution Device:   {device}")
    logger.info(f"Sequence Length:    {seq_len} frames")
    logger.info(f"Checkpoint Size:    {file_size_mb:.2f} MB")
    logger.info(f"Mean Latency:       {pt_latency['mean_ms']:.2f} ms")
    logger.info(f"Std Latency:        {pt_latency['std_ms']:.2f} ms")
    logger.info(f"Min / Max Latency:  {pt_latency['min_ms']:.2f} ms / {pt_latency['max_ms']:.2f} ms")
    logger.info(f"Throughput (FPS):   {pt_latency['fps']:.1f} FPS")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark MTAN-SLR inference performance")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--n_runs", type=int, default=100)
    args = parser.parse_args()

    benchmark_system(checkpoint_path=args.checkpoint, n_runs=args.n_runs)
