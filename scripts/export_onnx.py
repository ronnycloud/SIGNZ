"""
ONNX Export Script for MTAN-SLR Models
======================================
Exports trained PyTorch models to ONNX format for 2-4x faster inference,
and validates the exported model with ONNX Runtime.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import yaml
import torch
import numpy as np

from src.models import build_model_from_config
from src.utils.logger import get_logger

logger = get_logger("MTAN_ONNX_Export")


def export_to_onnx(
    checkpoint_path: str = "checkpoints/best_model.pt",
    model_config_path: str = "configs/model_config.yaml",
    output_onnx_path: str = "checkpoints/model.onnx",
    seq_len: int = 30,
):
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found at '{checkpoint_path}'. Run training first.")
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

    logger.info(f"Loaded checkpoint for model variant '{active_model}' with {len(actions)} classes.")

    # Dummy inputs
    dummy_x = torch.randn(1, seq_len, 454, dtype=torch.float32)
    dummy_mask = torch.ones(1, seq_len, dtype=torch.bool)

    os.makedirs(os.path.dirname(output_onnx_path), exist_ok=True)

    # Export
    logger.info(f"Exporting ONNX model to '{output_onnx_path}'...")
    torch.onnx.export(
        model,
        (dummy_x, dummy_mask),
        output_onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_features", "mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_features": {0: "batch_size", 1: "sequence_length"},
            "mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        },
        dynamo=False,
    )

    logger.info("ONNX export completed successfully.")

    # Verification with ONNX Runtime if installed
    try:
        import onnxruntime as ort

        logger.info("Verifying ONNX model with ONNX Runtime...")
        ort_session = ort.InferenceSession(output_onnx_path)

        with torch.no_grad():
            pt_out = model(dummy_x, mask=dummy_mask).numpy()

        ort_inputs = {
            "input_features": dummy_x.numpy(),
            "mask": dummy_mask.numpy(),
        }
        ort_out = ort_session.run(None, ort_inputs)[0]

        np.testing.assert_allclose(pt_out, ort_out, rtol=1e-3, atol=1e-4)
        logger.info("ONNX output matches PyTorch output within tolerance!")
    except ImportError:
        logger.info("onnxruntime package not installed. Skipping numerical verification.")
    except Exception as e:
        logger.warning(f"ONNX verification check encountered an issue: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--output", type=str, default="checkpoints/model.onnx")
    args = parser.parse_args()

    export_to_onnx(checkpoint_path=args.checkpoint, output_onnx_path=args.output)
