"""
MTAN-SLR Model Package
======================
Exports primary and alternative architectures, plus model factory helper.
"""

from src.models.temporal_encoder import MultiModalTemporalEncoder
from src.models.segmenter import GestureSegmenter
from src.models.classifier import SignClassifier
from src.models.alternatives import TransformerSignModel, CNNLSTMModel, BaselineLSTMModel

__all__ = [
    "MultiModalTemporalEncoder",
    "GestureSegmenter",
    "SignClassifier",
    "TransformerSignModel",
    "CNNLSTMModel",
    "BaselineLSTMModel",
    "build_model_from_config",
]


def build_model_from_config(model_config: dict, num_classes: int):
    """Instantiates a PyTorch model based on configuration parameters."""
    active_model = model_config.get("active_model", "bilstm_attention")

    if active_model == "bilstm_attention":
        cfg = model_config.get("bilstm_attention", {})
        return MultiModalTemporalEncoder(
            num_classes=num_classes,
            hand_input_dim=cfg.get("hand_input_dim", 126),
            pose_input_dim=cfg.get("pose_input_dim", 100),
            hand_hidden_dim=cfg.get("hand_hidden_dim", 64),
            pose_hidden_dim=cfg.get("pose_hidden_dim", 32),
            hand_num_layers=cfg.get("hand_num_layers", 2),
            pose_num_layers=cfg.get("pose_num_layers", 1),
            hand_dropout=cfg.get("hand_dropout", 0.3),
            pose_dropout=cfg.get("pose_dropout", 0.2),
            cross_attn_heads=cfg.get("cross_attn_heads", 4),
            cross_attn_dropout=cfg.get("cross_attn_dropout", 0.1),
            temporal_attn_heads=cfg.get("temporal_attn_heads", 4),
            temporal_attn_dropout=cfg.get("temporal_attn_dropout", 0.1),
            classifier_hidden=cfg.get("classifier_hidden", 128),
            classifier_dropout=cfg.get("classifier_dropout", 0.3),
        )

    elif active_model == "transformer":
        cfg = model_config.get("transformer", {})
        return TransformerSignModel(
            num_classes=num_classes,
            input_dim=cfg.get("input_dim", 454),
            d_model=cfg.get("d_model", 256),
            nhead=cfg.get("nhead", 8),
            num_layers=cfg.get("num_layers", 4),
            dim_feedforward=cfg.get("dim_feedforward", 512),
            dropout=cfg.get("dropout", 0.1),
            max_seq_len=cfg.get("max_seq_len", 120),
            classifier_hidden=cfg.get("classifier_hidden", 128),
            classifier_dropout=cfg.get("classifier_dropout", 0.3),
        )

    elif active_model == "cnn_lstm":
        cfg = model_config.get("cnn_lstm", {})
        return CNNLSTMModel(
            num_classes=num_classes,
            input_dim=cfg.get("input_dim", 454),
            cnn_channels=cfg.get("cnn_channels", [64, 128, 128]),
            cnn_kernels=cfg.get("cnn_kernels", [3, 5, 7]),
            lstm_hidden=cfg.get("lstm_hidden", 128),
            lstm_layers=cfg.get("lstm_layers", 2),
            lstm_dropout=cfg.get("lstm_dropout", 0.3),
            classifier_hidden=cfg.get("classifier_hidden", 128),
            classifier_dropout=cfg.get("classifier_dropout", 0.3),
        )

    elif active_model == "baseline_lstm":
        cfg = model_config.get("baseline_lstm", {})
        return BaselineLSTMModel(
            num_classes=num_classes,
            input_dim=cfg.get("input_dim", 454),
            hidden_dims=cfg.get("hidden_dims", [64, 128, 64]),
            dense_dims=cfg.get("dense_dims", [64, 32]),
            dropout=cfg.get("dropout", 0.2),
        )

    else:
        raise ValueError(f"Unknown active_model: {active_model}")
