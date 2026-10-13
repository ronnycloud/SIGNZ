"""
Sign Classifier
================
Wraps any temporal encoder with a classification head and provides
convenience methods for prediction, top-K, and confidence extraction.

This is the top-level model used during training and inference.
It delegates to the chosen encoder (BiLSTM+Attention, Transformer, etc.)
and adds a unified interface.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from pathlib import Path
from typing import Optional, Dict, Tuple, List

from .temporal_encoder import MultiModalTemporalEncoder


class SignClassifier(nn.Module):
    """Top-level sign language classifier.

    Wraps a temporal encoder and provides:
    - forward() → logits
    - predict() → class index + confidence
    - predict_top_k() → top-K classes with probabilities
    """

    def __init__(
        self,
        num_classes: int,
        encoder_type: str = "bilstm_attention",
        config: Optional[Dict] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.encoder_type = encoder_type

        if encoder_type == "bilstm_attention":
            cfg = config or {}
            self.encoder = MultiModalTemporalEncoder(
                num_classes=num_classes,
                hand_input_dim=cfg.get("hand_input_dim", 126),
                pose_input_dim=cfg.get("pose_input_dim", 100),
                velocity_dim=cfg.get("velocity_dim", 226),
                flags_dim=cfg.get("flags_dim", 2),
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
        elif encoder_type == "transformer":
            from .alternatives.transformer_encoder import TransformerSignEncoder
            cfg = config or {}
            self.encoder = TransformerSignEncoder(
                num_classes=num_classes, **cfg
            )
        elif encoder_type == "cnn_lstm":
            from .alternatives.cnn_lstm import CNNLSTMEncoder
            cfg = config or {}
            self.encoder = CNNLSTMEncoder(
                num_classes=num_classes, **cfg
            )
        elif encoder_type == "baseline_lstm":
            from .alternatives.baseline_lstm import BaselineLSTM
            cfg = config or {}
            self.encoder = BaselineLSTM(
                num_classes=num_classes, **cfg
            )
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : (batch, T, feature_dim)
        mask : (batch, T), optional

        Returns
        -------
        (batch, num_classes) logits
        """
        return self.encoder(x, mask)

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict class and confidence.

        Returns
        -------
        class_indices : (batch,)
        confidences : (batch,)
        """
        self.eval()
        logits = self.forward(x, mask)
        probs = F.softmax(logits, dim=-1)
        confidences, indices = probs.max(dim=-1)
        return indices, confidences

    @torch.no_grad()
    def predict_top_k(
        self,
        x: torch.Tensor,
        k: int = 5,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict top-K classes with probabilities.

        Returns
        -------
        top_k_indices : (batch, k)
        top_k_probs : (batch, k)
        """
        self.eval()
        logits = self.forward(x, mask)
        probs = F.softmax(logits, dim=-1)
        top_k_probs, top_k_indices = probs.topk(k, dim=-1)
        return top_k_indices, top_k_probs

    @torch.no_grad()
    def get_probabilities(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get full probability distribution.

        Returns
        -------
        (batch, num_classes) probabilities
        """
        self.eval()
        logits = self.forward(x, mask)
        return F.softmax(logits, dim=-1)

    @classmethod
    def from_config(cls, config_path: str, num_classes: int) -> "SignClassifier":
        """Create a classifier from a YAML config file."""
        with open(config_path, "r") as f:
            full_config = yaml.safe_load(f)

        encoder_type = full_config.get("active_model", "bilstm_attention")
        model_config = full_config.get(encoder_type, {})

        return cls(
            num_classes=num_classes,
            encoder_type=encoder_type,
            config=model_config,
        )

    def save(self, path: str, metadata: Optional[Dict] = None):
        """Save model checkpoint with metadata."""
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "num_classes": self.num_classes,
            "encoder_type": self.encoder_type,
        }
        if metadata:
            checkpoint["metadata"] = metadata
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str, config: Optional[Dict] = None) -> "SignClassifier":
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(
            num_classes=checkpoint["num_classes"],
            encoder_type=checkpoint.get("encoder_type", "bilstm_attention"),
            config=config,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
