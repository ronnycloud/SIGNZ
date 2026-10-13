"""
Transformer Encoder Model for Sign Language Recognition
=========================================================
Alternative architecture specified in model_config.yaml.
Uses sinusoidal positional encoding, standard PyTorch TransformerEncoder layers,
and a learnable CLS token for temporal pooling.
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence tokens."""

    def __init__(self, d_model: int, max_len: int = 200, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class TransformerSignModel(nn.Module):
    """Transformer Encoder model for action recognition from 454-dim keypoint features."""

    def __init__(
        self,
        num_classes: int,
        input_dim: int = 454,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 120,
        classifier_hidden: int = 128,
        classifier_dropout: float = 0.3,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (batch, T, 454)
        mask: (batch, T) bool tensor, True for valid positions
        """
        B, T, _ = x.shape
        h = self.input_proj(x)  # (B, T, d_model)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        h = torch.cat([cls_tokens, h], dim=1)  # (B, T+1, d_model)

        h = self.pos_encoder(h)

        if mask is not None:
            cls_mask = torch.ones(B, 1, dtype=torch.bool, device=mask.device)
            full_mask = torch.cat([cls_mask, mask], dim=1)
            src_key_padding_mask = ~full_mask
        else:
            src_key_padding_mask = None

        encoded = self.transformer_encoder(h, src_key_padding_mask=src_key_padding_mask)
        cls_out = encoded[:, 0, :]  # (B, d_model)
        return self.classifier(cls_out)
