"""
CNN-LSTM Hybrid Model for Sign Language Recognition
===================================================
Alternative architecture specified in model_config.yaml.
Uses 1D Conv layers to extract local temporal patterns followed by an LSTM.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class CNNLSTMModel(nn.Module):
    """Hybrid 1D Conv + LSTM model for action recognition."""

    def __init__(
        self,
        num_classes: int,
        input_dim: int = 454,
        cnn_channels: List[int] = None,
        cnn_kernels: List[int] = None,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.3,
        classifier_hidden: int = 128,
        classifier_dropout: float = 0.3,
    ):
        super().__init__()
        cnn_channels = cnn_channels or [64, 128, 128]
        cnn_kernels = cnn_kernels or [3, 5, 7]

        conv_layers = []
        in_ch = input_dim
        for out_ch, k in zip(cnn_channels, cnn_kernels):
            padding = k // 2
            conv_layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=padding),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
            ])
            in_ch = out_ch

        self.cnn = nn.Sequential(*conv_layers)
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (batch, T, 454)
        mask: (batch, T) bool tensor
        """
        B, T, D = x.shape
        # Conv1D expects (B, C, T)
        x_conv = x.transpose(1, 2)
        feat = self.cnn(x_conv)  # (B, channels, T)
        feat = feat.transpose(1, 2)  # (B, T, channels)

        lstm_out, (hn, _) = self.lstm(feat)  # (B, T, lstm_hidden)

        if mask is not None:
            # Masked pooling: take last valid time step per sequence
            lengths = mask.sum(dim=1) - 1
            lengths = torch.clamp(lengths, min=0)
            pooled = lstm_out[torch.arange(B, device=x.device), lengths]
        else:
            pooled = lstm_out[:, -1, :]

        return self.classifier(pooled)
