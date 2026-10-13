"""
Baseline PyTorch LSTM Model for Sign Language Recognition
===========================================================
PyTorch implementation reproducing baseline stacked LSTM architecture.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class BaselineLSTMModel(nn.Module):
    """Stacked LSTM model for keypoint sequence classification."""

    def __init__(
        self,
        num_classes: int,
        input_dim: int = 454,
        hidden_dims: List[int] = None,
        dense_dims: List[int] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [64, 128, 64]
        dense_dims = dense_dims or [64, 32]

        self.lstm_layers = nn.ModuleList()
        in_dim = input_dim
        for h_dim in hidden_dims:
            self.lstm_layers.append(
                nn.LSTM(input_size=in_dim, hidden_size=h_dim, batch_first=True)
            )
            in_dim = h_dim

        dense_list = []
        in_dense = hidden_dims[-1]
        for d_dim in dense_dims:
            dense_list.extend([
                nn.Linear(in_dense, d_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dense = d_dim

        dense_list.append(nn.Linear(in_dense, num_classes))
        self.classifier = nn.Sequential(*dense_list)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (batch, T, 454)
        mask: (batch, T) bool tensor
        """
        out = x
        for lstm in self.lstm_layers:
            out, _ = lstm(out)

        B = x.shape[0]
        if mask is not None:
            lengths = mask.sum(dim=1) - 1
            lengths = torch.clamp(lengths, min=0)
            pooled = out[torch.arange(B, device=x.device), lengths]
        else:
            pooled = out[:, -1, :]

        return self.classifier(pooled)
