"""
Alternative model architectures for MTAN-SLR.
"""

from src.models.alternatives.transformer import TransformerSignModel
from src.models.alternatives.cnn_lstm import CNNLSTMModel
from src.models.alternatives.baseline_lstm import BaselineLSTMModel

__all__ = [
    "TransformerSignModel",
    "CNNLSTMModel",
    "BaselineLSTMModel",
]
