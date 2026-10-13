"""
MultiModal Temporal Encoder (BiLSTM + Cross-Modal Attention)
=============================================================
The primary recognition model for MTAN-SLR.

Architecture:
    1. **Modality-separated BiLSTMs** — Hand landmarks and pose landmarks are
       encoded by independent bidirectional LSTMs. This prevents the larger
       pose stream (100 dims) from dominating the smaller but more informative
       hand stream (126 dims) during early training.

    2. **Cross-Modal Attention** — The hand stream queries the pose stream
       via multi-head attention, learning *which* body posture context is
       relevant for the current hand configuration.  Example: the sign "THINK"
       has the same hand shape as "FEEL" but differs in that the hand is near
       the forehead vs. the chest — cross-modal attention resolves this.

    3. **Temporal Self-Attention** — A single attention layer over the time
       dimension learns which frames in the sequence are most discriminative,
       assigning low weight to transition frames and high weight to the
       hold phase of each sign.

    4. **Classification Head** — A two-layer MLP with dropout maps the
       attention-pooled representation to class logits.

Input: (batch, T, 454) where T is variable (padded sequences with mask)
Output: (batch, num_classes) logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CrossModalAttention(nn.Module):
    """Multi-head attention where one modality queries another.

    Q = hand embeddings, K = V = pose embeddings.
    This lets the hand stream *select* which pose context is informative.
    """

    def __init__(
        self,
        hand_dim: int,
        pose_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Project both modalities to a common dimension for attention
        self.common_dim = hand_dim  # Use hand dim as the common space
        self.num_heads = num_heads
        assert self.common_dim % num_heads == 0, \
            f"common_dim ({self.common_dim}) must be divisible by num_heads ({num_heads})"

        self.q_proj = nn.Linear(hand_dim, self.common_dim)
        self.k_proj = nn.Linear(pose_dim, self.common_dim)
        self.v_proj = nn.Linear(pose_dim, self.common_dim)
        self.out_proj = nn.Linear(self.common_dim, hand_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hand_dim)

    def forward(
        self,
        hand_features: torch.Tensor,
        pose_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        hand_features : (batch, T, hand_dim)
        pose_features : (batch, T, pose_dim)
        mask : (batch, T), optional — True for valid positions

        Returns
        -------
        (batch, T, hand_dim) — hand features enriched with pose context
        """
        B, T, _ = hand_features.shape

        Q = self.q_proj(hand_features)  # (B, T, common_dim)
        K = self.k_proj(pose_features)  # (B, T, common_dim)
        V = self.v_proj(pose_features)  # (B, T, common_dim)

        # Reshape for multi-head attention
        head_dim = self.common_dim // self.num_heads
        Q = Q.view(B, T, self.num_heads, head_dim).transpose(1, 2)  # (B, H, T, d)
        K = K.view(B, T, self.num_heads, head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim ** 0.5)  # (B, H, T, T)

        if mask is not None:
            # Expand mask for broadcasting: (B, 1, 1, T)
            expanded_mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~expanded_mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)  # (B, H, T, d)
        context = context.transpose(1, 2).contiguous().view(B, T, self.common_dim)
        output = self.out_proj(context)

        # Residual connection + layer norm
        return self.layer_norm(hand_features + output)


class TemporalSelfAttention(nn.Module):
    """Self-attention over the time dimension to pool into a fixed-size vector.

    Learns which frames are most discriminative (high weight on hold phase,
    low weight on transition frames).
    """

    def __init__(self, input_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(input_dim)

        # Learnable [CLS] token for sequence pooling
        self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim) * 0.02)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, T, D) — temporal feature sequence
        mask : (batch, T) — True for valid positions

        Returns
        -------
        (batch, D) — single vector summarizing the sequence
        """
        B, T, D = x.shape

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x_with_cls = torch.cat([cls_tokens, x], dim=1)  # (B, T+1, D)

        # Extend mask to include CLS token (always valid)
        if mask is not None:
            cls_mask = torch.ones(B, 1, dtype=torch.bool, device=mask.device)
            mask = torch.cat([cls_mask, mask], dim=1)  # (B, T+1)
            # MultiheadAttention key_padding_mask: True = IGNORE
            key_padding_mask = ~mask
        else:
            key_padding_mask = None

        attended, _ = self.attention(
            x_with_cls, x_with_cls, x_with_cls,
            key_padding_mask=key_padding_mask,
        )
        attended = self.layer_norm(attended + x_with_cls)

        # Extract CLS token output as the sequence representation
        return attended[:, 0, :]  # (B, D)


class MultiModalTemporalEncoder(nn.Module):
    """Complete MTAN-SLR temporal encoder.

    Architecture:
        Input (B, T, 454)
            ├── Split → Hand (B, T, 126), Pose (B, T, 100), Velocity (B, T, 226), Flags (B, T, 2)
            ├── Hand BiLSTM → (B, T, 128)
            ├── Pose BiLSTM → (B, T, 64)
            ├── Cross-Modal Attention(hand, pose) → (B, T, 128)
            ├── Concatenate [attended_hand, pose] → (B, T, 192)
            ├── Temporal Self-Attention → (B, 192)  [CLS pooling]
            └── Classification Head → (B, num_classes)
    """

    def __init__(
        self,
        num_classes: int,
        hand_input_dim: int = 126,
        pose_input_dim: int = 100,
        velocity_dim: int = 226,
        flags_dim: int = 2,
        hand_hidden_dim: int = 64,
        pose_hidden_dim: int = 32,
        hand_num_layers: int = 2,
        pose_num_layers: int = 1,
        hand_dropout: float = 0.3,
        pose_dropout: float = 0.2,
        cross_attn_heads: int = 4,
        cross_attn_dropout: float = 0.1,
        temporal_attn_heads: int = 4,
        temporal_attn_dropout: float = 0.1,
        classifier_hidden: int = 128,
        classifier_dropout: float = 0.3,
    ):
        super().__init__()
        self.hand_input_dim = hand_input_dim
        self.pose_input_dim = pose_input_dim
        self.velocity_dim = velocity_dim
        self.flags_dim = flags_dim

        # Dimensionality after BiLSTM (bidirectional doubles the hidden dim)
        hand_out_dim = hand_hidden_dim * 2  # 128
        pose_out_dim = pose_hidden_dim * 2  # 64

        # ---- Modality-specific BiLSTMs ----
        # The velocity features (226 dims) are split and concatenated with
        # their respective modality streams.
        # Hand stream input: hand(126) + hand_velocity(126) + flags(2) = 254
        # Pose stream input: pose(100) + pose_velocity(100) = 200
        hand_lstm_input = hand_input_dim + hand_input_dim + flags_dim  # 254
        pose_lstm_input = pose_input_dim + pose_input_dim              # 200

        self.hand_bilstm = nn.LSTM(
            input_size=hand_lstm_input,
            hidden_size=hand_hidden_dim,
            num_layers=hand_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=hand_dropout if hand_num_layers > 1 else 0,
        )

        self.pose_bilstm = nn.LSTM(
            input_size=pose_lstm_input,
            hidden_size=pose_hidden_dim,
            num_layers=pose_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=pose_dropout if pose_num_layers > 1 else 0,
        )

        # ---- Cross-Modal Attention ----
        self.cross_attention = CrossModalAttention(
            hand_dim=hand_out_dim,
            pose_dim=pose_out_dim,
            num_heads=cross_attn_heads,
            dropout=cross_attn_dropout,
        )

        # ---- Temporal Self-Attention ----
        fused_dim = hand_out_dim + pose_out_dim  # 192
        self.temporal_attention = TemporalSelfAttention(
            input_dim=fused_dim,
            num_heads=temporal_attn_heads,
            dropout=temporal_attn_dropout,
        )

        # ---- Classification Head ----
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden, num_classes),
        )

        # Store output dimension for external use
        self.feature_dim = fused_dim

    def _split_features(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split the 454-dim feature vector into modality streams.

        Input layout: [left_hand(63) | right_hand(63) | pose(100) |
                       left_hand_vel(63) | right_hand_vel(63) | pose_vel(100) |
                       flags(2)]

        Returns: hand_features, pose_features, hand_velocity, pose_velocity, flags
        """
        hand = x[:, :, :self.hand_input_dim]                     # (B, T, 126)
        pose = x[:, :, self.hand_input_dim:
                       self.hand_input_dim + self.pose_input_dim]  # (B, T, 100)

        vel_start = self.hand_input_dim + self.pose_input_dim     # 226
        hand_vel = x[:, :, vel_start:vel_start + self.hand_input_dim]  # (B, T, 126)
        pose_vel = x[:, :, vel_start + self.hand_input_dim:
                           vel_start + self.hand_input_dim + self.pose_input_dim]  # (B, T, 100)

        flags = x[:, :, -self.flags_dim:]  # (B, T, 2)

        return hand, pose, hand_vel, pose_vel, flags

    def encode(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode a sequence into a fixed-size feature vector.

        Useful for downstream tasks (e.g., feeding to the segmenter).

        Parameters
        ----------
        x : (batch, T, 454)
        mask : (batch, T), optional

        Returns
        -------
        (batch, 192) — sequence-level feature vector
        """
        hand, pose, hand_vel, pose_vel, flags = self._split_features(x)

        # Concatenate each modality with its velocity stream
        hand_input = torch.cat([hand, hand_vel, flags], dim=-1)  # (B, T, 254)
        pose_input = torch.cat([pose, pose_vel], dim=-1)         # (B, T, 200)

        # BiLSTM encoding
        hand_encoded, _ = self.hand_bilstm(hand_input)  # (B, T, 128)
        pose_encoded, _ = self.pose_bilstm(pose_input)  # (B, T, 64)

        # Cross-modal attention: hand queries pose
        hand_attended = self.cross_attention(hand_encoded, pose_encoded, mask)

        # Fuse modalities
        fused = torch.cat([hand_attended, pose_encoded], dim=-1)  # (B, T, 192)

        # Temporal self-attention pooling
        pooled = self.temporal_attention(fused, mask)  # (B, 192)

        return pooled

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Full forward pass: sequence → class logits.

        Parameters
        ----------
        x : (batch, T, 454)
        mask : (batch, T), optional — True for valid (non-padding) frames

        Returns
        -------
        (batch, num_classes) logits (pre-softmax)
        """
        pooled = self.encode(x, mask)
        logits = self.classifier(pooled)
        return logits

    def get_attention_weights(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract attention weights for interpretability / visualization.

        Returns
        -------
        cross_attn_weights : attention weights from cross-modal attention
        temporal_attn_weights : attention weights from temporal self-attention
        """
        # This is a diagnostic method — not optimised for speed
        hand, pose, hand_vel, pose_vel, flags = self._split_features(x)

        hand_input = torch.cat([hand, hand_vel, flags], dim=-1)
        pose_input = torch.cat([pose, pose_vel], dim=-1)

        hand_encoded, _ = self.hand_bilstm(hand_input)
        pose_encoded, _ = self.pose_bilstm(pose_input)

        # Get cross-modal attention weights
        B, T, _ = hand_encoded.shape
        Q = self.cross_attention.q_proj(hand_encoded)
        K = self.cross_attention.k_proj(pose_encoded)
        head_dim = self.cross_attention.common_dim // self.cross_attention.num_heads
        Q = Q.view(B, T, self.cross_attention.num_heads, head_dim).transpose(1, 2)
        K = K.view(B, T, self.cross_attention.num_heads, head_dim).transpose(1, 2)
        cross_scores = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim ** 0.5)
        cross_attn_weights = F.softmax(cross_scores, dim=-1)

        # For temporal attention, use the built-in hook (simplified)
        hand_attended = self.cross_attention(hand_encoded, pose_encoded, mask)
        fused = torch.cat([hand_attended, pose_encoded], dim=-1)
        cls_tokens = self.temporal_attention.cls_token.expand(B, -1, -1)
        x_with_cls = torch.cat([cls_tokens, fused], dim=1)
        _, temporal_attn_weights = self.temporal_attention.attention(
            x_with_cls, x_with_cls, x_with_cls,
        )

        return cross_attn_weights, temporal_attn_weights
