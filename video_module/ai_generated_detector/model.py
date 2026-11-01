"""
model.py
========
BiGRU classification head for AI-generated video detection.

Input:  (batch, num_frames, feature_dim) — pre-extracted EfficientNet-B0 features.
Output: (batch, 2)                        — logits for [REAL, AI_GENERATED].
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import GRU_HIDDEN, GRU_LAYERS, GRU_BIDIR, DROPOUT, NUM_CLASSES, FEATURE_DIM


class BiGRUHead(nn.Module):
    """
    Bidirectional GRU over frame-level features → binary classifier.

    Architecture
    ------------
    Input (B, T, F) — B=batch, T=frames, F=feature_dim
      → BiGRU(hidden=256, bidirectional) → last hidden concat (B, 512)
      → LayerNorm → Dropout → Linear(512, 2)
      → logits (B, 2)
    """

    def __init__(
        self,
        feature_dim: int   = FEATURE_DIM,
        hidden_dim:  int   = GRU_HIDDEN,
        num_layers:  int   = GRU_LAYERS,
        bidirectional: bool = GRU_BIDIR,
        dropout:     float = DROPOUT,
        num_classes: int   = NUM_CLASSES,
    ) -> None:
        super().__init__()
        self.hidden_dim    = hidden_dim
        self.bidirectional = bidirectional
        dir_factor         = 2 if bidirectional else 1
        out_dim            = hidden_dim * dir_factor

        self.gru = nn.GRU(
            input_size   = feature_dim,
            hidden_size  = hidden_dim,
            num_layers   = num_layers,
            batch_first  = True,
            bidirectional= bidirectional,
            dropout      = dropout if num_layers > 1 else 0.0,
        )
        self.norm      = nn.LayerNorm(out_dim)
        self.drop      = nn.Dropout(dropout)
        self.classifier= nn.Linear(out_dim, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T, F)
        returns: logits (B, num_classes)
        """
        # GRU output: (B, T, dir_factor*hidden)
        out, _ = self.gru(x)
        # Use the last time-step output
        pooled = out[:, -1, :]               # (B, dir_factor*hidden)
        pooled = self.norm(pooled)
        pooled = self.drop(pooled)
        logits = self.classifier(pooled)     # (B, 2)
        return logits


def build_model(
    feature_dim: int   = FEATURE_DIM,
    hidden_dim:  int   = GRU_HIDDEN,
    dropout:     float = DROPOUT,
) -> BiGRUHead:
    """Factory function to build and return the classification head."""
    return BiGRUHead(
        feature_dim   = feature_dim,
        hidden_dim    = hidden_dim,
        bidirectional = GRU_BIDIR,
        dropout       = dropout,
        num_classes   = NUM_CLASSES,
    )
