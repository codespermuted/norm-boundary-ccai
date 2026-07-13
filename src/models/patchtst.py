"""PatchTST (Nie et al., ICLR 2023) — compact channel-independent encoder.

Faithful to the paper's structure: per-channel patching (patch_len/stride),
linear patch embedding + learnable positional encoding, Transformer encoder,
flatten head. Deviation from the official repo (documented): LayerNorm
Transformer blocks via nn.TransformerEncoder instead of the custom BatchNorm
encoder. Normalization (RevIN etc.) stays OUTSIDE via NormWrapper, like every
backbone in this repo.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PatchTST(nn.Module):
    def __init__(self, lookback: int, horizon: int, num_features: int,
                 patch_len: int = 16, stride: int = 8, d_model: int = 128,
                 n_heads: int = 16, e_layers: int = 3, d_ff: int = 256,
                 dropout: float = 0.2):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = (lookback - patch_len) // stride + 2  # +1 pad patch
        self.embed = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.randn(self.n_patches, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=e_layers)
        self.head = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.n_patches * d_model, horizon),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, C)
        b, t, c = x.shape
        z = x.permute(0, 2, 1).reshape(b * c, t)          # CI: fold channels
        z = torch.cat([z, z[:, -self.stride:]], dim=1)    # replication pad
        patches = z.unfold(1, self.patch_len, self.stride)[:, : self.n_patches]
        h = self.embed(patches) + self.pos
        h = self.encoder(h)
        out = self.head(h)                                 # (B*C, horizon)
        return out.reshape(b, c, self.horizon).permute(0, 2, 1)
