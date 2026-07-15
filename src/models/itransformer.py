"""iTransformer (Liu et al., ICLR 2024) — faithful port.

Pinned source: github.com/thuml/iTransformer @ c2426e6
(model/iTransformer.py + layers/Transformer_EncDec.py). Each variate's whole
lookback series is one token (Linear L->d_model); a post-LN encoder with
Conv1d(k=1) FFN attends ACROSS variates; a linear projector maps d_model->h
per variate. `use_norm` reproduces the built-in Non-stationary-Transformer
instance norm — enabled only for the paper-number validation; the grid runs
use_norm=False with normalization handled externally by the arm modules
(identical treatment to every other backbone).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _EncoderLayer(nn.Module):
    """Official TSLib EncoderLayer: post-LN, Conv1d(k=1) FFN, ReLU."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout(a))
        y = self.dropout(self.act(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)


class ITransformer(nn.Module):
    def __init__(self, lookback: int, horizon: int, num_features: int,
                 d_model: int = 256, n_heads: int = 8, e_layers: int = 2,
                 d_ff: int = 256, dropout: float = 0.1,
                 use_norm: bool = False):
        super().__init__()
        self.horizon = horizon
        self.use_norm = use_norm
        self.embed = nn.Linear(lookback, d_model)
        self.emb_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [_EncoderLayer(d_model, n_heads, d_ff, dropout)
             for _ in range(e_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.projector = nn.Linear(d_model, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, C) -> (B, h, C)
        if self.use_norm:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(
                torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x = x / stdev
        tok = self.emb_dropout(self.embed(x.permute(0, 2, 1)))  # (B, C, d)
        for layer in self.layers:
            tok = layer(tok)
        out = self.projector(self.final_norm(tok)).permute(0, 2, 1)  # (B, h, C)
        if self.use_norm:
            out = out * stdev[:, 0].unsqueeze(1) + means[:, 0].unsqueeze(1)
        return out
