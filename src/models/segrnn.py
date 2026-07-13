"""SegRNN (Lin et al., 2023, arXiv:2308.11200) — compact implementation.

Segment-wise iteration: the lookback is cut into segments of seg_len, each
linearly embedded and encoded by a GRU (channel-independent). Decoding is the
paper's parallel multi-step scheme: one learnable positional embedding per
output segment queries the encoder state, and a linear head emits seg_len
values per segment.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SegRNN(nn.Module):
    def __init__(self, lookback: int, horizon: int, num_features: int,
                 seg_len: int = 24, d_model: int = 512, dropout: float = 0.1):
        super().__init__()
        if lookback % seg_len or horizon % seg_len:
            raise ValueError("lookback and horizon must be multiples of seg_len")
        self.seg_len = seg_len
        self.horizon = horizon
        self.n_in = lookback // seg_len
        self.n_out = horizon // seg_len
        self.embed = nn.Sequential(nn.Linear(seg_len, d_model), nn.ReLU())
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.pos = nn.Parameter(torch.randn(self.n_out, d_model) * 0.02)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, seg_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, C)
        b, t, c = x.shape
        z = x.permute(0, 2, 1).reshape(b * c, self.n_in, self.seg_len)
        _, state = self.gru(self.embed(z))                    # (1, B*C, d)
        queries = self.pos.unsqueeze(0).expand(b * c, -1, -1)  # (B*C, n_out, d)
        dec, _ = self.gru(queries, state)
        out = self.head(dec).reshape(b * c, self.horizon)
        return out.reshape(b, c, self.horizon).permute(0, 2, 1)
