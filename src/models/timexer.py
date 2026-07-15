"""TimeXer (Wang et al., NeurIPS 2024) — faithful port.

Pinned source: github.com/thuml/TimeXer @ 7601190 (models/TimeXer.py).
Endogenous series -> non-overlapping patch tokens + a learnable GLOBAL token;
exogenous variates -> one whole-series token each (inverted embedding); each
encoder layer: self-attention over [patches, glb], cross-attention of the
GLOBAL token onto the exogenous tokens, Conv1d(k=1) FFN (post-LN throughout);
FlattenHead over (patch_num+1) x d_model -> horizon.

Two published modes, both ported:
  forward(x, cov)  'MS': endogenous = univariate target, exogenous = cov
                   series (the exogenous-forecasting setting of the paper;
                   PAST covariate windows only — the published model consumes
                   no future exogenous values)
  forward_multi(x) 'M' : every variate endogenous, all variates exogenous
                   (used for the ETTh1 paper-number validation)

use_norm reproduces the built-in instance norm for validation only; grid
runs use_norm=False with external arm normalization.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class _PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div = (torch.arange(0, d_model, 2).float()
               * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class _EnEmbedding(nn.Module):
    def __init__(self, d_model: int, patch_len: int, dropout: float):
        super().__init__()
        self.patch_len = patch_len
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, 1, 1, d_model))
        self.position_embedding = _PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # (B, n_vars, L) -> (B*n_vars, n_patch+1, d)
        b, n_vars, _ = x.shape
        glb = self.glb_token.repeat(b, n_vars, 1, 1)
        p = x.unfold(-1, self.patch_len, self.patch_len)      # (B,n,np,pl)
        p = p.reshape(b * n_vars, p.shape[2], self.patch_len)
        p = self.value_embedding(p) + self.position_embedding(p)
        p = p.reshape(b, n_vars, p.shape[-2], p.shape[-1])
        p = torch.cat([p, glb], dim=2)
        return self.dropout(p.reshape(b * n_vars, p.shape[2], p.shape[3])), n_vars


class _EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads,
                                               dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads,
                                                dropout=dropout, batch_first=True)
        self.conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x, cross):
        """x: (B*n_vars, np+1, d) — last token is GLOBAL; cross: (B, n_ex, d)"""
        B = cross.shape[0]
        a, _ = self.self_attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout(a))

        glb_ori = x[:, -1:, :]                       # (B*n, 1, d)
        glb = glb_ori.reshape(B, -1, x.shape[-1])    # (B, n, d)
        ca, _ = self.cross_attn(glb, cross, cross, need_weights=False)
        glb = self.norm2(glb_ori + self.dropout(ca).reshape(-1, 1, x.shape[-1]))

        y = x = torch.cat([x[:, :-1, :], glb], dim=1)
        y = self.dropout(self.act(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm3(x + y)


class _FlattenHead(nn.Module):
    def __init__(self, nf, horizon, dropout):
        super().__init__()
        self.linear = nn.Linear(nf, horizon)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z):  # (B, n_vars, d, np+1)
        return self.dropout(self.linear(z.flatten(-2)))


class TimeXer(nn.Module):
    def __init__(self, lookback: int, horizon: int, d_cov: int = 0,
                 patch_len: int = 24, d_model: int = 256, n_heads: int = 8,
                 e_layers: int = 1, d_ff: int = 1024, dropout: float = 0.1,
                 use_norm: bool = False):
        super().__init__()
        if lookback % patch_len:
            raise ValueError("lookback must be a multiple of patch_len")
        self.horizon = horizon
        self.use_norm = use_norm
        self.patch_num = lookback // patch_len
        self.en_embedding = _EnEmbedding(d_model, patch_len, dropout)
        self.ex_embedding = nn.Linear(lookback, d_model)
        self.ex_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [_EncoderLayer(d_model, n_heads, d_ff, dropout)
             for _ in range(e_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.head = _FlattenHead(d_model * (self.patch_num + 1), horizon,
                                 dropout)

    def _encode(self, en_vars, ex_vars):
        """en_vars (B, n_en, L), ex_vars (B, n_ex, L)"""
        x, n_vars = self.en_embedding(en_vars)
        cross = self.ex_dropout(self.ex_embedding(ex_vars))
        for layer in self.layers:
            x = layer(x, cross)
        x = self.final_norm(x)
        z = x.reshape(-1, n_vars, x.shape[-2], x.shape[-1]).permute(0, 1, 3, 2)
        return self.head(z)  # (B, n_vars, h)

    def forward(self, x: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
        """'MS' mode. x (B, L) target, cov (B, L, d) -> (B, h)"""
        out = self._encode(x.unsqueeze(1), cov.permute(0, 2, 1))
        return out[:, 0]

    def forward_multi(self, x: torch.Tensor) -> torch.Tensor:
        """'M' mode. x (B, L, C) -> (B, h, C) (validation vs paper numbers)"""
        if self.use_norm:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(
                torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x = x / stdev
        v = x.permute(0, 2, 1)                       # (B, C, L)
        out = self._encode(v, v).permute(0, 2, 1)    # (B, h, C)
        if self.use_norm:
            out = out * stdev[:, 0].unsqueeze(1) + means[:, 0].unsqueeze(1)
        return out
