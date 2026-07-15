"""Minimal covariate extensions of the main-grid deep backbones for the
information-fair ablation. Design principle: preserve each architecture's
identity; inject covariates at its natural entry points only.

  PatchTSTCov: patch embedding sees [target patch ; covariate patches]
               (past covs); the flatten head additionally receives the
               FUTURE covariate block (lead-matched, known at origin).
  SegRNNCov:   segment embedding sees [target segment ; covariate segments];
               each decoder positional query is augmented with its own
               future-covariate segment.

These are OUR minimal design choices (no canonical covariate recipe exists
for channel-independent architectures) — disclosed as such in the paper.
Univariate target only (the exogenous datasets).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PatchTSTCov(nn.Module):
    def __init__(self, lookback: int, horizon: int, d_cov: int,
                 patch_len: int = 16, stride: int = 8, d_model: int = 128,
                 n_heads: int = 16, e_layers: int = 3, d_ff: int = 256,
                 dropout: float = 0.2):
        super().__init__()
        self.lookback, self.horizon = lookback, horizon
        self.patch_len, self.stride = patch_len, stride
        self.n_patches = (lookback - patch_len) // stride + 2
        self.embed = nn.Linear(patch_len * (1 + d_cov), d_model)
        self.pos = nn.Parameter(torch.randn(self.n_patches, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=e_layers)
        self.head = nn.Sequential(
            nn.Linear(self.n_patches * d_model + horizon * d_cov, horizon),
            nn.Dropout(dropout))

    def _patches(self, z):  # (B, L[, d]) -> (B, n_patches, patch_len[*d])
        if z.dim() == 2:
            z = z.unsqueeze(-1)
        b, t, d = z.shape
        z = torch.cat([z, z[:, -self.stride :]], dim=1)
        p = z.unfold(1, self.patch_len, self.stride)[:, : self.n_patches]
        return p.permute(0, 1, 3, 2).reshape(b, self.n_patches, -1)

    def forward(self, x_win, cov_past, cov_future):
        """x_win (B,L), cov_past (B,L,d), cov_future (B,h,d) -> (B,h)"""
        tok = torch.cat([self._patches(x_win), self._patches(cov_past)], dim=-1)
        hdn = self.encoder(self.embed(tok) + self.pos)
        flat = torch.cat([hdn.flatten(1), cov_future.flatten(1)], dim=1)
        return self.head(flat)


class SegRNNCov(nn.Module):
    def __init__(self, lookback: int, horizon: int, d_cov: int,
                 seg_len: int = 24, d_model: int = 512, dropout: float = 0.1):
        super().__init__()
        if lookback % seg_len or horizon % seg_len:
            raise ValueError("lookback/horizon must be multiples of seg_len")
        self.seg_len, self.horizon = seg_len, horizon
        self.n_in, self.n_out = lookback // seg_len, horizon // seg_len
        self.embed = nn.Sequential(
            nn.Linear(seg_len * (1 + d_cov), d_model), nn.ReLU())
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.pos = nn.Parameter(torch.randn(self.n_out, d_model) * 0.02)
        self.fut_proj = nn.Linear(seg_len * d_cov, d_model)
        self.head = nn.Sequential(nn.Dropout(dropout),
                                  nn.Linear(d_model, seg_len))

    def forward(self, x_win, cov_past, cov_future):
        b = x_win.shape[0]
        past = torch.cat([x_win.unsqueeze(-1), cov_past], dim=-1)
        segs = past.reshape(b, self.n_in, -1)
        _, state = self.gru(self.embed(segs))
        fut = cov_future.reshape(b, self.n_out, -1)
        queries = self.pos.unsqueeze(0) + self.fut_proj(fut)
        dec, _ = self.gru(queries, state)
        return self.head(dec).reshape(b, self.horizon)
