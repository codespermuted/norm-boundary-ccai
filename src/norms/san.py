"""SAN — Slice-based Adaptive Normalization (Liu et al., NeurIPS 2023).

Ported from the official implementation, pinned commit:
  https://github.com/icantnamemyself/SAN @ 7e1ca66251a91a89290846b310145c5f5db3ffc3
  (models/Statistics_prediction.py, station_type='adaptive')

Faithful adaptation to this repo's forward(x, mode) interface:
  - 'norm'  : slice-normalize the lookback, predict future slice statistics
              with the MLP pair, store them for denormalization
  - 'denorm': rescale backbone output with the PREDICTED future statistics

Training protocol (as in official exp_main.py): the statistics modules are
pretrained for a few epochs on stats_loss() with their own optimizer, then
frozen while the backbone trains. src/train.py implements this when
norm.requires_pretrain is True.

SAN is NOT an invertible transform: denorm(norm(x)) != x by design (the
restoration uses predicted statistics). The reconstruction identity that
must hold instead is tested in tests/test_san_fan.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _SanMLP(nn.Module):
    """Official SAN MLP (512/1024 hidden), predicting future slice stats."""

    def __init__(self, seq_slices: int, lookback: int, pred_slices: int, mode: str):
        super().__init__()
        self.mode = mode
        self.input = nn.Linear(seq_slices, 512)
        self.input_raw = nn.Linear(lookback, 512)
        self.activation = nn.ReLU() if mode == "std" else nn.Tanh()
        self.final_activation = nn.ReLU() if mode == "std" else nn.Identity()
        self.output = nn.Linear(1024, pred_slices)

    def forward(self, x_slice: torch.Tensor, x_raw: torch.Tensor) -> torch.Tensor:
        x_slice = x_slice.permute(0, 2, 1)
        x_raw = x_raw.permute(0, 2, 1)
        h = torch.cat([self.input(x_slice), self.input_raw(x_raw)], dim=-1)
        out = self.final_activation(self.output(self.activation(h)))
        return out.permute(0, 2, 1)


class SAN(nn.Module):
    requires_pretrain = True

    def __init__(self, num_features: int, lookback: int, horizon: int,
                 period_len: int = 24, eps: float = 1e-5):
        super().__init__()
        if lookback % period_len or horizon % period_len:
            raise ValueError("lookback and horizon must be multiples of period_len")
        self.num_features = num_features
        self.period_len = period_len
        self.seq_slices = lookback // period_len
        self.pred_slices = horizon // period_len
        self.eps = eps
        self.mean_model = _SanMLP(self.seq_slices, lookback, self.pred_slices, "mean")
        self.std_model = _SanMLP(self.seq_slices, lookback, self.pred_slices, "std")
        self.weight = nn.Parameter(torch.ones(2, num_features))
        self._station_pred = None

    def stats_parameters(self):
        yield from self.mean_model.parameters()
        yield from self.std_model.parameters()
        yield self.weight

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            b, t, c = x.shape
            sliced = x.reshape(b, -1, self.period_len, c)
            mean = sliced.mean(dim=-2, keepdim=True)
            std = sliced.std(dim=-2, keepdim=True)
            norm = ((sliced - mean) / (std + self.eps)).reshape(b, t, c)
            mean_all = x.mean(dim=1, keepdim=True)
            pred_mean = (self.mean_model(mean.squeeze(2) - mean_all, x - mean_all)
                         * self.weight[0] + mean_all * self.weight[1])
            pred_std = self.std_model(std.squeeze(2), x)
            self._station_pred = torch.cat([pred_mean, pred_std], dim=-1)
            return norm
        elif mode == "denorm":
            if self._station_pred is None:
                raise RuntimeError("denorm before norm")
            b, t, c = x.shape
            sliced = x.reshape(b, -1, self.period_len, c)
            mean = self._station_pred[:, :, : self.num_features].unsqueeze(2)
            std = self._station_pred[:, :, self.num_features :].unsqueeze(2)
            return (sliced * (std + self.eps) + mean).reshape(b, t, c)
        raise ValueError(f"mode must be 'norm' or 'denorm', got {mode!r}")

    def stats_loss(self, y_true: torch.Tensor) -> torch.Tensor:
        """MSE between predicted and true future slice statistics
        (official exp_main.station_loss)."""
        b, t, c = y_true.shape
        sliced = y_true.reshape(b, -1, self.period_len, c)
        target = torch.cat([sliced.mean(dim=-2), sliced.std(dim=-2)], dim=-1)
        return nn.functional.mse_loss(self._station_pred, target)
