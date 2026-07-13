"""FAN — Frequency Adaptive Normalization (Ye et al., NeurIPS 2024).

Ported from the official implementation, pinned commit:
  https://github.com/wayne155/FAN @ 838e1b002aa0e8cbc3889dfb69967c40c0c15761
  (torch_timeseries/normalizations/FAN.py)

Adaptation: mode names 'norm'/'denorm' (official 'n'/'d'); aux_loss(y) is the
official loss(true), added to the training objective by src/train.py.

FAN is NOT an invertible transform (restoration adds the PREDICTED main
frequency signal). The identity that must hold is the frequency decomposition
additivity: residual + filtered == x, tested in tests/test_san_fan.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def main_freq_part(x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    xf = torch.fft.rfft(x, dim=1)
    k = min(k, xf.shape[1])  # short horizons have fewer rfft bins than topk
    indices = torch.topk(xf.abs(), k, dim=1).indices
    mask = torch.zeros_like(xf)
    mask.scatter_(1, indices, 1)
    x_filtered = torch.fft.irfft(xf * mask, n=x.shape[1], dim=1).real.float()
    return x - x_filtered, x_filtered


class _MLPfreq(nn.Module):
    def __init__(self, lookback: int, horizon: int):
        super().__init__()
        self.model_freq = nn.Sequential(nn.Linear(lookback, 64), nn.ReLU())
        self.model_all = nn.Sequential(
            nn.Linear(64 + lookback, 128), nn.ReLU(), nn.Linear(128, horizon)
        )

    def forward(self, main_freq: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.model_all(torch.cat([self.model_freq(main_freq), x], dim=-1))


class FAN(nn.Module):
    requires_pretrain = False

    def __init__(self, num_features: int, lookback: int, horizon: int,
                 freq_topk: int = 20):
        super().__init__()
        self.freq_topk = freq_topk
        self.model_freq = _MLPfreq(lookback, horizon)
        self.weight = nn.Parameter(torch.ones(2, num_features))
        self._pred_main = None
        self._pred_residual = None

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            norm_input, x_filtered = main_freq_part(x, self.freq_topk)
            self._pred_main = self.model_freq(
                x_filtered.transpose(1, 2), x.transpose(1, 2)
            ).transpose(1, 2)
            return norm_input
        elif mode == "denorm":
            if self._pred_main is None:
                raise RuntimeError("denorm before norm")
            self._pred_residual = x
            return x + self._pred_main
        raise ValueError(f"mode must be 'norm' or 'denorm', got {mode!r}")

    def aux_loss(self, y_true: torch.Tensor) -> torch.Tensor:
        residual, true_main = main_freq_part(y_true, self.freq_topk)
        lf = nn.functional.mse_loss
        return lf(self._pred_main, true_main) + lf(self._pred_residual, residual)
