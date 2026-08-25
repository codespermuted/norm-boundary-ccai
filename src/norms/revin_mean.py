import torch
import torch.nn as nn


class RevINMean(nn.Module):
    """RevIN with the *mean channel only* — the mechanism ablation (G9).

    RevIN removes three things at once: the lookback window mean, the window
    scale, and a learnable affine. The theory in the paper attributes the cost
    of the layer to the first of those (a frozen window mean is a persistence
    forecast of the level, stale exactly at a ramp), but toggling RevIN toggles
    all three, so the attribution is motivated rather than isolated.

    This module isolates it:

        norm:   x - mean(x over the lookback window)      (per instance, per channel)
        denorm: x + that same stored mean

    No instance scale, no affine. The series it is applied to has already been
    globally z-scored on train statistics, so scale handling here is identical
    to the RAW arm and the only difference from RAW is the mean channel.

    That gives the decomposition
        RevIN - RAW           = total layer cost      (Table 1)
        RevINMean - RAW       = mean channel alone
        RevIN - RevINMean     = scale + affine

    Layout is (batch, time, channels), matching src/norms/revin.py, and the
    stored statistic is reused verbatim over the horizon — the staleness whose
    consequences this project characterizes.
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self._mean = None

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            self._mean = x.mean(dim=1, keepdim=True).detach()
            return x - self._mean
        elif mode == "denorm":
            if self._mean is None:
                raise RuntimeError("denorm called before norm")
            return x + self._mean
        raise ValueError(f"mode must be 'norm' or 'denorm', got {mode!r}")
