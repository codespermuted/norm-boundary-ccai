import torch
import torch.nn as nn


class RevIN(nn.Module):
    """Reversible Instance Normalization (Kim et al., ICLR 2022).

    norm:   subtract lookback-window mean, divide by window std (per instance,
            per channel), then apply learnable affine.
    denorm: invert affine, multiply by stored std, add stored mean.

    Input/output layout: (batch, time, channels). Statistics are computed on
    the time axis of the tensor passed with mode='norm' and reused verbatim
    for mode='denorm' — the window statistics are extrapolated over the whole
    horizon, which is exactly the mechanism whose failure conditions this
    project characterizes.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
        self._mean = None
        self._stdev = None

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            self._mean = x.mean(dim=1, keepdim=True).detach()
            self._stdev = torch.sqrt(
                x.var(dim=1, keepdim=True, unbiased=False) + self.eps
            ).detach()
            x = (x - self._mean) / self._stdev
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
            return x
        elif mode == "denorm":
            if self._mean is None:
                raise RuntimeError("denorm called before norm")
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
            return x * self._stdev + self._mean
        raise ValueError(f"mode must be 'norm' or 'denorm', got {mode!r}")
