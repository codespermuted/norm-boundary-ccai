import torch
import torch.nn as nn


class RLinear(nn.Module):
    """Linear backbone of RLinear (Li et al., 2023, arXiv:2305.10721).

    A single linear map lookback -> horizon shared across channels
    (channel-independent). Normalization (the "R" in RLinear) is applied
    outside via a norm module, so this class stays norm-agnostic and the
    same backbone serves RAW / RevIN / CondNorm variants.

    Input:  (batch, lookback, channels)
    Output: (batch, horizon, channels)
    """

    def __init__(
        self,
        lookback: int,
        horizon: int,
        num_features: int,
        individual: bool = False,
    ):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.num_features = num_features
        self.individual = individual
        if individual:
            self.linear = nn.ModuleList(
                [nn.Linear(lookback, horizon) for _ in range(num_features)]
            )
        else:
            self.linear = nn.Linear(lookback, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)  # (B, C, L)
        if self.individual:
            out = torch.stack(
                [self.linear[c](x[:, c, :]) for c in range(self.num_features)], dim=1
            )
        else:
            out = self.linear(x)  # (B, C, h)
        return out.permute(0, 2, 1)  # (B, h, C)


class NormWrapper(nn.Module):
    """norm(x) -> backbone -> denorm(y_hat). RAW uses the identity norm."""

    def __init__(self, backbone: nn.Module, norm: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.norm = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x, "norm")
        y = self.backbone(x)
        return self.norm(y, "denorm")
