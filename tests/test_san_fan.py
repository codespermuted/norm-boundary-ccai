"""SAN/FAN ports: structural identities + end-to-end trainability.

SAN and FAN are predictive normalizers, not invertible maps — the invariants
tested here are the ones that ARE exact:
  - SAN: slice-normalization reconstructs x with TRUE slice stats
  - FAN: frequency decomposition additivity (residual + main == x)
Plus: NormWrapper forward shapes and a few optimizer steps without NaN.
"""

import torch

from src.models.rlinear import NormWrapper, RLinear
from src.norms import build_norm
from src.norms.fan import main_freq_part


def test_san_slice_reconstruction():
    torch.manual_seed(0)
    san = build_norm("san", num_features=3, lookback=96, horizon=24, period_len=24)
    x = torch.randn(8, 96, 3) * 2 + 5
    xn = san(x, "norm")
    b, t, c = x.shape
    sliced = x.reshape(b, -1, 24, c)
    mean = sliced.mean(dim=-2, keepdim=True)
    std = sliced.std(dim=-2, keepdim=True)
    recon = (xn.reshape(b, -1, 24, c) * (std + san.eps) + mean).reshape(b, t, c)
    assert torch.allclose(recon, x, atol=1e-4)


def test_san_denorm_shape_and_stats_loss():
    san = build_norm("san", num_features=2, lookback=96, horizon=48, period_len=24)
    x, y = torch.randn(4, 96, 2), torch.randn(4, 48, 2)
    san(x, "norm")
    out = san(y, "denorm")
    assert out.shape == (4, 48, 2)
    loss = san.stats_loss(y)
    assert torch.isfinite(loss) and loss.item() >= 0


def test_fan_decomposition_additivity():
    torch.manual_seed(0)
    x = torch.randn(8, 96, 3)
    residual, main = main_freq_part(x, k=10)
    assert torch.allclose(residual + main, x, atol=1e-5)


def test_fan_norm_denorm_flow():
    fan = build_norm("fan", num_features=2, lookback=96, horizon=24, freq_topk=8)
    x, y = torch.randn(4, 96, 2), torch.randn(4, 24, 2)
    xn = fan(x, "norm")
    assert xn.shape == x.shape
    out = fan(y, "denorm")
    assert out.shape == y.shape
    assert torch.isfinite(fan.aux_loss(y))


def _few_steps(norm_name, **norm_kwargs):
    torch.manual_seed(0)
    norm = build_norm(norm_name, num_features=2, lookback=96, horizon=24,
                      **norm_kwargs)
    model = NormWrapper(RLinear(96, 24, 2), norm)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x, y = torch.randn(16, 96, 2), torch.randn(16, 24, 2)
    for _ in range(3):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        if hasattr(norm, "aux_loss"):
            loss = loss + norm.aux_loss(y)
        loss.backward()
        opt.step()
    assert torch.isfinite(loss)


def test_san_trains():
    _few_steps("san", period_len=24)


def test_fan_trains():
    _few_steps("fan", freq_topk=8)
