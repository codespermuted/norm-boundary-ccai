import torch

from src.norms import build_norm


def test_revin_invertibility():
    """denorm(norm(x)) must reconstruct x (invariant required by plan §5.2)."""
    torch.manual_seed(0)
    revin = build_norm("revin", num_features=7)
    x = torch.randn(32, 336, 7) * 5 + 3
    x_norm = revin(x, "norm")
    x_back = revin(x_norm, "denorm")
    assert torch.allclose(x_back, x, atol=1e-4), (x_back - x).abs().max()


def test_revin_norm_statistics():
    """After norm (pre-affine, weight=1/bias=0 at init) each window is ~zero-mean unit-var."""
    torch.manual_seed(0)
    revin = build_norm("revin", num_features=3)
    x = torch.randn(16, 96, 3) * 10 - 7
    x_norm = revin(x, "norm")
    assert torch.allclose(x_norm.mean(dim=1), torch.zeros(16, 3), atol=1e-5)
    assert torch.allclose(x_norm.std(dim=1, unbiased=False), torch.ones(16, 3), atol=1e-3)


def test_revin_horizon_shape_change():
    """denorm must work on a tensor with a different time length (h != L)."""
    revin = build_norm("revin", num_features=7)
    x = torch.randn(8, 336, 7)
    revin(x, "norm")
    y = torch.randn(8, 96, 7)
    out = revin(y, "denorm")
    assert out.shape == (8, 96, 7)


def test_revin_constant_window_stability():
    """Zero-variance windows must not produce NaN/Inf (eps guard)."""
    revin = build_norm("revin", num_features=2)
    x = torch.ones(4, 48, 2) * 42.0
    x_norm = revin(x, "norm")
    x_back = revin(x_norm, "denorm")
    assert torch.isfinite(x_norm).all() and torch.isfinite(x_back).all()
    assert torch.allclose(x_back, x, atol=1e-3)


def test_raw_is_identity():
    raw = build_norm("raw", num_features=7)
    x = torch.randn(4, 96, 7)
    assert torch.equal(raw(x, "norm"), x)
    assert torch.equal(raw(x, "denorm"), x)
