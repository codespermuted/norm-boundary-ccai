import torch

from src.models.itransformer import ITransformer
from src.models.timexer import TimeXer


def test_itransformer_shapes_grad():
    torch.manual_seed(0)
    m = ITransformer(lookback=96, horizon=24, num_features=7,
                     d_model=64, n_heads=4, e_layers=2, d_ff=64)
    x = torch.randn(4, 96, 7, requires_grad=True)
    y = m(x)
    assert y.shape == (4, 24, 7)
    y.mean().backward()
    assert torch.isfinite(x.grad).all()


def test_itransformer_use_norm_scale_invariance():
    """With built-in norm, adding a constant level shifts output by ~same."""
    torch.manual_seed(0)
    m = ITransformer(96, 24, 3, d_model=32, n_heads=4, e_layers=1, d_ff=32,
                     use_norm=True).eval()
    x = torch.randn(2, 96, 3)
    with torch.no_grad():
        delta = m(x + 100.0) - m(x)
    assert torch.allclose(delta, torch.full_like(delta, 100.0), atol=1e-2)


def test_timexer_ms_mode():
    torch.manual_seed(0)
    m = TimeXer(lookback=96, horizon=24, d_cov=2, patch_len=24,
                d_model=64, n_heads=4, e_layers=1, d_ff=128)
    x = torch.randn(4, 96, requires_grad=True)
    cov = torch.randn(4, 96, 2)
    y = m(x, cov)
    assert y.shape == (4, 24)
    y.mean().backward()
    assert torch.isfinite(x.grad).all()


def test_timexer_multi_mode():
    torch.manual_seed(0)
    m = TimeXer(lookback=96, horizon=24, patch_len=24, d_model=64,
                n_heads=4, e_layers=1, d_ff=128, use_norm=True)
    x = torch.randn(4, 96, 7)
    y = m.forward_multi(x)
    assert y.shape == (4, 24, 7)


def test_timexer_cross_attention_uses_covariates():
    """Changing covariates must change the prediction (exog path is live)."""
    torch.manual_seed(0)
    m = TimeXer(96, 24, d_cov=1, patch_len=24, d_model=64, n_heads=4,
                e_layers=1, d_ff=128).eval()
    x = torch.randn(2, 96)
    with torch.no_grad():
        y1 = m(x, torch.zeros(2, 96, 1))
        y2 = m(x, torch.ones(2, 96, 1) * 3)
    assert not torch.allclose(y1, y2)
