import numpy as np
import pytest
import torch

from src.models import build_backbone
from src.models.lgbm_dms import LgbmDMS, window_znorm


@pytest.mark.parametrize("name,kwargs", [
    ("rlinear", {}),
    ("patchtst", {"d_model": 32, "n_heads": 4, "e_layers": 1, "d_ff": 64}),
    ("segrnn", {"seg_len": 24, "d_model": 64}),
])
def test_backbone_shapes_and_grad(name, kwargs):
    torch.manual_seed(0)
    m = build_backbone(name, lookback=96, horizon=24, num_features=7, **kwargs)
    x = torch.randn(4, 96, 7, requires_grad=True)
    y = m(x)
    assert y.shape == (4, 24, 7)
    y.mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_lgbm_dms_fit_predict():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 48))
    beta = rng.normal(size=48)
    y = np.stack([x @ beta + rng.normal(0, 0.1, 300) for _ in range(4)], axis=1)
    model = LgbmDMS(horizon=4, n_estimators=40).fit(x[:200], y[:200])
    pred = model.predict(x[200:])
    assert pred.shape == (100, 4)
    mse = np.mean((pred - y[200:]) ** 2)
    assert mse < np.var(y[200:]), "must beat the unconditional mean"


def test_window_znorm_roundtrip():
    rng = np.random.default_rng(0)
    x = rng.normal(3, 2, size=(10, 48))
    xn, (mu, sd) = window_znorm(x)
    np.testing.assert_allclose(xn * sd + mu, x, atol=1e-8)
    np.testing.assert_allclose(xn.mean(axis=1), 0, atol=1e-8)
