import numpy as np
import pytest

from src.norms.condnorm import CondNormTransform, first_stage_level


@pytest.fixture()
def series():
    rng = np.random.default_rng(0)
    T = 2000
    x = rng.normal(0, 1, (T, 2))
    level = 3.0 * x[:, 0] - 1.5 * x[:, 1]
    y = level + rng.normal(0, 0.3, T)
    return x, y, level


def test_invertibility(series):
    x, y, _ = series
    level = first_stage_level(x, y, train_end=1200)
    tr = CondNormTransform(y, level, train_end=1200)
    r = tr.transform(y)
    y_back = tr.inverse(r)
    np.testing.assert_allclose(y_back, y, rtol=0, atol=1e-9)


def test_inverse_at_positions(series):
    x, y, _ = series
    level = first_stage_level(x, y, train_end=1200)
    tr = CondNormTransform(y, level, train_end=1200)
    pos = np.array([[1500, 1501], [1600, 1601]])
    r = tr.transform(y)
    np.testing.assert_allclose(tr.inverse(r[pos], pos), y[pos], atol=1e-9)


def test_first_stage_no_leakage(series):
    """Fitting must not see rows >= train_end: corrupting the future must not
    change in-sample predictions."""
    x, y, _ = series
    m1 = first_stage_level(x, y, train_end=1200)
    y_corrupt = y.copy()
    y_corrupt[1200:] += 1000.0
    m2 = first_stage_level(x, y_corrupt, train_end=1200)
    np.testing.assert_allclose(m1[:1200], m2[:1200], atol=1e-12)


def test_first_stage_learns_level(series):
    x, y, level = series
    m = first_stage_level(x, y, train_end=1200)
    oos = slice(1200, None)
    r2 = 1 - np.var(level[oos] - m[oos]) / np.var(level[oos])
    assert r2 > 0.9, f"first stage failed to learn level (R2={r2:.3f})"
