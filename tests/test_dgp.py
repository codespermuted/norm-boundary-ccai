"""G2 acceptance: the DGP's lambda knob must equal empirical R^2_level."""

import numpy as np
import pytest

from src.synth.dgp import generate_series, r2_level, validate_lambda


@pytest.mark.parametrize("lam", [0.0, 0.3, 0.7, 1.0])
def test_lambda_matches_r2_level(lam):
    mean_r2, ok = validate_lambda(lam, seeds=range(5))
    assert ok, f"lam={lam}: mean R2_level={mean_r2:.3f} deviates > 0.08"


def test_r2_level_monotone_in_lambda():
    r2s = [
        np.mean([r2_level(generate_series(lam, s)) for s in range(3)])
        for lam in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]
    assert all(a < b for a, b in zip(r2s, r2s[1:])), r2s


def test_series_reproducible_and_shapes():
    s1 = generate_series(0.5, seed=3)
    s2 = generate_series(0.5, seed=3)
    np.testing.assert_array_equal(s1["y"], s2["y"])
    assert len(s1["y"]) == 20_000
    s3 = generate_series(0.5, seed=4)
    assert not np.array_equal(s1["y"], s3["y"])


def test_components_orthogonal_scale():
    """Level variance shares: window means of m should split ~lam/(1-lam)."""
    lam = 0.6
    s = generate_series(lam, seed=0)
    w = s["cfg"]["w_ref"]
    from src.synth.dgp import _rolling_mean

    mbar = _rolling_mean(s["m"], w)
    assert abs(mbar.var() - 1.0) < 0.35, "window-mean level variance ~ 1"
