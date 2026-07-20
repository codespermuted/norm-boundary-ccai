"""LPS inference module (G7 Block E) — fast synthetic unit tests.

Small n, B=49, ridge first stage wherever LightGBM adds nothing to the
property under test (keeps the whole file well under 60s). One test pins the
permutation statistic to the official LightGBM LPS exactly.
"""

import numpy as np
import pytest

from src.theory.lps import lps
from src.theory.lps_inference import lambda_star_hat, mbb_ci, permutation_test

W = 8


@pytest.fixture
def st_lgbm(monkeypatch):
    """Single-thread LightGBM (same rationale as compute_lps_delta.py),
    scoped to the test via monkeypatch so the suite stays unpolluted."""
    import lightgbm as lgb

    orig = lgb.LGBMRegressor

    class _ST(orig):
        def __init__(self, **kw):
            kw.setdefault("n_jobs", 1)
            super().__init__(**kw)

    monkeypatch.setattr(lgb, "LGBMRegressor", _ST)


def _strong(n=60, w=W, noise=0.1, seed=3):
    """Window-level signal: covariate explains the window mean almost fully."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=n)
    X = np.repeat(base, w)[:, None]
    y = np.repeat(base, w) + noise * rng.normal(size=n * w)
    return y, X


def _noise(n=60, w=W, seed=11):
    rng = np.random.default_rng(seed)
    return rng.normal(size=n * w), rng.normal(size=(n * w, 1))


def test_statistic_matches_official_lps(st_lgbm):
    """The permutation/bootstrap statistic must BE the official LPS."""
    y, X = _strong(n=40)
    official = lps(y, X, W, model="lgbm")["lps"]
    perm = permutation_test(y, X, W, B=5, model="lgbm", seed=0)
    boot = mbb_ci(y, X, W, B=5, model="lgbm", seed=0)
    assert perm["lps"] == pytest.approx(official, abs=1e-12)
    assert boot["lps"] == pytest.approx(official, abs=1e-12)


def test_strong_signal_significant():
    y, X = _strong()
    res = permutation_test(y, X, W, B=49, model="ridge", seed=0)
    assert res["lps"] > 0.9
    assert res["p_value"] < 0.05
    assert (res["null"] < res["lps"]).all()


def test_strong_signal_ci_excludes_zero():
    y, X = _strong()
    res = mbb_ci(y, X, W, B=49, model="ridge", seed=0)
    assert res["ci_lo"] > 0
    assert res["ci_lo"] <= res["lps"] <= res["ci_hi"]
    assert res["block_len"] == int(np.ceil(60 ** (1 / 3)))


def test_pure_noise_not_significant():
    y, X = _noise()
    res = permutation_test(y, X, W, B=49, model="ridge", seed=0)
    assert res["p_value"] > 0.2
    assert abs(res["lps"]) < 0.3, "pure noise should have LPS near 0"


def test_aligned_option_restricts_offsets():
    y, X = _noise(n=40)
    res = permutation_test(y, X, W, B=49, model="ridge", align_windows=7,
                           seed=0)
    # k in {7, 14, 21, 28, 35}: multiples of 7 below n_windows=40, all
    # enumerated because B >= count
    assert res["n_windows"] == 40
    assert (res["offsets"] % 7 == 0).all()
    assert res["B_effective"] == len(res["offsets"]) == 5
    # unaligned with B > n-1 enumerates every shift 1..n-1
    full = permutation_test(y, X, W, B=49, model="ridge", seed=0)
    assert full["B_effective"] == 39
    assert (np.sort(full["offsets"]) == np.arange(1, 40)).all()
    # B below the offset count subsamples without replacement
    sub = permutation_test(y, X, W, B=10, model="ridge", seed=0)
    assert sub["B_effective"] == 10
    assert len(np.unique(sub["offsets"])) == 10


def test_multichannel_statistic_is_channel_mean():
    y1, X = _strong(seed=3)
    y2, _ = _noise(seed=5)
    Y = np.column_stack([y1, y2])
    res = permutation_test(Y, X, W, B=5, model="ridge", seed=0)
    per_ch = np.mean([lps(y1, X, W, model="ridge")["lps"],
                      lps(y2, X, W, model="ridge")["lps"]])
    assert res["lps"] == pytest.approx(per_ch, abs=1e-12)


def test_lambda_star_hat_components():
    y, X = _strong()
    res = lambda_star_hat(y, X, W, h=W, model="ridge")
    assert np.isfinite(res["lambda_star_hat"])
    assert res["k_windows"] == 1  # ceil(h/w) with h == w
    assert res["V_hat"] > 0
    assert res["sigma_est2_hat"] >= 0
    assert res["S_h_hat"] >= 0
    assert res["sigma_z2_over_w_hat"] >= 0
    assert lambda_star_hat(y, X, W, h=3 * W, model="ridge")["k_windows"] == 3


def test_lambda_star_hat_multichannel_mean():
    y1, X = _strong(seed=3)
    y2, _ = _noise(seed=5)
    Y = np.column_stack([y1, y2])
    res = lambda_star_hat(Y, X, W, h=W, model="ridge")
    singles = [lambda_star_hat(v, X, W, h=W, model="ridge")["lambda_star_hat"]
               for v in (y1, y2)]
    assert res["lambda_star_hat"] == pytest.approx(np.mean(singles), abs=1e-12)
    assert res["per_channel"] == pytest.approx(singles, abs=1e-12)
