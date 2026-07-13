"""G1 acceptance criteria:
  (1) closed form vs Monte Carlo relative error < 1% across a parameter grid
  (2) lambda_star behavior fixed by unit tests (monotonicity, crossover identity)
"""

import math
from dataclasses import replace

import numpy as np
import pytest

from src.theory.closed_form import (
    TheoryParams,
    dominance_grid,
    in_minus_cn_gap,
    lambda_star,
    lambda_star_numeric,
    mse_cn,
    mse_in,
    mse_raw,
    prop1_interaction_gap,
)
from src.theory.simulate import mc_mse, mc_prop1_gap

GRID = [
    TheoryParams(),
    TheoryParams(lam=0.0, h=24),
    TheoryParams(lam=1.0, h=336, sigma_delta=0.5),
    TheoryParams(lam=0.3, h=96, sigma_u=0.06, sigma_est=0.15, sigma_eps=0.2),
    TheoryParams(lam=0.8, h=336, w=336, s_x2h=0.2, sigma_delta=0.8),
    TheoryParams(lam=0.6, h=24, w=96, sigma_z=2.0, sigma_u=0.02),
]


@pytest.mark.parametrize("p", GRID)
def test_closed_form_matches_monte_carlo(p):
    mc = mc_mse(p, n=2_000_000, seed=42)
    cf = {
        "raw": mse_raw(p),
        "in": mse_in(p),
        "cn_oracle": mse_cn(p, oracle=True),
        "cn_est": mse_cn(p, oracle=False),
    }
    for k in cf:
        rel = abs(mc[k] - cf[k]) / cf[k]
        assert rel < 0.01, f"{k}: closed={cf[k]:.6f} mc={mc[k]:.6f} rel={rel:.4%}"


def test_lambda_star_crossover_identity():
    """At lam = lambda_star, MSE_CN-est == MSE_IN exactly."""
    p = TheoryParams(h=96, sigma_u=0.06, sigma_est=0.1, sigma_delta=0.3)
    ls = lambda_star(p)
    assert 0 < ls < 1, f"choose params so crossover is interior, got {ls}"
    q = replace(p, lam=ls)
    assert math.isclose(mse_cn(q), mse_in(q), rel_tol=1e-12)


def test_lambda_star_numeric_agrees_with_closed_form():
    p = TheoryParams(h=96, sigma_u=0.06, sigma_est=0.1, sigma_delta=0.3)
    assert math.isclose(lambda_star(p), lambda_star_numeric(p), abs_tol=1e-8)


def test_lambda_star_monotone_decreasing_in_h():
    """Proposition 3 corollary: longer horizon -> CN dominates from smaller lam."""
    ps = [TheoryParams(h=h, sigma_u=0.06) for h in (24, 96, 336)]
    stars = [lambda_star(p) for p in ps]
    assert stars[0] > stars[1] > stars[2]


def test_lambda_star_monotone_increasing_in_drift():
    """More covariate-orthogonal drift -> IN region grows."""
    stars = [
        lambda_star(TheoryParams(h=96, sigma_u=0.06, sigma_delta=sd))
        for sd in (0.0, 0.3, 0.6)
    ]
    assert stars[0] < stars[1] < stars[2]


def test_in_cn_gap_monotone_in_h():
    """Proposition 3: MSE_IN(h) - MSE_CN(h) increases with h."""
    gaps = [
        in_minus_cn_gap(TheoryParams(lam=0.7, h=h, sigma_u=0.06))
        for h in (24, 96, 336)
    ]
    assert gaps[0] < gaps[1] < gaps[2]


def test_dominance_grid_corners():
    p = TheoryParams(h=96, sigma_u=0.06, sigma_est=math.sqrt(0.02))
    lam_grid = np.linspace(0, 1, 21)
    drift2_grid = np.linspace(0, 1.5, 16)
    labels = dominance_grid(lam_grid, drift2_grid, p)  # 0=RAW 1=IN 2=CN
    assert labels[-1, 0] == 1, "high drift + lam=0 must be IN"
    assert labels[0, -1] == 2, "no drift + lam=1 must be CN"


def test_prop1_gap_closed_form_vs_mc():
    p = TheoryParams(lam=0.6, h=24)
    kappa = 0.5
    cf = prop1_interaction_gap(p, kappa)
    mc = mc_prop1_gap(p, kappa, n=2_000_000, seed=7)
    assert abs(mc - cf) / cf < 0.01, f"closed={cf:.6f} mc={mc:.6f}"


def test_prop1_gap_zero_without_interaction():
    p = TheoryParams(lam=0.6)
    assert prop1_interaction_gap(p, kappa=0.0) == 0.0
