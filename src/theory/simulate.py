"""Monte Carlo validation of the closed forms in closed_form.py.

Simulates model M1 exactly as specified in docs/theory_g1.md §1 and applies
the three restoration rules operationally, so any algebra slip in the closed
forms shows up as a >1% relative discrepancy (G1 acceptance criterion).
"""

from __future__ import annotations

import numpy as np

from src.theory.closed_form import TheoryParams, var_ybar


def mc_mse(p: TheoryParams, n: int = 2_000_000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)

    g_T = rng.normal(0, np.sqrt(p.lam * p.V), n)
    v_T = rng.normal(0, np.sqrt((1 - p.lam) * p.V), n)
    delta = rng.normal(0, p.sigma_delta, n)
    d_x = rng.normal(0, np.sqrt(p.s_x2h), n) if p.s_x2h > 0 else 0.0
    d_u = rng.normal(0, np.sqrt(p.h) * p.sigma_u, n)
    eps_bar = rng.normal(0, p.sigma_z / np.sqrt(p.w), n)
    e = rng.normal(0, p.sigma_est, n)
    zeta = rng.normal(0, p.sigma_eps, n)

    L = delta + g_T + v_T
    M = delta + (g_T - d_x) + (v_T - d_u)
    y_bar = M + eps_bar
    y = L + zeta
    m_hat = g_T + e

    return {
        "raw": float(np.mean((y - 0.0) ** 2)),
        "in": float(np.mean((y - y_bar) ** 2)),
        "cn_oracle": float(np.mean((y - g_T) ** 2)),
        "cn_est": float(np.mean((y - m_hat) ** 2)),
    }


def mc_prop1_gap(
    p: TheoryParams, kappa: float, n: int = 2_000_000, seed: int = 0,
    noise_std: float = 0.1,
) -> float:
    """Empirical excess MSE of the interaction-free OLS vs full OLS.

    DGP: y = a*y_bar + b*g + kappa*y_bar*g + noise. Fit both feature sets by
    OLS on half the sample, evaluate on the other half, return MSE difference.
    """
    rng = np.random.default_rng(seed)
    a, b = 0.7, 0.5

    g = rng.normal(0, np.sqrt(p.lam * p.V), n)
    # y_bar correlated with g: Cov(y_bar, g) = lam*V, Var(y_bar) = var_ybar(p)
    resid_var = var_ybar(p) - p.lam * p.V  # variance of y_bar orthogonal to g
    y_bar = g + rng.normal(0, np.sqrt(resid_var), n)
    y = a * y_bar + b * g + kappa * y_bar * g + rng.normal(0, noise_std, n)

    half = n // 2
    ones = np.ones(half)

    def ols_mse(features_tr, features_te):
        X_tr = np.column_stack([ones, *features_tr])
        X_te = np.column_stack([np.ones(n - half), *features_te])
        coef, *_ = np.linalg.lstsq(X_tr, y[:half], rcond=None)
        resid = y[half:] - X_te @ coef
        return float(np.mean(resid**2))

    mse_restricted = ols_mse(
        (y_bar[:half], g[:half]), (y_bar[half:], g[half:])
    )
    mse_full = ols_mse(
        (y_bar[:half], g[:half], (y_bar * g)[:half]),
        (y_bar[half:], g[half:], (y_bar * g)[half:]),
    )
    return mse_restricted - mse_full
