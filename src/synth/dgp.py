"""Synthetic DGP (RESEARCH_PLAN.md §4.1), theory-faithful to model M1.

    x_t = a x_{t-1} + eps_t                      exogenous covariate, AR(1)
    m_t = sqrt(lam) * x_l_t + sqrt(1-lam) * u_l_t   level (covariate + random walk)
    z_t = A sin(2 pi t / P) + rho z_{t-1} + nu_t    quasi-stationary shape
    y_t = m_t + sigma0 * z_t

Two normalization details make lambda interpretable as R^2_level by
construction (docs/theory_g1.md §5):

1. x and u are standardized over the sample, then rescaled so that their
   *w_ref-window means* have unit variance. R^2_level is defined on window
   means (plan §2.1), and plain pointwise standardization would shrink the
   AR(1) component's window-mean variance far more than the random walk's,
   silently decoupling lambda from R^2_level.
2. w_ref is a multiple of P so the seasonal shape averages out of window
   means. validate_lambda() measures R^2_level empirically; tests assert the
   match (G2 acceptance: self-validation included).

The covariate exposed to the CN first stage is x (standardized); m depends on
x only through a deterministic rescale, so g(x) is learnable, and the u part
is never observable from covariates — exactly M1's g/v split.
"""

from __future__ import annotations

import numpy as np

DEFAULTS = dict(T=20_000, a=0.9, P=24, A=1.0, rho=0.5, sigma0=0.5,
                w_ref=96, burnin=1_000)


def _rolling_mean(arr: np.ndarray, w: int) -> np.ndarray:
    c = np.cumsum(np.insert(arr, 0, 0.0))
    return (c[w:] - c[:-w]) / w


def _window_scale(arr: np.ndarray, w: int) -> np.ndarray:
    """Rescale so that w-window means of the result have unit variance."""
    std = _rolling_mean(arr, w).std()
    return arr / std


def generate_series(lam: float, seed: int, **kwargs) -> dict:
    cfg = {**DEFAULTS, **kwargs}
    T, a, P, A, rho, sigma0, w_ref, burnin = (
        cfg["T"], cfg["a"], cfg["P"], cfg["A"], cfg["rho"], cfg["sigma0"],
        cfg["w_ref"], cfg["burnin"],
    )
    rng = np.random.default_rng([seed, int(round(lam * 100))])

    eps = rng.normal(0, 1, T + burnin)
    x = np.empty(T + burnin)
    x[0] = eps[0]
    for t in range(1, T + burnin):
        x[t] = a * x[t - 1] + eps[t]
    x = x[burnin:]
    x = (x - x.mean()) / x.std()

    u = np.cumsum(rng.normal(0, 1, T))
    u = (u - u.mean()) / u.std()

    x_l = _window_scale(x, w_ref)
    u_l = _window_scale(u, w_ref)
    m = np.sqrt(lam) * x_l + np.sqrt(1 - lam) * u_l

    nu = rng.normal(0, 1, T)
    z = np.empty(T)
    z[0] = nu[0]
    for t in range(1, T):
        z[t] = rho * z[t - 1] + nu[t]
    z = z + A * np.sin(2 * np.pi * np.arange(T) / P)

    y = m + sigma0 * z
    return {"y": y, "m": m, "x": x, "z": z, "lam": lam, "seed": seed,
            "sigma0": sigma0, "cfg": cfg}


def r2_level(series: dict, w: int | None = None) -> float:
    """Empirical R^2_level: OLS of w-window means of y on w-window means of x.

    Linear first stage suffices here because m is linear in x by construction;
    on real data the analogous quantity is LPS with a nonlinear g (plan §3.1).
    """
    w = w or series["cfg"]["w_ref"]
    ybar = _rolling_mean(series["y"], w)
    xbar = _rolling_mean(series["x"], w)
    X = np.column_stack([np.ones_like(xbar), xbar])
    coef, *_ = np.linalg.lstsq(X, ybar, rcond=None)
    resid = ybar - X @ coef
    return 1 - resid.var() / ybar.var()


def validate_lambda(lam: float, seeds=range(5), tol: float = 0.08, **kwargs):
    """Mean |R^2_level - lam| over seeds; returns (mean_r2, ok)."""
    r2s = [r2_level(generate_series(lam, s, **kwargs)) for s in seeds]
    mean_r2 = float(np.mean(r2s))
    return mean_r2, abs(mean_r2 - lam) < tol
