"""Closed-form MSEs for RAW / IN / CN under the stylized model M1.

Mathematical specification: docs/theory_g1.md (this module implements it
verbatim; if they disagree, the document wins). All MSEs are level-restoration
errors plus the common shape residual sigma_eps^2 (assumption A3).

Per-instance Gaussian model (all independent):
    g_T  ~ N(0, lam*V)          covariate-explained target-time level (CN observes it)
    v_T  ~ N(0, (1-lam)*V)      unexplained target-time level
    Delta~ N(0, sigma_delta^2)  covariate-orthogonal train->test shift (shared by window & target)
    d_x  ~ N(0, s_x2h)          within-horizon covariate-driven level change (g_w = g_T - d_x)
    d_u  ~ N(0, h*sigma_u^2)    within-horizon unexplained drift        (v_w = v_T - d_u)
    eps_bar ~ N(0, sigma_z^2/w) window-mean shape noise
    e    ~ N(0, sigma_est^2)    CN first-stage error (m_hat = g_T + e)
    zeta ~ N(0, sigma_eps^2)    common shape residual

    L = Delta + g_T + v_T,  M = Delta + g_w + v_w,  y_bar = M + eps_bar,  y = L + zeta
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class TheoryParams:
    V: float = 1.0
    lam: float = 0.5
    w: int = 96
    h: int = 24
    sigma_z: float = 1.0
    sigma_u: float = 0.06
    s_x2h: float = 0.0
    sigma_delta: float = 0.0
    sigma_est: float = 0.0
    sigma_eps: float = 0.0

    def __post_init__(self):
        if not 0.0 <= self.lam <= 1.0:
            raise ValueError(f"lam must be in [0,1], got {self.lam}")
        if self.V <= 0 or self.w <= 0 or self.h <= 0:
            raise ValueError("V, w, h must be positive")


def window_noise_var(p: TheoryParams) -> float:
    """Var(eps_bar) = sigma_z^2 / w (iid shape; autocorrelated shape uses
    effective_window_noise to inflate this — see docs/theory_g1.md §5)."""
    return p.sigma_z**2 / p.w


def horizon_drift_var(p: TheoryParams) -> float:
    """Total within-horizon level displacement variance paid by IN."""
    return p.s_x2h + p.h * p.sigma_u**2


def effective_window_noise(sigma_z: float, w: int, rho1: float = 0.0) -> float:
    """Window-mean noise variance for AR(1) shape with lag-1 autocorrelation rho1.

    Var(z_bar) = (sigma_z^2/w) * tau with integrated autocorrelation time
    tau ~= (1+rho1)/(1-rho1) (large-w approximation). rho1=0 recovers iid.
    """
    tau = (1 + rho1) / (1 - rho1)
    return sigma_z**2 * tau / w


def mse_raw(p: TheoryParams) -> float:
    return p.V + p.sigma_delta**2 + p.sigma_eps**2


def mse_in(p: TheoryParams) -> float:
    return horizon_drift_var(p) + window_noise_var(p) + p.sigma_eps**2


def mse_cn(p: TheoryParams, oracle: bool = False) -> float:
    est = 0.0 if oracle else p.sigma_est**2
    return (1 - p.lam) * p.V + p.sigma_delta**2 + est + p.sigma_eps**2


def in_minus_cn_gap(p: TheoryParams, oracle: bool = False) -> float:
    """Proposition 3 object: MSE_IN(h) - MSE_CN(h), monotone increasing in h."""
    return mse_in(p) - mse_cn(p, oracle=oracle)


def lambda_star(p: TheoryParams) -> float:
    """Proposition 2 crossover: CN-est beats IN iff lam > lambda_star.

    Returned value may fall outside [0,1]:
      < 0  -> CN dominates for every lam;  > 1 -> IN dominates for every lam.
    Valid when s_x2h does not depend on lam; otherwise use lambda_star_numeric.
    """
    return 1 - (horizon_drift_var(p) + window_noise_var(p)
                - p.sigma_delta**2 - p.sigma_est**2) / p.V


def lambda_star_numeric(p: TheoryParams, s_x2h_of_lam=None, tol: float = 1e-10) -> float:
    """Crossover solved by bisection, allowing s_x2h = f(lam) couplings
    (e.g. the G2 DGP mapping where the covariate share scales the ramp term).

    Returns nan if MSE_CN - MSE_IN has no sign change on [0,1].
    """
    def diff(lam: float) -> float:
        s = p.s_x2h if s_x2h_of_lam is None else s_x2h_of_lam(lam)
        q = replace(p, lam=lam, s_x2h=s)
        return mse_cn(q) - mse_in(q)

    lo, hi = 0.0, 1.0
    f_lo, f_hi = diff(lo), diff(hi)
    if f_lo * f_hi > 0:
        return float("nan")
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if diff(mid) * f_lo <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def in_beats_raw(p: TheoryParams) -> bool:
    """Proposition 2, first inequality."""
    return mse_in(p) < mse_raw(p)


METHODS = ("raw", "in", "cn_est")


def dominance_grid(
    lam_grid: np.ndarray, drift2_grid: np.ndarray, p: TheoryParams
) -> np.ndarray:
    """Label array (len(drift2_grid), len(lam_grid)) of argmin MSE among METHODS.

    drift2_grid is sigma_delta^2 (variance, not std). Fig 1 input.
    """
    labels = np.empty((len(drift2_grid), len(lam_grid)), dtype=int)
    for i, d2 in enumerate(drift2_grid):
        for j, lam in enumerate(lam_grid):
            q = replace(p, lam=float(lam), sigma_delta=float(np.sqrt(d2)))
            mses = [mse_raw(q), mse_in(q), mse_cn(q, oracle=False)]
            labels[i, j] = int(np.argmin(mses))
    return labels


def var_ybar(p: TheoryParams) -> float:
    """Var(y_bar) under train distribution (Delta included for completeness)."""
    return (p.V + p.s_x2h + p.h * p.sigma_u**2 + window_noise_var(p)
            + p.sigma_delta**2)


def prop1_interaction_gap(p: TheoryParams, kappa: float) -> float:
    """Proposition 1: irreducible excess MSE of the interaction-free linear
    class {1, y_bar, g} when the optimal predictor needs kappa * y_bar * g.

    For centered jointly Gaussian (y_bar, g): the interaction term is
    orthogonal to the linear span, so the gap is
        kappa^2 * Var(y_bar * g) = kappa^2 * (Var(y_bar)Var(g) + Cov(y_bar,g)^2)
    with Cov(y_bar, g_T) = lam * V (docs/theory_g1.md §3, Isserlis).
    """
    var_g = p.lam * p.V
    cov = p.lam * p.V
    return kappa**2 * (var_ybar(p) * var_g + cov**2)
