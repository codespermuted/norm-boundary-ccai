"""Exact (closed-form) optimal linear predictor MSE per normalization class.

Proposition 2' machinery: following Toner & Darlow (ICML 2024), a linear
backbone under each normalization is a CONSTRAINED linear regression with a
closed-form optimum. Solving the normal equations on the train segment and
evaluating on the test segment gives the exact theory value that SGD training
approximates — no stochastic optimization involved.

This refines the stylized model M1 (docs/theory_g1.md): M1's estimators are
pure restoration rules, but a linear backbone ALSO tracks residual level from
its input window (implicit instance normalization in-distribution). M1's
lambda* is therefore an upper bound; this module gives the exact crossover
for the linear class.

Classes:
  raw      ŷ = W s_win + b                      (s = global-z of y)
  revin    ŷ = W (s_win − mean(s_win)) + mean(s_win) + b   (mean restoration)
  cn(ℓ)    ŷ = W r_win + b, restored with ℓ_target        (r = z-scored y − ℓ)

All MSEs reported in the global-z scale of y (same as runner.py).
"""

from __future__ import annotations

import numpy as np


def _windows(arr: np.ndarray, L: int, h: int):
    from numpy.lib.stride_tricks import sliding_window_view

    n = len(arr) - L - h + 1
    xw = sliding_window_view(arr, L)[:n]
    yw = sliding_window_view(arr, h)[L : L + n]
    return xw, yw


def _fit_eval(x_tr, y_tr, x_te, y_te, ridge: float = 1e-6):
    """Multi-output ridge-OLS closed form; returns test predictions."""
    n, L = x_tr.shape
    X = np.column_stack([np.ones(n), x_tr])
    G = X.T @ X + ridge * np.eye(L + 1)
    W = np.linalg.solve(G, X.T @ y_tr)
    Xte = np.column_stack([np.ones(len(x_te)), x_te])
    return Xte @ W


def ols_class_mse(y: np.ndarray, level: np.ndarray, arm: str, L: int, h: int,
                  train_frac: float = 0.6, val_frac: float = 0.2) -> float:
    """Closed-form optimal linear test MSE for one arm, global-z scale.

    level: restoration series (zeros for raw/revin, m or m_hat for CN arms).
    Splits and scalings mirror runner.py exactly (train stats only).
    """
    T = len(y)
    t1, t2 = int(T * train_frac), int(T * (train_frac + val_frac))
    mu_g, sd_g = y[:t1].mean(), y[:t1].std()

    resid = y - level
    mu_r, sd_r = resid[:t1].mean(), resid[:t1].std()
    r = (resid - mu_r) / sd_r

    xw_tr, yw_tr = _windows(r[:t1], L, h)
    xw_te, yw_te = _windows(r[t2 - L :], L, h)

    if arm == "revin":
        mean_tr = xw_tr.mean(axis=1, keepdims=True)
        mean_te = xw_te.mean(axis=1, keepdims=True)
        pred_r = _fit_eval(xw_tr - mean_tr, yw_tr - mean_tr,
                           xw_te - mean_te, yw_te - mean_te) + mean_te
    else:
        pred_r = _fit_eval(xw_tr, yw_tr, xw_te, yw_te)

    # back to original scale, add restoration level at target positions
    n_te = xw_te.shape[0]
    tgt = (t2 - L) + L + np.arange(n_te)[:, None] + np.arange(h)[None, :]
    pred_y = pred_r * sd_r + mu_r + level[tgt]
    true_y = y[tgt]
    return float(np.mean(((pred_y - true_y) / sd_g) ** 2))


def ols_theory_row(series: dict, L: int, h: int) -> dict:
    """Closed-form class-optimal MSE for all four arms of one series."""
    y, m, mhat = series["y"], series["m"], series["mhat"]
    zeros = np.zeros_like(y)
    return {
        "raw": ols_class_mse(y, zeros, "raw", L, h),
        "revin": ols_class_mse(y, zeros, "revin", L, h),
        "cn_oracle": ols_class_mse(y, m, "cn", L, h),
        "cn_est": ols_class_mse(y, mhat, "cn", L, h),
    }
