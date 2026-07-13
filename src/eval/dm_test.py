"""Diebold-Mariano test with Harvey small-sample correction (plan §5.4).

Loss series = per-origin h-step-mean squared errors on the test set.
HAC variance with h-1 lags (Newey-West), Harvey et al. (1997) correction,
two-sided p-value from t distribution with n-1 df.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def dm_test(e1: np.ndarray, e2: np.ndarray, h: int) -> dict:
    """e1, e2: per-origin losses of models 1 and 2 (same origins).
    Positive statistic -> model 1 has HIGHER loss (model 2 better)."""
    d = np.asarray(e1, float) - np.asarray(e2, float)
    n = len(d)
    dbar = d.mean()
    lags = max(h - 1, 0)
    gamma0 = np.mean((d - dbar) ** 2)
    var = gamma0
    for k in range(1, min(lags, n - 1) + 1):
        cov = np.mean((d[k:] - dbar) * (d[:-k] - dbar))
        var += 2 * (1 - k / (lags + 1)) * cov  # Bartlett weights
    var = max(var, 1e-300) / n
    dm = dbar / np.sqrt(var)
    harvey = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = float(dm * harvey)
    p = float(2 * stats.t.sf(abs(stat), df=n - 1))
    return {"stat": stat, "p": p, "mean_diff": float(dbar), "n": n}
