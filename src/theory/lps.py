"""LPS — Level Predictability Score (plan §3.1, contribution C3).

LPS = out-of-sample R^2 of regressing w-window TARGET MEANS on window-mean
covariates, under chronological expanding-window CV (no leakage):

    ybar_i ~ g(xbar_i),   windows non-overlapping (stride w) so CV blocks
                          are serially decoupled.

g defaults to LightGBM; ridge is the sensitivity alternative (plan §3.1).
R^2 uses the fold-train mean of ybar as the baseline (never the OOS mean).

Two variants, selected by the `eval_end` flag of `lps()`:

  FULL (`eval_end=None`, the default) — the frozen, pre-registered protocol.
    Evaluation blocks are 5 equal splits of the final 60% of the windows and
    therefore span the whole series, including rows the forecasting grid later
    uses as its test segment. Nothing in the screen consumes model outputs or
    split boundaries, but the blocks are not confined to data a practitioner
    would hold at the forecast origin. This is what `results/lps_official.csv`
    and `results/graded_lps_lps.csv` contain and it must not change.

  DEPLOYMENT (`eval_end=<first test row>`) — added 2026-08-25 after review.
    The identical protocol run on the pre-origin prefix only, so the screen is
    computable before the forecaster's evaluation window exists.
"""

from __future__ import annotations

import numpy as np


def _window_means(y: np.ndarray, X: np.ndarray, w: int):
    n = len(y) // w
    ybar = y[: n * w].reshape(n, w).mean(axis=1)
    Xbar = X[: n * w].reshape(n, w, -1).mean(axis=1)
    return ybar, Xbar


def _fit_predict(model: str, Xtr, ytr, Xte):
    if model == "lgbm":
        import lightgbm as lgb

        m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                              num_leaves=31, min_child_samples=20,
                              verbose=-1, random_state=0)
    elif model == "ridge":
        from sklearn.linear_model import Ridge

        m = Ridge(alpha=1.0)
    else:
        raise KeyError(model)
    m.fit(Xtr, ytr)
    return m.predict(Xte)


def lps(y: np.ndarray, X: np.ndarray, w: int, n_folds: int = 5,
        min_train_frac: float = 0.4, model: str = "lgbm",
        eval_end: int | None = None) -> dict:
    """Returns {'lps': pooled OOS R^2, 'n_windows': ..., 'per_fold': [...]}.

    `eval_end` selects the DEPLOYMENT variant (added 2026-08-25, in response
    to review): rows at or after `eval_end` — the forecast origin, i.e. the
    first row of the grid's test segment — are removed before any window is
    formed, so no evaluation block can overlap the segment the forecaster is
    later scored on. The default `eval_end=None` is the frozen pre-registered
    protocol (`compute_lps_official.py`, commit cab17c1) and is left
    bit-identical: evaluation blocks span the full series.

    The deployment variant re-runs the *identical* protocol on the pre-origin
    prefix — same w, same 5 expanding folds over the final 60% of the windows
    that exist there — rather than truncating the frozen fold boundaries, so
    the two variants differ in the data available, not in the estimator.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if eval_end is not None:
        eval_end = int(eval_end)
        if eval_end < 0 or eval_end > len(y):
            raise ValueError(f"eval_end {eval_end} outside series of length {len(y)}")
        if eval_end // w < n_folds + 1:
            raise ValueError(
                f"eval_end {eval_end} leaves {eval_end // w} windows of width {w}; "
                f"need at least {n_folds + 1} for {n_folds} expanding folds")
        y, X = y[:eval_end], X[:eval_end]
    ybar, Xbar = _window_means(y, X, w)
    n = len(ybar)
    first_test = int(n * min_train_frac)
    bounds = np.linspace(first_test, n, n_folds + 1, dtype=int)

    sse = sst = 0.0
    per_fold = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if hi <= lo:
            continue
        pred = _fit_predict(model, Xbar[:lo], ybar[:lo], Xbar[lo:hi])
        base = ybar[:lo].mean()
        e = float(np.sum((ybar[lo:hi] - pred) ** 2))
        t = float(np.sum((ybar[lo:hi] - base) ** 2))
        sse += e
        sst += t
        per_fold.append(1 - e / t if t > 0 else np.nan)
    return {"lps": 1 - sse / sst if sst > 0 else float("nan"),
            "n_windows": n, "per_fold": per_fold,
            "variant": "full" if eval_end is None else "deployment",
            "eval_end": eval_end,
            "first_eval_window": int(first_test)}


def delta_lps(y: np.ndarray, X: np.ndarray, w: int, n_folds: int = 5,
              min_train_frac: float = 0.4, model: str = "lgbm") -> dict:
    """Incremental LPS: exogenous predictability of the NEXT window mean
    BEYOND persistence (past window mean). Corrects the absolute-LPS blind
    spot on strongly persistent series (e.g., SMP-like):

        dLPS = R2(ybar_{i+1} ~ xbar_{i+1}, ybar_i) - R2(ybar_{i+1} ~ ybar_i)
    """
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    ybar, Xbar = _window_means(y, X, w)
    tgt = ybar[1:]
    lag = ybar[:-1][:, None]
    cov_next = Xbar[1:]
    n = len(tgt)
    first_test = int(n * min_train_frac)
    bounds = np.linspace(first_test, n, n_folds + 1, dtype=int)

    def oos_r2(feats):
        sse = sst = 0.0
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            if hi <= lo:
                continue
            pred = _fit_predict(model, feats[:lo], tgt[:lo], feats[lo:hi])
            base = tgt[:lo].mean()
            sse += float(np.sum((tgt[lo:hi] - pred) ** 2))
            sst += float(np.sum((tgt[lo:hi] - base) ** 2))
        return 1 - sse / sst if sst > 0 else float("nan")

    r2_full = oos_r2(np.column_stack([cov_next, lag]))
    r2_pers = oos_r2(lag)
    return {"delta_lps": r2_full - r2_pers, "r2_full": r2_full,
            "r2_persistence": r2_pers, "n_windows": n}


def calendar_features(index) -> np.ndarray:
    """Deterministic exogenous covariates available for ANY dataset:
    daily/weekly/yearly harmonics."""
    import pandas as pd

    idx = pd.DatetimeIndex(index)
    hour = idx.hour.values + idx.minute.values / 60
    dow = idx.dayofweek.values
    doy = idx.dayofyear.values
    two_pi = 2 * np.pi
    return np.column_stack([
        np.sin(two_pi * hour / 24), np.cos(two_pi * hour / 24),
        np.sin(two_pi * (dow * 24 + hour) / (7 * 24)),
        np.cos(two_pi * (dow * 24 + hour) / (7 * 24)),
        np.sin(two_pi * doy / 365.25), np.cos(two_pi * doy / 365.25),
    ])
