"""Inference add-ons for the official LPS (G7 audit, plan Block E).

Three diagnostics layered ON TOP of the pre-registered LPS statistic
(src/theory/lps.py — w=96, LightGBM first stage, expanding chronological CV):

  * permutation_test — p-value against the null "the LPS is a chance
    alignment": circularly shift the window-mean covariate rows relative to
    the window-mean target sequence and recompute the LPS per shift.
  * mbb_ci — moving-block bootstrap percentile CI for the LPS point value.
  * lambda_star_hat — rough plug-in estimate of the theory crossover
    lambda* (src/theory/closed_form.lambda_star) from measurable quantities.

POST-HOC DIAGNOSTIC ONLY. The pre-registered decision rule remains the
absolute-LPS tau threshold fixed before the G4 grid (results/gate1.md);
nothing in this module alters, replaces, or re-tunes that rule — outputs are
reported as supporting inference (results/lps_inference.*) and labelled as
such.

The statistic is EXACTLY the official LPS: window means come from
lps._window_means and the expanding-CV loop in _cv_oos mirrors lps.lps
verbatim (equality is guarded by tests/test_lps_inference.py).

Multivariate targets: pass y with shape (T, C) and a shared covariate matrix
X of shape (T, p). The statistic is then the per-channel-mean LPS (matching
compute_lps_official.py), and every shift / bootstrap replicate applies the
SAME offsets or block indices to all channels, preserving the cross-channel
dependence structure under the null.
"""

from __future__ import annotations

import math

import numpy as np
from joblib import Parallel, delayed

from src.theory.lps import _fit_predict, _window_means


def _channel_means(y, X, w):
    """Per-channel ybar list + shared Xbar via the official _window_means."""
    Y = np.asarray(y, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    ybars = []
    Xbar = None
    for c in range(Y.shape[1]):
        yb, Xbar = _window_means(Y[:, c], X, w)
        ybars.append(yb)
    return ybars, Xbar


def _cv_oos(ybar, Xbar, n_folds=5, min_train_frac=0.4, model="lgbm"):
    """Expanding-CV pooled OOS R^2 on precomputed window means.

    Must stay numerically identical to src.theory.lps.lps (same fold bounds,
    same fold-train baseline, same pooling) — the permutation/bootstrap
    statistic is only valid if it IS the official LPS. Also returns the
    squared OOS residuals (sigma_est^2 proxy for lambda_star_hat).
    """
    n = len(ybar)
    first_test = int(n * min_train_frac)
    bounds = np.linspace(first_test, n, n_folds + 1, dtype=int)
    sse = sst = 0.0
    resid_sq = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if hi <= lo:
            continue
        pred = _fit_predict(model, Xbar[:lo], ybar[:lo], Xbar[lo:hi])
        r = ybar[lo:hi] - pred
        resid_sq.append(r * r)
        sse += float(np.sum(r * r))
        sst += float(np.sum((ybar[lo:hi] - ybar[:lo].mean()) ** 2))
    return {"lps": 1 - sse / sst if sst > 0 else float("nan"),
            "resid_sq": np.concatenate(resid_sq) if resid_sq else np.empty(0)}


def _mean_lps_batch(ybars, Xbar_list, n_folds, min_train_frac, model, n_jobs):
    """Per-channel-mean LPS for each Xbar/index configuration in the batch.

    Flattens (configuration, channel) into one joblib task list so both the
    many-channel case (electricity) and the many-replicate univariate case
    parallelise well.
    """
    C = len(ybars)
    per = Parallel(n_jobs=n_jobs)(
        delayed(_cv_oos)(yb, Xb, n_folds, min_train_frac, model)
        for Xb, sel in Xbar_list for yb in (ybars if sel is None
                                            else [y[sel] for y in ybars]))
    stats = np.array([p["lps"] for p in per], dtype=float)
    return stats.reshape(len(Xbar_list), C).mean(axis=1)


def permutation_test(y, X, w, B=999, model="lgbm", align_windows=None,
                     seed=0, n_folds=5, min_train_frac=0.4, n_jobs=1):
    """Circular-shift permutation test for the official LPS.

    Null: the window-mean covariate rows Xbar are circularly shifted by
    k windows (k in 1..n-1) relative to the window-mean targets ybar,
    destroying the alignment while preserving both marginal serial
    structures. Because windows are stride-w and non-overlapping, a
    one-window shift is a w-step shift of the raw series.

    align_windows: if an integer a, restrict k to multiples of a — the
    season-alignment-preserving option (e.g. a = one week's worth of windows
    keeps daily/weekly phase intact so the null cannot be rejected on
    calendar seasonality alone).

    If the admissible offsets number <= B they are ALL enumerated (exact
    randomization test over the shift group); otherwise B are sampled
    without replacement. p = (1 + #{LPS_null >= LPS_obs}) / (1 + B_eff).

    Multivariate y (T, C): statistic = per-channel-mean LPS; the same shift
    k is applied for all channels (shared Xbar), preserving cross-channel
    structure.

    Returns {'lps', 'p_value', 'B_effective', 'offsets', 'null', 'n_windows'}.
    """
    ybars, Xbar = _channel_means(y, X, w)
    n = len(ybars[0])
    step = 1 if align_windows is None else int(align_windows)
    if step < 1:
        raise ValueError(f"align_windows must be >= 1, got {align_windows}")
    offsets = np.arange(step, n, step)
    if len(offsets) == 0:
        raise ValueError(
            f"no admissible shifts: n_windows={n}, align_windows={align_windows}")
    if len(offsets) > B:
        rng = np.random.default_rng(seed)
        offsets = np.sort(rng.choice(offsets, size=B, replace=False))
    # index 0 = unshifted -> observed statistic; the rest = null replicates
    configs = [(np.roll(Xbar, int(k), axis=0), None) for k in (0, *offsets)]
    stats = _mean_lps_batch(ybars, configs, n_folds, min_train_frac, model,
                            n_jobs)
    obs, null = float(stats[0]), stats[1:]
    b_eff = len(null)
    p = (1 + int(np.sum(null >= obs))) / (1 + b_eff)
    return {"lps": obs, "p_value": float(p), "B_effective": b_eff,
            "offsets": offsets, "null": null, "n_windows": n}


def mbb_ci(y, X, w, B=499, block_len=None, alpha=0.1, model="lgbm",
           seed=0, n_folds=5, min_train_frac=0.4, n_jobs=1):
    """Moving-block bootstrap percentile CI for the official LPS.

    The paired window-mean sequence (ybar_i, Xbar_i) is resampled in
    CIRCULAR blocks of length block_len (default ceil(n_windows^(1/3))):
    block starts drawn uniformly with replacement, blocks concatenated and
    truncated to n windows, LPS recomputed per replicate, percentile
    interval [alpha/2, 1-alpha/2]. Pairs stay matched, so the target
    functional (window-level covariate R^2) is preserved; only sampling
    variability is measured.

    Multivariate y (T, C): the same block index sequence is applied to all
    channels (cross-channel structure preserved); the replicate statistic is
    the per-channel-mean LPS.

    Returns {'lps', 'ci_lo', 'ci_hi', 'block_len', 'reps', 'n_windows'}.
    """
    ybars, Xbar = _channel_means(y, X, w)
    n = len(ybars[0])
    L = int(block_len) if block_len else math.ceil(n ** (1 / 3))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(B, math.ceil(n / L)))
    base = np.arange(L)
    # config 0 = identity indices -> observed statistic
    configs = [(Xbar, None)] + [
        (Xbar[idx], idx) for idx in
        ((s[:, None] + base[None, :]).reshape(-1)[:n] % n for s in starts)]
    stats = _mean_lps_batch(ybars, configs, n_folds, min_train_frac, model,
                            n_jobs)
    obs, reps = float(stats[0]), stats[1:]
    lo, hi = np.nanpercentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"lps": obs, "ci_lo": float(lo), "ci_hi": float(hi),
            "block_len": L, "reps": reps, "n_windows": n}


def lambda_star_hat(y, X, w, h, model="lgbm", n_folds=5, min_train_frac=0.4):
    """Rough plug-in estimate of the crossover lambda* (Proposition 2).

    Implements closed_form.lambda_star with sigma_Delta^2 = 0:

        lambda*_hat = 1 - (S_h_hat + sigma_z2_hat / w - sigma_est2_hat) / V_hat

    Setting sigma_Delta^2 = 0 is CONSERVATIVE in the sense of a lower bound:
    covariate-orthogonal train->test drift only raises the true crossover,
    so lam > lambda*_hat is necessary but NOT sufficient for CN dominance
    (and lam < lambda*_hat does certify the IN region).

    Proxies (all rough, each documented with its known bias):
      V_hat        = Var(ybar), the window-mean variance. Theory Var(ybar)
                     = V + S_h + sigma_z^2/w (+ drift), so this OVERSTATES V
                     — pushes lambda*_hat up (toward IN), reinforcing the
                     conservative reading.
      sigma_est2_hat = mean squared OOS residual of the official LPS
                     expanding CV — E[(ybar - g_hat(xbar))^2]. Conflates
                     first-stage estimation error with the unexplained level
                     (1-lam)V + window noise, i.e. an UPPER proxy for
                     sigma_est^2.
      S_h_hat      = Var(ybar_{i+k} - ybar_i) with k = ceil(h/w): the level
                     displacement over ~one horizon. Includes 2x window
                     noise -> upward-biased.
      sigma_z2_hat / w = mean within-window variance of y, divided by w.
                     Within-window variation also contains genuine
                     sub-window signal -> upward-biased.

    POST-HOC DIAGNOSTIC ONLY — never a replacement for the pre-registered
    tau rule (module docstring). Values can fall outside [0, 1], with the
    same reading as closed_form.lambda_star (<0: CN for all lam; >1: IN for
    all lam).

    Multivariate y (T, C): per-channel estimates averaged (matching the
    per-channel-mean LPS convention). Returns the component proxies plus
    per-channel arrays.
    """
    ybars, Xbar = _channel_means(y, X, w)
    Y = np.asarray(y, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    n = len(ybars[0])
    k = math.ceil(h / w)
    per_ch = []
    for c, yb in enumerate(ybars):
        V_hat = float(np.var(yb))
        sigma_est2 = float(np.mean(_cv_oos(yb, Xbar, n_folds, min_train_frac,
                                           model)["resid_sq"]))
        S_h = float(np.var(yb[k:] - yb[:-k])) if k < n else float("nan")
        yw = Y[: n * w, c].reshape(n, w)
        sigma_z2_over_w = float(np.mean(yw.var(axis=1))) / w
        per_ch.append({
            "lambda_star_hat": 1 - (S_h + sigma_z2_over_w - sigma_est2) / V_hat,
            "V_hat": V_hat, "sigma_est2_hat": sigma_est2, "S_h_hat": S_h,
            "sigma_z2_over_w_hat": sigma_z2_over_w})
    out = {key: float(np.mean([p[key] for p in per_ch])) for key in per_ch[0]}
    out.update({"k_windows": k, "n_windows": n,
                "per_channel": np.array([p["lambda_star_hat"]
                                         for p in per_ch])})
    return out
