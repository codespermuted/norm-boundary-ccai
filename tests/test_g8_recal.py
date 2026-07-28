"""G8 recalibration — the conformal wrapper must restore nominal coverage.

Synthetic check (no GPU, no data): an under-dispersed quantile forecaster
(true y ~ N(0,1), predicted quantiles from N(0, 0.5^2)) under-covers by
construction; per-quantile additive offsets fitted on a calibration sample
must bring held-out coverage of EVERY level back to nominal, and must be
exact on the calibration sample itself (the defining property).
"""

import numpy as np
from scipy.stats import norm as sp_norm

from experiments.g8_recal import apply_offsets, conformal_offsets, q_metrics

QS = np.asarray([round(0.1 * k, 1) for k in range(1, 10)])


def _synthetic(n, h, c, rng, spread=0.5):
    true = rng.normal(size=(n, h, c))
    base = sp_norm.ppf(QS) * spread                       # under-dispersed
    pred = np.broadcast_to(base, (n, h, c, len(QS))).copy()
    return pred, true


def test_offsets_restore_nominal_coverage():
    rng = np.random.default_rng(0)
    cal_pred, cal_true = _synthetic(400, 24, 2, rng)
    te_pred, te_true = _synthetic(400, 24, 2, rng)

    before = q_metrics(te_pred, te_true, QS)
    assert before["cov80"] < 0.70                          # broken on purpose

    for per_step in (False, True):
        delta = conformal_offsets(cal_pred, cal_true, QS, per_step=per_step)
        after = q_metrics(apply_offsets(te_pred, delta), te_true, QS)
        assert abs(after["cov80"] - 0.80) < 0.03
        assert abs(after["cov_lo"] - 0.10) < 0.02
        assert abs(after["cov_hi"] - 0.90) < 0.02
        # pinball must improve when miscalibration is the failure mode
        assert after["pinball"] < before["pinball"]

    # exactness on the calibration sample (pooled): each level's exceedance
    # equals its nominal q up to 1/n discreteness
    delta = conformal_offsets(cal_pred, cal_true, QS)
    cal_adj = apply_offsets(cal_pred, delta)
    n_tot = cal_true.size
    for k, q in enumerate(QS):
        emp = (cal_true <= cal_adj[..., k]).mean()
        assert abs(emp - q) < 2.0 / np.sqrt(n_tot) + 1e-3


def test_offsets_monotone_and_shape():
    rng = np.random.default_rng(1)
    cal_pred, cal_true = _synthetic(200, 8, 1, rng)
    d_pool = conformal_offsets(cal_pred, cal_true, QS)
    d_step = conformal_offsets(cal_pred, cal_true, QS, per_step=True)
    assert d_pool.shape == (9,) and d_step.shape == (8, 9)
    # residual quantiles at increasing levels are non-decreasing
    assert np.all(np.diff(d_pool) >= -1e-12)
    # rearrangement keeps the quantile axis sorted after any shift
    out = apply_offsets(cal_pred, d_step)
    assert np.all(np.diff(out, axis=-1) >= 0)
