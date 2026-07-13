"""CondNorm — covariate-conditional normalization (proposed method).

Two pieces:
  1. first_stage_level(): g(x_t) -> level series m_hat, fitted with LightGBM
     on TRAIN indices only, predicting y_t from exogenous covariates known at
     target time (lead-matched NWP / calendar). No past-y features — the level
     estimate must be exogenous (docs/theory_g1.md M1: CN observes g_T).
  2. CondNormTransform: exactly invertible series transform
        r_t = (y_t - m_hat_t) / s        (s = train std of residual)
     with inverse(transform(y)) == y. The backbone consumes r-windows; the
     restoration adds m_hat at TARGET positions (future covariates -> no
     window-statistic extrapolation, unlike RevIN).

The heteroscedastic variant (divide by sigma(x)) is a second regression on
squared residuals — deferred until a dataset shows conditional scale (kept
out per don't-over-engineer)."""

from __future__ import annotations

import numpy as np


def first_stage_level(features: np.ndarray, y: np.ndarray, train_end: int,
                      **lgbm_kwargs) -> np.ndarray:
    """Fit LightGBM g(features)->y on [:train_end], predict every t.

    features: (T, d) exogenous covariates valid at each target time t.
    Leakage contract: only rows < train_end enter the fit (asserted in tests).
    """
    import lightgbm as lgb

    params = dict(n_estimators=300, learning_rate=0.05, num_leaves=31,
                  min_child_samples=50, verbose=-1, random_state=0)
    params.update(lgbm_kwargs)
    model = lgb.LGBMRegressor(**params)
    model.fit(features[:train_end], y[:train_end])
    return model.predict(features)


class CondNormTransform:
    """Invertible: inverse(transform(y)) == y (up to float precision)."""

    def __init__(self, y: np.ndarray, level: np.ndarray, train_end: int):
        if y.shape != level.shape:
            raise ValueError("y and level must align")
        self.level = level.astype(np.float64)
        resid = y.astype(np.float64) - self.level
        self.mu = resid[:train_end].mean()
        self.sigma = resid[:train_end].std()
        if self.sigma <= 0:
            raise ValueError("degenerate residual scale")

    def transform(self, y: np.ndarray) -> np.ndarray:
        return (y - self.level - self.mu) / self.sigma

    def inverse(self, r: np.ndarray, positions: np.ndarray | None = None) -> np.ndarray:
        """positions: indices into the series for windowed outputs; None = full."""
        level = self.level if positions is None else self.level[positions]
        return r * self.sigma + self.mu + level
