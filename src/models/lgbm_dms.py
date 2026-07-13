"""LightGBM direct multi-step (DMS) backbone — one regressor per horizon step.

Channel-independent: windows from every channel are pooled (like the CI
linear backbones). Not a torch module — src/train.py routes to fit_predict()
when backbone == 'lgbm_dms'. Deterministic given random_state, so seeds are
not swept (plan §5.2: LightGBM runs once).

Instance normalization for GBMs ('revin' arm) is the plain window z-score
applied in numpy: subtract/divide window stats, restore on the prediction —
the affine-free RevIN equivalent (a linear tree input shift has no learnable
affine anyway).
"""

from __future__ import annotations

import numpy as np


class LgbmDMS:
    def __init__(self, horizon: int, n_estimators: int = 200,
                 learning_rate: float = 0.07, num_leaves: int = 31,
                 n_jobs: int = -1):
        self.horizon = horizon
        self.params = dict(n_estimators=n_estimators, learning_rate=learning_rate,
                           num_leaves=num_leaves, verbose=-1, random_state=0,
                           n_jobs=n_jobs)
        self.models: list = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LgbmDMS":
        """x: (N, L) pooled CI windows, y: (N, h)."""
        import lightgbm as lgb

        self.models = []
        for step in range(self.horizon):
            m = lgb.LGBMRegressor(**self.params)
            m.fit(x, y[:, step])
            self.models.append(m)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.stack([m.predict(x) for m in self.models], axis=1)


def window_znorm(x: np.ndarray, eps: float = 1e-5):
    """Window z-score for GBM arms: returns normalized x and (mu, sd)."""
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True) + eps
    return (x - mu) / sd, (mu, sd)
