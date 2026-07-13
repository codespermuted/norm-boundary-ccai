"""Windowed dataset over a curated covariate frame (curation.py output).

Univariate target 'y' + exogenous covariates at target time. Chronological
6:2:2 split, global z-score from train stats only, drop_last=False semantics
(WindowDataset covers every admissible window).

CondNorm arms use `covariates` + `first_stage_level` (src/norms/condnorm.py)
to build the level series, then wrap the TRANSFORMED series with the same
windowing — see src/train.py routing in G4.
"""

from __future__ import annotations

import numpy as np

from src.data.curation import BUILDERS
from src.data.etth import WindowDataset


class CovariateSeries:
    def __init__(self, name: str, lookback: int, horizon: int,
                 train_frac: float = 0.6, val_frac: float = 0.2):
        if name not in BUILDERS:
            raise KeyError(f"unknown curated dataset '{name}'")
        df = BUILDERS[name]()
        self.frame = df
        self.y_raw = df["y"].values.astype(np.float64)
        self.covariates = (df.drop(columns=["y"]).values.astype(np.float64)
                           if df.shape[1] > 1 else None)
        self.index = df.index
        self.lookback, self.horizon = lookback, horizon
        T = len(self.y_raw)
        self.t1, self.t2 = int(T * train_frac), int(T * (train_frac + val_frac))
        self.mu = self.y_raw[: self.t1].mean()
        self.sigma = self.y_raw[: self.t1].std()
        self.y = (self.y_raw - self.mu) / self.sigma
        self.num_features = 1

    def segment(self, split: str) -> np.ndarray:
        L = self.lookback
        lo, hi = {"train": (0, self.t1), "val": (self.t1 - L, self.t2),
                  "test": (self.t2 - L, len(self.y))}[split]
        return self.y[lo:hi]

    def windows(self, split: str) -> WindowDataset:
        seg = self.segment(split)[:, None]  # (T, 1)
        return WindowDataset(seg, self.lookback, self.horizon)
