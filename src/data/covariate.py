"""Windowed dataset over a curated covariate frame (curation.py output).

Univariate target 'y' + exogenous covariates at target time. Chronological
6:2:2 split, global z-score from train stats only, drop_last=False semantics.

Segment-aware: curated frames may contain documented archive holes (e.g.
jeju_wind 2023-06-25..07-04, a KMA-side -99 period). Rows are contiguous
segments; a window is admissible only if it lies entirely inside one segment
— no window ever spans a gap, and nothing is interpolated.

CondNorm arms use `covariates` + `first_stage_level` (src/norms/condnorm.py)
to build the level series, then wrap the TRANSFORMED series with the same
windowing — see src/train.py routing in G4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.curation import BUILDERS


def segment_ids(index: pd.DatetimeIndex) -> np.ndarray:
    """0-based contiguous-segment id per row. The regular step is the
    dataset's own median step (hourly for KPX/ETT, 10-min for weather)."""
    steps = np.diff(index.values) / np.timedelta64(1, "s")
    if len(steps) == 0:
        return np.zeros(len(index), dtype=int)
    regular = np.median(steps)
    return np.concatenate([[0], np.cumsum(steps != regular)]).astype(int)


def longest_contiguous(df: pd.DataFrame) -> pd.DataFrame:
    seg = segment_ids(df.index)
    best = np.bincount(seg).argmax()
    return df[seg == best]


class SegmentedWindowDataset(Dataset):
    """Windows over rows [lo, hi) of a scaled series, never crossing a
    segment boundary. x: (L, 1), y: (h, 1)."""

    def __init__(self, series: np.ndarray, seg: np.ndarray, lo: int, hi: int,
                 lookback: int, horizon: int):
        self.series = series.astype(np.float32)
        self.L, self.h = lookback, horizon
        span = lookback + horizon
        starts = np.arange(max(lo, 0), hi - span + 1)
        ok = seg[starts] == seg[starts + span - 1]
        self.starts = starts[ok]
        if len(self.starts) == 0:
            raise ValueError("no admissible windows in range")

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, i: int):
        s = self.starts[i]
        x = self.series[s : s + self.L, None]
        y = self.series[s + self.L : s + self.L + self.h, None]
        return torch.from_numpy(x), torch.from_numpy(y)


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
        self.seg = segment_ids(df.index)
        self.lookback, self.horizon = lookback, horizon
        T = len(self.y_raw)
        self.t1, self.t2 = int(T * train_frac), int(T * (train_frac + val_frac))
        self.mu = self.y_raw[: self.t1].mean()
        self.sigma = self.y_raw[: self.t1].std()
        self.y = (self.y_raw - self.mu) / self.sigma
        self.num_features = 1

    def windows(self, split: str) -> SegmentedWindowDataset:
        L = self.lookback
        lo, hi = {"train": (0, self.t1), "val": (self.t1 - L, self.t2),
                  "test": (self.t2 - L, len(self.y))}[split]
        return SegmentedWindowDataset(self.y, self.seg, lo, hi, L, self.horizon)
