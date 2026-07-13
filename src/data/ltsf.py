"""Standard LTSF multivariate benchmarks (electricity, weather) — official
Autoformer csv distribution, literature protocol: ratio split 0.7/0.1/0.2,
global z-score from train stats only, val/test segments prefixed with
lookback rows of history.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.etth import Splits, WindowDataset

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "curated", "raw", "ltsf")

FILES = {"electricity": "electricity.csv", "weather": "weather.csv"}


def load_ltsf_frame(name: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(RAW_DIR, FILES[name]), parse_dates=["date"])
    df = df.set_index("date").astype("float64")
    assert df.index.is_monotonic_increasing
    assert not df.isna().any().any(), f"{name}: NaN in raw frame"
    return df


class LTSFDataset:
    def __init__(self, name: str, lookback: int, horizon: int,
                 train_frac: float = 0.7, val_frac: float = 0.1):
        df = load_ltsf_frame(name)
        values = df.values
        T = len(values)
        t1, t2 = int(T * train_frac), int(T * (train_frac + val_frac))

        train_seg = values[:t1]
        self.mean = train_seg.mean(axis=0)
        self.std = train_seg.std(axis=0)
        scaled = (values - self.mean) / self.std

        L = lookback
        self.lookback, self.horizon = lookback, horizon
        self.splits = Splits(train=scaled[:t1], val=scaled[t1 - L : t2],
                             test=scaled[t2 - L :], mean=self.mean, std=self.std)
        self.num_features = values.shape[1]

    def windows(self, split: str) -> WindowDataset:
        return WindowDataset(getattr(self.splits, split), self.lookback, self.horizon)
