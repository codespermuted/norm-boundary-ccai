"""ETT hourly datasets with the standard LTSF protocol.

Data contract (curated/ Parquet):
  - hourly DatetimeIndex named 'date', strictly increasing, no gaps
  - float64 value columns, no NaN
  - split: train 12 months / val 4 months / test 4 months (border-based,
    literature standard), global z-score fitted on the train segment only
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ETT_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/{name}.csv"
CURATED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "curated")

# literature-standard borders for ETTh (hours)
_TRAIN_END = 12 * 30 * 24
_VAL_END = 12 * 30 * 24 + 4 * 30 * 24
_TEST_END = 12 * 30 * 24 + 8 * 30 * 24


def load_etth_frame(name: str = "ETTh1", curated_dir: str | None = None) -> pd.DataFrame:
    """Download (once), validate against the data contract, cache as Parquet."""
    curated_dir = curated_dir or CURATED_DIR
    os.makedirs(curated_dir, exist_ok=True)
    parquet_path = os.path.join(curated_dir, f"{name}.parquet")
    csv_path = os.path.join(curated_dir, f"{name}.csv")
    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
    else:
        if not os.path.exists(csv_path):
            import requests

            resp = requests.get(ETT_URL.format(name=name), timeout=60)
            resp.raise_for_status()
            with open(csv_path, "wb") as f:
                f.write(resp.content)
        df = pd.read_csv(csv_path, parse_dates=["date"]).set_index("date")
        df = df.astype("float64")
        df.to_parquet(parquet_path)

    # data contract validation
    assert df.index.name == "date" and df.index.is_monotonic_increasing
    assert not df.isna().any().any(), f"{name}: NaN in curated frame"
    deltas = np.diff(df.index.values)
    assert (deltas == np.timedelta64(1, "h")).all(), f"{name}: non-hourly gaps"
    assert len(df) >= _TEST_END, f"{name}: too short ({len(df)} < {_TEST_END})"
    return df


@dataclass
class Splits:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    mean: np.ndarray
    std: np.ndarray


class ETTHourly:
    def __init__(self, name: str, lookback: int, horizon: int, curated_dir: str | None = None):
        self.name = name
        self.lookback = lookback
        self.horizon = horizon
        df = load_etth_frame(name, curated_dir)
        values = df.values[:_TEST_END]

        train_seg = values[:_TRAIN_END]
        self.mean = train_seg.mean(axis=0)
        self.std = train_seg.std(axis=0)
        scaled = (values - self.mean) / self.std

        L = lookback
        self.splits = Splits(
            train=scaled[:_TRAIN_END],
            val=scaled[_TRAIN_END - L : _VAL_END],
            test=scaled[_VAL_END - L : _TEST_END],
            mean=self.mean,
            std=self.std,
        )
        self.num_features = values.shape[1]

    def windows(self, split: str) -> "WindowDataset":
        return WindowDataset(getattr(self.splits, split), self.lookback, self.horizon)


class WindowDataset(Dataset):
    """Sliding windows over one contiguous split segment.

    x: (lookback, C), y: (horizon, C), both float32 and globally z-scored.
    """

    def __init__(self, segment: np.ndarray, lookback: int, horizon: int):
        self.segment = segment.astype(np.float32)
        self.lookback = lookback
        self.horizon = horizon
        self.n = len(segment) - lookback - horizon + 1
        if self.n <= 0:
            raise ValueError("segment too short for lookback+horizon")

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        L, h = self.lookback, self.horizon
        x = self.segment[i : i + L]
        y = self.segment[i + L : i + L + h]
        return torch.from_numpy(x), torch.from_numpy(y)
