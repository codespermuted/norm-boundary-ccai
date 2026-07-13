"""jeju_wind curated frame contract + segment-aware windowing."""

import numpy as np
import pandas as pd
import pytest

from src.data.covariate import CovariateSeries, longest_contiguous, segment_ids
from src.data.curation import build_jeju_wind


@pytest.fixture(scope="module")
def frame():
    return build_jeju_wind()


def test_contract(frame):
    assert frame.index.name == "date" and frame.index.is_monotonic_increasing
    assert not frame.isna().any().any()
    assert (frame["y"] >= 0).all()
    assert set(frame.columns) == {"y", "ws_da", "ws_d2"}
    # known KMA archive hole 2023-06-25..07-04: exactly one gap
    steps = np.diff(frame.index.values) / np.timedelta64(1, "h")
    assert (steps != 1.0).sum() == 1, "expected exactly one documented gap"


def test_nwp_correlation(frame):
    """Spatial mapping + lead matching validity: generation must correlate
    strongly with forecast wind speed, decaying with lead."""
    c_da = frame["y"].corr(frame["ws_da"])
    c_d2 = frame["y"].corr(frame["ws_d2"])
    assert c_da > 0.6, f"day-ahead corr too low: {c_da:.3f}"
    assert c_da > c_d2, "longer lead must not correlate better"


def test_windows_never_span_gap(frame):
    ds = CovariateSeries("jeju_wind", lookback=96, horizon=24)
    gap_row = int(np.where(segment_ids(frame.index) == 1)[0][0])
    span = 96 + 24
    for split in ("train", "val", "test"):
        starts = ds.windows(split).starts
        assert not ((starts < gap_row) & (starts + span > gap_row)).any()


def test_longest_contiguous(frame):
    seg = longest_contiguous(frame)
    steps = np.diff(seg.index.values) / np.timedelta64(1, "h")
    assert (steps == 1.0).all()
    assert len(seg) > 0.6 * len(frame)
