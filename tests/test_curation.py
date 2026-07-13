import numpy as np
import pytest

from src.data.covariate import CovariateSeries
from src.data.curation import build_gefcom_load, build_gefcom_wind
from src.theory.lps import calendar_features, lps


def test_gefcom_wind_contract():
    df = build_gefcom_wind()
    assert list(df.columns[:1]) == ["y"]
    assert df["y"].between(0, 1).all(), "capacity-normalized power in [0,1]"
    assert not df.isna().any().any()
    assert df.index.is_monotonic_increasing


def test_gefcom_load_contract():
    df = build_gefcom_load()
    assert (df["y"] > 0).all()
    assert df.index.freq is None or df.index.is_monotonic_increasing
    assert len(df) > 24 * 365 * 5, "multi-year load history expected"


def test_covariate_series_windows_and_scaling():
    ds = CovariateSeries("gefcom_wind", lookback=96, horizon=24)
    x, y = ds.windows("train")[0]
    assert x.shape == (96, 1) and y.shape == (24, 1)
    train_seg = ds.y[: ds.t1]
    np.testing.assert_allclose(train_seg.mean(), 0, atol=1e-9)
    np.testing.assert_allclose(train_seg.std(), 1, atol=1e-9)


def test_lps_sanity_bounds():
    rng = np.random.default_rng(0)
    T, w = 20_000, 96
    x = rng.normal(size=T)
    smooth = np.convolve(x, np.ones(200) / 200, mode="same")
    y_pred = smooth + rng.normal(0, 0.01, T)  # level ~ covariate
    r = lps(y_pred, smooth, w)
    assert r["lps"] > 0.8
    y_noise = np.cumsum(rng.normal(size=T)) * 0.05  # level ⟂ covariate
    r2 = lps(y_noise, smooth, w)
    assert r2["lps"] < 0.3


def test_calendar_features_shape():
    import pandas as pd

    idx = pd.date_range("2021-01-01", periods=100, freq="h")
    f = calendar_features(idx)
    assert f.shape == (100, 6) and np.isfinite(f).all()
