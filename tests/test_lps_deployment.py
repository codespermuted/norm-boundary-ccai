"""Deployment-variant LPS: the legacy path is untouched, the new path cannot
see past the forecast origin.

Two obligations, one test file:

  (1) `eval_end=None` must remain the frozen, pre-registered protocol. Nothing
      here may move `results/lps_official.csv`.
  (2) `eval_end=origin` must be a function of the pre-origin rows ALONE. The
      sharp test is not "the number is different" but "perturbing everything
      at and after the origin leaves it bit-identical", which is what a
      practitioner computing the screen before gate closure actually gets.
"""

from __future__ import annotations

import lightgbm as _lgb

# Same rationale as tests/test_canaries.py: the wrapper's all-cores default
# thrashes on these tiny frames.
_OrigLGBM = _lgb.LGBMRegressor


class _SingleThreadLGBM(_OrigLGBM):
    def __init__(self, **kw):
        kw.setdefault("n_jobs", 1)
        super().__init__(**kw)


_lgb.LGBMRegressor = _SingleThreadLGBM

import numpy as np
import pytest

from src.theory.lps import lps

pytestmark = pytest.mark.filterwarnings("ignore:X does not have valid feature names")

W = 24
N = 96 * W  # 96 non-overlapping windows


@pytest.fixture(scope="module")
def series():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(N, 2))
    y = 1.7 * x[:, 0] - 0.4 * x[:, 1] + 0.3 * rng.normal(size=N)
    return y, x


def test_eval_end_none_is_the_legacy_path(series):
    y, x = series
    assert lps(y, x, W)["lps"] == lps(y, x, W, eval_end=None)["lps"]


def test_eval_end_at_series_length_is_a_noop(series):
    """Cutting at the end of the series is the full variant by definition."""
    y, x = series
    assert lps(y, x, W, eval_end=len(y))["lps"] == pytest.approx(
        lps(y, x, W)["lps"], abs=1e-12)


def test_deployment_ignores_everything_at_and_after_the_origin(series):
    """The leakage test. Corrupt the post-origin tail beyond recognition; the
    deployment score must not move by a single bit, while the full score does."""
    y, x = series
    origin = 60 * W
    y2, x2 = y.copy(), x.copy()
    rng = np.random.default_rng(1)
    y2[origin:] = 50.0 + 10.0 * rng.normal(size=N - origin)
    x2[origin:] = rng.normal(size=(N - origin, 2))

    dep = lps(y, x, W, eval_end=origin)["lps"]
    dep_corrupt = lps(y2, x2, W, eval_end=origin)["lps"]
    assert dep == pytest.approx(dep_corrupt, abs=1e-12)

    full = lps(y, x, W)["lps"]
    full_corrupt = lps(y2, x2, W)["lps"]
    assert abs(full - full_corrupt) > 1e-3, (
        "the full variant must be sensitive to the post-origin tail -- that "
        "sensitivity is the reviewer's objection, and it has to be real")


def test_deployment_is_the_same_protocol_on_the_prefix(series):
    """Not a truncation of the frozen folds: the identical estimator, re-run on
    the prefix. Equivalent to handing lps() a series that simply ends there."""
    y, x = series
    origin = 60 * W
    assert lps(y, x, W, eval_end=origin)["lps"] == pytest.approx(
        lps(y[:origin], x[:origin], W)["lps"], abs=1e-12)


def test_metadata_records_the_variant(series):
    y, x = series
    full = lps(y, x, W)
    dep = lps(y, x, W, eval_end=60 * W)
    assert full["variant"] == "full" and full["eval_end"] is None
    assert dep["variant"] == "deployment" and dep["eval_end"] == 60 * W
    assert dep["n_windows"] == 60 and full["n_windows"] == 96


def test_too_few_pre_origin_windows_raises(series):
    y, x = series
    with pytest.raises(ValueError, match="expanding folds"):
        lps(y, x, W, eval_end=5 * W)


def test_origin_outside_the_series_raises(series):
    y, x = series
    with pytest.raises(ValueError, match="outside series"):
        lps(y, x, W, eval_end=len(y) + 1)
    with pytest.raises(ValueError, match="outside series"):
        lps(y, x, W, eval_end=-1)


# --------------------------------------------------------------------------
# Origin resolution against the real frames (skipped if curated data absent).
# --------------------------------------------------------------------------
def _panel():
    return pytest.importorskip("experiments.compute_lps_deployment")


@pytest.mark.parametrize("name", ["jeju_wind", "gefcom_wind"])
def test_wind_sets_are_already_pre_origin(name):
    """Both wind sets' LPS runs on the longest contiguous segment, which ends
    before the grid's test segment begins (jeju: the 2023-06-25..07-04 archive
    hole; gefcom_wind: 42 segments). For these two the deployment variant is
    numerically identical by construction, and the script must say so rather
    than quietly recompute the same number."""
    mod = _panel()
    try:
        s = mod.panel_series(name)
    except (FileNotFoundError, OSError) as e:
        pytest.skip(f"curated data unavailable: {e}")
    assert s["coincides"], "expected the LPS frame to end before the origin"
    assert s["eval_end"] >= s["n_rows"]


@pytest.mark.parametrize("name", ["etth1", "electricity", "gefcom_load",
                                  "gefcom_solar"])
def test_origin_is_strictly_inside_the_lps_frame(name):
    mod = _panel()
    try:
        s = mod.panel_series(name)
    except (FileNotFoundError, OSError) as e:
        pytest.skip(f"data unavailable: {e}")
    assert 0 < s["eval_end"] < s["n_rows"]
    assert not s["coincides"]
    assert s["eval_end"] // 96 >= 6, "too few pre-origin windows for 5 folds"
    assert s["index"][s["eval_end"] - 1] < s["origin_ts"], (
        "the last row entering the deployment variant must predate the origin")
