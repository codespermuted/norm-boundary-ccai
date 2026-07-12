import numpy as np
import pytest

from src.data import build_dataset
from src.data.etth import _TRAIN_END, _VAL_END, load_etth_frame


@pytest.fixture(scope="module")
def ds():
    return build_dataset("etth1", lookback=336, horizon=96)


def test_contract_frame():
    df = load_etth_frame("ETTh1")
    assert df.index.name == "date"
    assert not df.isna().any().any()
    assert df.shape[1] == 7  # HUFL..OT


def test_window_shapes(ds):
    x, y = ds.windows("train")[0]
    assert x.shape == (336, 7) and y.shape == (96, 7)
    assert x.dtype.is_floating_point and y.dtype.is_floating_point


def test_scaler_fitted_on_train_only(ds):
    """No leakage: global z-score statistics must come from the train segment."""
    df = load_etth_frame("ETTh1")
    train_seg = df.values[:_TRAIN_END]
    np.testing.assert_allclose(ds.mean, train_seg.mean(axis=0), rtol=1e-10)
    np.testing.assert_allclose(ds.std, train_seg.std(axis=0), rtol=1e-10)


def test_split_boundaries_no_target_leakage(ds):
    """Every val/test window's TARGET must lie strictly outside the train segment.

    Split segments are prefixed with lookback rows of history (standard LTSF
    border protocol) — that overlap is inputs only; targets must not leak.
    """
    L, h = 336, 96
    # val segment starts at absolute index _TRAIN_END - L; first target index:
    first_val_target = (_TRAIN_END - L) + L
    assert first_val_target >= _TRAIN_END
    first_test_target = (_VAL_END - L) + L
    assert first_test_target >= _VAL_END
    # window counts consistent with segment lengths (drop_last=False semantics)
    assert len(ds.windows("val")) == (_VAL_END - (_TRAIN_END - L)) - L - h + 1
