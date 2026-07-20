"""Leakage canaries (G7 audit item 2) — end-to-end leak detectors.

Each canary plants a known leak (or known non-signal) in small synthetic data
and asserts the pipeline responds the way a leak-free pipeline must:

(a) white-noise covariates: CondNorm's first stage fed pure noise must show
    OOS R^2 <= 0 and CondNorm must degenerate to Raw (restoration collapses
    toward the global mean); with the TRUE covariates it must clearly win.
(b) circular-shift: official LPS on an exogenous-driven series (lam=0.8) is
    high; circularly shifting the covariates destroys the alignment, so LPS
    computed identically must collapse. Deliberately imports only the frozen
    src/theory/lps.py — never src/theory/lps_inference.py (separate track).
(c) shuffled-split signature: splitting windows randomly instead of
    chronologically leaks neighboring windows across the split, so 'test'
    loss becomes dramatically better. Our real loaders must NOT show this:
    SegmentedWindowDataset + chronological splits on ETTh1 are asserted to
    have disjoint train/test index ranges.

All on CPU with fixed seeds; total runtime well under 4 minutes.
"""

from __future__ import annotations

import lightgbm as _lgb

# Single-threaded LightGBM: on these tiny datasets the wrapper's all-cores
# default thrashes on thread sync (see experiments/compute_lps_delta.py).
# Patching the module attribute is picked up by src.theory.lps and
# src.norms.condnorm, which import lightgbm lazily inside their functions.
_OrigLGBM = _lgb.LGBMRegressor


class _SingleThreadLGBM(_OrigLGBM):
    def __init__(self, **kw):
        kw.setdefault("n_jobs", 1)
        super().__init__(**kw)


_lgb.LGBMRegressor = _SingleThreadLGBM

import numpy as np
import pytest
import torch

from src.data.covariate import SegmentedWindowDataset, segment_ids
from src.models.rlinear import RLinear
from src.norms.condnorm import CondNormTransform, first_stage_level
from src.synth.dgp import generate_series
from src.theory.lps import lps

torch.set_num_threads(2)  # small matmuls; all-cores torch thrashes too

pytestmark = [
    pytest.mark.canary,
    # cosmetic sklearn notice triggered by the n_jobs subclass above
    pytest.mark.filterwarnings("ignore:X does not have valid feature names"),
]

L, H = 48, 24  # tiny window geometry shared by canaries (a) and (c)


# ---------------------------------------------------------------- helpers
def _stack(ds: SegmentedWindowDataset):
    xs = torch.stack([ds[i][0] for i in range(len(ds))])
    ys = torch.stack([ds[i][1] for i in range(len(ds))])
    return xs, ys


def _fit(model, xs, ys, epochs, seed, lr=1e-3, bs=64):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    n = len(xs)
    for _ in range(epochs):
        for idx in np.array_split(rng.permutation(n), max(1, n // bs)):
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(model(xs[idx]), ys[idx])
            loss.backward()
            opt.step()


def _train_predict(series: np.ndarray, t_train_end: int, t_test_start: int,
                   seed: int, epochs: int = 5):
    """Tiny RLinear on chronological train windows; predict all test windows.

    Returns (preds (N, H), positions (N, H) absolute target indices)."""
    s32 = series.astype(np.float32)
    seg = np.zeros(len(s32), dtype=int)
    tr = SegmentedWindowDataset(s32, seg, 0, t_train_end, L, H)
    te = SegmentedWindowDataset(s32, seg, t_test_start - L, len(s32), L, H)
    xs, ys = _stack(tr)
    torch.manual_seed(seed)
    model = RLinear(L, H, 1)
    _fit(model, xs, ys, epochs, seed)
    xte, _ = _stack(te)
    with torch.no_grad():
        preds = model(xte).squeeze(-1).numpy().astype(np.float64)
    positions = te.starts[:, None] + L + np.arange(H)[None, :]
    return preds, positions


# ------------------------------------------- canary (a): noise covariates
@pytest.fixture(scope="module")
def noise_canary():
    """Interaction-free level DGP (src/synth/dgp.py M1) with three arms:
    Raw, CondNorm(noise covariates), CondNorm(true covariates)."""
    T = 4000
    s = generate_series(0.8, seed=0, T=T)
    y, x_true = s["y"], s["x"][:, None]
    t1, t2 = int(T * 0.6), int(T * 0.8)  # chronological 6:2:2
    noise = np.random.default_rng(123).normal(0, 1, (T, 2))

    # first-stage validation R^2 (train-mean baseline, as in LPS)
    lvl_noise = first_stage_level(noise, y, t1)
    val = slice(t1, t2)
    base = y[:t1].mean()
    r2_noise = 1 - (np.sum((y[val] - lvl_noise[val]) ** 2)
                    / np.sum((y[val] - base) ** 2))

    # Raw arm: global z-score from train stats, MSE in original units
    mu, sd = y[:t1].mean(), y[:t1].std()
    preds_z, pos = _train_predict((y - mu) / sd, t1, t2, seed=0)
    mse_raw = float(np.mean((preds_z * sd + mu - y[pos]) ** 2))

    def cn_mse(feats):
        lvl = first_stage_level(feats, y, t1)
        tr = CondNormTransform(y, lvl, train_end=t1)
        preds_r, pos = _train_predict(tr.transform(y), t1, t2, seed=0)
        return float(np.mean((tr.inverse(preds_r, pos) - y[pos]) ** 2))

    return {"r2_noise": r2_noise, "mse_raw": mse_raw,
            "mse_cn_noise": cn_mse(noise), "mse_cn_true": cn_mse(x_true)}


def test_noise_covariates_first_stage_r2_nonpositive(noise_canary):
    """Pure-noise covariates must yield no OOS level skill; positive R^2
    here would mean the first stage sees the future."""
    assert noise_canary["r2_noise"] <= 0.0, noise_canary


def test_noise_covariates_condnorm_degenerates_to_raw(noise_canary):
    """Uninformative covariates -> restoration collapses toward the global
    mean, so CondNorm test MSE must sit within 15% of Raw's."""
    raw, cn = noise_canary["mse_raw"], noise_canary["mse_cn_noise"]
    assert abs(cn - raw) <= 0.15 * raw, noise_canary


def test_true_covariates_condnorm_beats_raw(noise_canary):
    """Sanity direction: with the TRUE covariates (lam=0.8) CondNorm must
    clearly beat Raw — otherwise the canary itself is broken."""
    assert noise_canary["mse_cn_true"] < 0.5 * noise_canary["mse_raw"], noise_canary


# --------------------------------------- canary (b): circular-shift LPS
def test_circular_shift_collapses_lps():
    """Official LPS (src/theory/lps.py, w=96, expanding CV) on an
    exogenous-driven series is high; the same covariates circularly
    shifted by half the series must collapse it."""
    s = generate_series(0.8, seed=1, T=12_000)
    orig = lps(s["y"], s["x"], w=96)["lps"]
    shifted = lps(s["y"], np.roll(s["x"], 6_000), w=96)["lps"]
    assert shifted < 0.1 < orig, f"orig={orig:.3f} shifted={shifted:.3f}"


# ------------------------------------ canary (c): shuffled-split leakage
class _TinyMLP(torch.nn.Module):
    """Enough capacity to overfit — that is what makes shuffle leakage
    visible (a leak-free chronological split cannot benefit from it)."""

    def __init__(self, hidden: int = 1024):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(L, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, H))

    def forward(self, x):  # (B, L, 1) -> (B, H, 1)
        return self.net(x.squeeze(-1)).unsqueeze(-1)


def test_shuffled_window_split_leaks():
    """Randomly splitting windows leaks neighbors across the split: the
    'test' loss becomes dramatically better than under the chronological
    split of the SAME data. If the real pipeline ever showed this
    signature, it would mean leakage."""
    s = generate_series(0.3, seed=2, T=1500, sigma0=0.2)
    y = s["y"]
    z = ((y - y.mean()) / y.std()).astype(np.float32)
    full = SegmentedWindowDataset(z, np.zeros(len(z), dtype=int),
                                  0, len(z), L, H)
    xs, ys = _stack(full)
    n = len(xs)
    cut = int(n * 0.8)

    def split_test_loss(order):
        tr, te = order[:cut], order[cut:]
        torch.manual_seed(3)
        model = _TinyMLP()
        _fit(model, xs[tr], ys[tr], epochs=150, seed=3, bs=128)
        with torch.no_grad():
            return float(torch.nn.functional.mse_loss(model(xs[te]), ys[te]))

    chrono = split_test_loss(np.arange(n))
    shuffled = split_test_loss(np.random.default_rng(7).permutation(n))
    # calibrated ratio ~0.33; 0.6 leaves a 2x margin
    assert shuffled < 0.6 * chrono, f"chrono={chrono:.3f} shuffled={shuffled:.3f}"


def test_real_loader_chronological_split_has_no_overlap():
    """The real pipeline must NOT carry the shuffle signature: on ETTh1,
    SegmentedWindowDataset with the chronological 6:2:2 bounds (the
    CovariateSeries.windows formulas) touches strictly disjoint index
    ranges for train vs test."""
    from src.data.etth import load_etth_frame

    df = load_etth_frame("ETTh1")
    y = df["OT"].values.astype(np.float32)
    seg = segment_ids(df.index)
    T = len(y)
    t1, t2 = int(T * 0.6), int(T * 0.8)
    Lr, Hr = 96, 24
    train = SegmentedWindowDataset(y, seg, 0, t1, Lr, Hr)
    test = SegmentedWindowDataset(y, seg, t2 - Lr, T, Lr, Hr)

    max_train_idx = int(train.starts.max()) + Lr + Hr - 1
    min_test_idx = int(test.starts.min())
    assert max_train_idx < min_test_idx, (max_train_idx, min_test_idx)
    # and train windows never reach past the train boundary at all
    assert max_train_idx < t1
