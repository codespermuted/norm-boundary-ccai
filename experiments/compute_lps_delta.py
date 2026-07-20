"""ΔLPS (incremental LPS) — companion diagnostic to the official LPS.

Same protocol as compute_lps_official.py (w=96, LightGBM first stage,
expanding chronological CV, per-channel mean), but measuring the covariate
R^2 gain BEYOND persistence (HANDOFF 2026-07-16 SMP discussion):

    dLPS = R2(ybar_next ~ xbar_next + ybar_past) - R2(ybar_next ~ ybar_past)

The pre-registered decision rule stays absolute LPS; dLPS is reported in
Tab 3 as the refinement for strongly persistent series.

Usage: uv run python -m experiments.compute_lps_delta
"""

import os

import lightgbm as _lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# Force single-threaded LightGBM: the window-mean datasets are tiny
# (hundreds of rows), where the wrapper's all-cores default thrashes on
# thread sync (minutes per fit instead of milliseconds). Patching the
# module attribute is picked up by src.theory.lps._fit_predict.
_OrigLGBM = _lgb.LGBMRegressor


class _SingleThreadLGBM(_OrigLGBM):
    def __init__(self, **kw):
        kw.setdefault("n_jobs", 1)
        super().__init__(**kw)


_lgb.LGBMRegressor = _SingleThreadLGBM

from src.data.covariate import longest_contiguous
from src.data.curation import BUILDERS
from src.data.etth import load_etth_frame
from src.data.ltsf import load_ltsf_frame
from src.theory.lps import calendar_features, delta_lps

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "results", "lps_delta.csv")
W = 96
N_JOBS = int(os.environ.get("LPS_JOBS", "12"))

MULTIVARIATE = {
    "etth1": lambda: load_etth_frame("ETTh1"),
    "etth2": lambda: load_etth_frame("ETTh2"),
    "weather": lambda: load_ltsf_frame("weather"),
    "electricity": lambda: load_ltsf_frame("electricity"),
}
CURATED = ("jeju_wind", "gefcom_wind", "gefcom_load", "gefcom_solar")


def _flush(rows):
    pd.DataFrame(rows).round(4).to_csv(OUT, index=False)


def main():
    rows = []
    for name in CURATED:
        df = longest_contiguous(BUILDERS[name]())
        cal = calendar_features(df.index)
        cov = df.drop(columns=["y"]).values if df.shape[1] > 1 else None
        X = cal if cov is None else np.column_stack([cov, cal])
        r = delta_lps(df["y"].values, X, W)
        rows.append({"dataset": name, "delta_lps": r["delta_lps"],
                     "r2_full": r["r2_full"],
                     "r2_persistence": r["r2_persistence"], "n_channels": 1})
        print(rows[-1], flush=True)
        _flush(rows)
    for name, loader in MULTIVARIATE.items():
        df = loader()
        cal = calendar_features(df.index)
        per = Parallel(n_jobs=N_JOBS)(
            delayed(delta_lps)(df[c].values, cal, W) for c in df.columns)
        rows.append({
            "dataset": name,
            "delta_lps": float(np.mean([p["delta_lps"] for p in per])),
            "r2_full": float(np.mean([p["r2_full"] for p in per])),
            "r2_persistence": float(np.mean([p["r2_persistence"]
                                             for p in per])),
            "n_channels": df.shape[1]})
        print(rows[-1], flush=True)
        _flush(rows)
    print(pd.read_csv(OUT).to_string(index=False))


if __name__ == "__main__":
    main()
