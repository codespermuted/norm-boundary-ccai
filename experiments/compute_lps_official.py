"""OFFICIAL LPS definition for pre-registration (fixed before the G4 grid).

  LPS(dataset) = mean over target channels of the per-channel LPS with
    - window w = 96 (matches the shortest grid horizon scale)
    - first stage g = LightGBM (same family as CondNorm's first stage)
    - expanding chronological CV (src/theory/lps.py), min_train_frac 0.4
    - covariates: the exact covariate set CondNorm will receive in the grid
      (exogenous drivers where they exist, plus calendar harmonics)
  Ridge is reported as sensitivity only; it does not enter the decision rule.

Multivariate LTSF benchmarks use per-channel LPS averaged over channels
because the grid MSE is averaged over channels. Univariate curated datasets
reduce to the single-channel LPS.

Usage: uv run python -m experiments.compute_lps_official
"""

import os

import numpy as np
import pandas as pd

from src.data.covariate import longest_contiguous
from src.data.curation import BUILDERS
from src.data.etth import load_etth_frame
from src.data.ltsf import load_ltsf_frame
from src.theory.lps import calendar_features, lps

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "results", "lps_official.csv")
W = 96

GRID_DATASETS_MULTIVARIATE = {
    "etth1": lambda: load_etth_frame("ETTh1"),
    "etth2": lambda: load_etth_frame("ETTh2"),
    "electricity": lambda: load_ltsf_frame("electricity"),
    "weather": lambda: load_ltsf_frame("weather"),
}
GRID_DATASETS_CURATED = ("jeju_wind", "gefcom_wind", "gefcom_load", "gefcom_solar")


def channel_lps(y, X, model):
    return lps(y, X, W, model=model)["lps"]


def main():
    rows = []
    for name, loader in GRID_DATASETS_MULTIVARIATE.items():
        df = loader()
        cal = calendar_features(df.index)
        per_ch = {m: [channel_lps(df[c].values, cal, m) for c in df.columns]
                  for m in ("lgbm", "ridge")}
        rows.append({"dataset": name, "lps": np.mean(per_ch["lgbm"]),
                     "lps_ridge": np.mean(per_ch["ridge"]),
                     "n_channels": df.shape[1], "covariates": "cal"})
        print(rows[-1], flush=True)
    for name in GRID_DATASETS_CURATED:
        df = longest_contiguous(BUILDERS[name]())
        cal = calendar_features(df.index)
        cov = df.drop(columns=["y"]).values if df.shape[1] > 1 else None
        X = cal if cov is None else np.column_stack([cov, cal])
        rows.append({"dataset": name,
                     "lps": channel_lps(df["y"].values, X, "lgbm"),
                     "lps_ridge": channel_lps(df["y"].values, X, "ridge"),
                     "n_channels": 1,
                     "covariates": "exog+cal" if cov is not None else "cal"})
        print(rows[-1], flush=True)
    out = pd.DataFrame(rows).round(4)
    out.to_csv(OUT, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
