"""Compute LPS for every available dataset -> results/lps.csv.

Covariates per dataset (exogenous, valid at target time):
  ETT / electricity / weather : calendar harmonics only (no exogenous drivers
                                in the standard benchmark -> expected LOW LPS)
  gefcom_wind  : forecast wind speeds ws10/ws100 (+calendar)
  gefcom_load  : temperature (+calendar)
  gefcom_solar : 12 competition NWP vars (+calendar)
  jeju_wind    : lead-matched KMA WSD forecasts (+calendar) — skipped until
                 NWP collection completes
  kpx_demand_national : calendar only (temperature TBD) — short series caveat

Usage: uv run python -m experiments.compute_lps
"""

import os

import numpy as np
import pandas as pd

from src.data.curation import BUILDERS
from src.data.etth import load_etth_frame
from src.theory.lps import calendar_features, lps

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "results", "lps.csv")
WINDOWS = (96, 336)


def frames():
    for name in ("ETTh1", "ETTh2"):
        df = load_etth_frame(name)
        yield name.lower(), df["OT"].values, None, df.index
    for name, path, target in (
        ("electricity", "curated/raw/ltsf/electricity.csv", "OT"),
        ("weather", "curated/raw/ltsf/weather.csv", "OT"),
    ):
        df = pd.read_csv(os.path.join(ROOT, path), parse_dates=["date"])
        yield name, df[target].values, None, df["date"]
    for name, build in BUILDERS.items():
        if name == "jeju_wind" and not os.path.exists(
                os.path.join(ROOT, "curated", "jeju_wind.parquet")):
            done = os.path.join(ROOT, "curated", "raw", "kma", "done_keys.txt")
            n_done = sum(1 for _ in open(done)) if os.path.exists(done) else 0
            if n_done < 60_000:  # collection still running
                print(f"skip jeju_wind (NWP collection {n_done}/65808)")
                continue
        try:
            df = build()
        except Exception as exc:
            print(f"skip {name}: {str(exc)[:80]}")
            continue
        from src.data.covariate import longest_contiguous

        df = longest_contiguous(df)  # LPS windows must not span archive holes
        cov = df.drop(columns=["y"]).values if df.shape[1] > 1 else None
        yield name, df["y"].values, cov, df.index


def main():
    rows = []
    for name, y, cov, index in frames():
        cal = calendar_features(index)
        X = cal if cov is None else np.column_stack([cov, cal])
        for w in WINDOWS:
            if len(y) < w * 25:
                continue
            for model in ("lgbm", "ridge"):
                r = lps(y, X, w, model=model)
                rows.append({"dataset": name, "w": w, "model": model,
                             "lps": round(r["lps"], 4),
                             "n_windows": r["n_windows"],
                             "covariates": "exog+cal" if cov is not None else "cal"})
            print(rows[-2] | {"ridge": rows[-1]["lps"]}, flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"saved {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
