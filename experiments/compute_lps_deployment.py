"""DEPLOYMENT-variant LPS for the eight panel datasets and the ten GEFCom zones.

Why this exists
---------------
The frozen, pre-registered LPS (`compute_lps_official.py`, commit `cab17c1`)
places its five evaluation blocks over the final 60% of the non-overlapping
windows of the *whole* series, so the blocks span the segment the forecasting
grid later scores on. The screen never touches a model output or a split
boundary, and it is computed before any forecaster is trained -- but "before
training" is not "before the forecast origin", and the paper's framing invites
the stronger reading. A reviewer is right to ask for the stronger quantity.

This script computes it. For every series it resolves the *forecast origin* --
the first row of the split the audited grid tests on, taken from the grid's own
split code, never re-derived here -- and re-runs the identical LPS protocol on
the pre-origin prefix (`src.theory.lps.lps(..., eval_end=origin)`). Both
variants are emitted side by side so the sign predictions can be re-read.

The full variant is recomputed here too, and `--check-legacy` asserts it
reproduces `results/lps_official.csv` / `results/graded_lps_lps.csv` to 4 dp.
That is the guard that this file changed nothing frozen.

What this script does NOT do
----------------------------
It does not re-run the forecasting grid, does not touch any frozen MSE, and
does not re-issue the pre-registered sign predictions. The predictions stay as
committed on 2026-07-13; this script only asks whether the *screen* that
produced them survives being confined to pre-origin data.

Usage
-----
  uv run python -m experiments.compute_lps_deployment --dry-run     # origins only, no fit
  uv run python -m experiments.compute_lps_deployment --check-legacy
  uv run python -m experiments.compute_lps_deployment               # panel (8 datasets)
  uv run python -m experiments.compute_lps_deployment --zones       # 10 GEFCom-Wind zones
  uv run python -m experiments.compute_lps_deployment --all         # both, the paper's base

Outputs
-------
  results/lps_deployment.csv        eight panel datasets, both variants
  results/lps_deployment_zones.csv  ten GEFCom-Wind zones, both variants
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

from experiments.graded_lps import ZONE_GLOB, zone_frame
from src.data.covariate import longest_contiguous
from src.data.curation import BUILDERS
from src.data.etth import _VAL_END, load_etth_frame
from src.data.ltsf import load_ltsf_frame
from src.theory.lps import calendar_features, lps

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT_PANEL = os.path.join(ROOT, "results", "lps_deployment.csv")
OUT_ZONES = os.path.join(ROOT, "results", "lps_deployment_zones.csv")
FROZEN_PANEL = os.path.join(ROOT, "results", "lps_official.csv")
FROZEN_ZONES = os.path.join(ROOT, "results", "graded_lps_lps.csv")

W = 96
TAU = 0.3  # pre-registered 2026-07-13; NOT re-tuned here.

# Pre-registered sign predictions, verbatim from paper/predictions.md @ cab17c1.
# Read-only: this script reports whether the deployment variant would have kept
# each side of tau, it never edits a prediction.
PREREG_SIDE = {
    "jeju_wind": "+", "gefcom_wind": "+", "gefcom_load": "+", "gefcom_solar": "+",
    "etth1": "-", "etth2": "-", "electricity": "-", "weather": "-",
}

MULTIVARIATE = ("etth1", "etth2", "electricity", "weather")
CURATED = ("jeju_wind", "gefcom_wind", "gefcom_load", "gefcom_solar")


# --------------------------------------------------------------------------
# forecast origins -- read off the grid's own split code, never re-derived
# --------------------------------------------------------------------------
def panel_series(name: str) -> dict:
    """Frame the official LPS consumes, plus the forecast origin in ITS rows.

    The origin is the first row of the test segment as `experiments.g4_grid`
    builds it:
      etth*        border split, test starts at _VAL_END (12+4 months)
      ltsf         ratio split, test starts at int(T * 0.8)
      curated      ratio split, test starts at int(T_full * 0.8)

    For the curated sets the grid indexes the FULL builder frame while the LPS
    indexes `longest_contiguous(...)` of it, so the origin is carried across by
    timestamp: `eval_end` counts the LPS rows strictly before the origin's
    timestamp. If the longest contiguous segment ends before the origin the two
    variants coincide by construction, and `coincides` records that.
    """
    if name in ("etth1", "etth2"):
        df = load_etth_frame(name.replace("etth", "ETTh"))
        X = calendar_features(df.index)
        y = df.values                       # (T, C), per-channel LPS
        origin_row = _VAL_END
        origin_ts = df.index[origin_row]
        cols = list(df.columns)
    elif name in ("electricity", "weather"):
        df = load_ltsf_frame(name)
        X = calendar_features(df.index)
        y = df.values
        origin_row = int(len(df) * 0.8)
        origin_ts = df.index[origin_row]
        cols = list(df.columns)
    elif name in CURATED:
        full = BUILDERS[name]()
        origin_row_full = int(len(full) * 0.8)
        origin_ts = full.index[origin_row_full]
        df = longest_contiguous(full)       # what compute_lps_official.py uses
        cal = calendar_features(df.index)
        cov = df.drop(columns=["y"]).values if df.shape[1] > 1 else None
        X = cal if cov is None else np.column_stack([cov, cal])
        y = df[["y"]].values
        origin_row = int(df.index.searchsorted(origin_ts))
        cols = ["y"]
    else:
        raise KeyError(name)
    return {"name": name, "y": y, "X": X, "index": df.index,
            "eval_end": int(origin_row), "origin_ts": origin_ts,
            "n_rows": len(df), "channels": cols,
            "covariates": "exog+cal" if name in CURATED and X.shape[1] > 6 else "cal",
            "coincides": int(origin_row) >= len(df)}


def zone_series(path: str) -> dict:
    """Zone frame + origin. `zone_frame` already carries the grid's t2."""
    fr = zone_frame(path)
    X = np.column_stack([fr["exog"], calendar_features(fr["index"])])
    return {"name": fr["name"], "zone": fr["zid"], "y": fr["values"],
            "X": X, "index": fr["index"], "eval_end": int(fr["t2"]),
            "origin_ts": fr["index"][fr["t2"]], "n_rows": len(fr["values"]),
            "channels": ["y"], "covariates": "exog+cal", "coincides": False}


# --------------------------------------------------------------------------
def score(s: dict, eval_end: int | None, model: str = "lgbm") -> dict:
    """Channel-averaged LPS, matching compute_lps_official.py's aggregation."""
    per_ch = [lps(s["y"][:, c], s["X"], W, model=model, eval_end=eval_end)
              for c in range(s["y"].shape[1])]
    return {"lps": float(np.mean([r["lps"] for r in per_ch])),
            "n_windows": int(np.mean([r["n_windows"] for r in per_ch])),
            "n_channels": len(per_ch)}


def side(v: float) -> str:
    return "+" if v >= TAU else "-"


def row_for(s: dict, model: str = "lgbm") -> dict:
    full = score(s, None, model)
    dep = score(s, s["eval_end"], model)
    r = {"dataset": s["name"], "n_rows": s["n_rows"],
         "forecast_origin_row": s["eval_end"],
         "forecast_origin_ts": str(s["origin_ts"]),
         "origin_frac": round(s["eval_end"] / s["n_rows"], 4),
         "variants_coincide": bool(s["coincides"]),
         "n_channels": full["n_channels"],
         "covariates": s["covariates"],
         "lps_full": round(full["lps"], 4),
         "lps_deployment": round(dep["lps"], 4),
         "delta": round(dep["lps"] - full["lps"], 4),
         "n_windows_full": full["n_windows"],
         "n_windows_deployment": dep["n_windows"],
         "side_full": side(full["lps"]),
         "side_deployment": side(dep["lps"])}
    r["side_agrees"] = r["side_full"] == r["side_deployment"]
    if s["name"] in PREREG_SIDE:
        r["prereg_side"] = PREREG_SIDE[s["name"]]
        r["deployment_matches_prereg"] = r["side_deployment"] == PREREG_SIDE[s["name"]]
    return r


# --------------------------------------------------------------------------
def cmd_dry_run() -> None:
    """Resolve every forecast origin and window budget without fitting."""
    rows = []
    for name in MULTIVARIATE + CURATED:
        s = panel_series(name)
        n_dep = s["eval_end"] // W
        rows.append({"series": name, "rows": s["n_rows"],
                     "origin_row": s["eval_end"],
                     "origin_ts": str(s["origin_ts"]),
                     "frac": round(s["eval_end"] / s["n_rows"], 3),
                     "win_full": s["n_rows"] // W, "win_dep": n_dep,
                     "folds_ok": n_dep >= 6, "coincide": s["coincides"],
                     "channels": s["y"].shape[1]})
    for path in sorted(glob.glob(ZONE_GLOB)):
        s = zone_series(path)
        n_dep = s["eval_end"] // W
        rows.append({"series": s["name"], "rows": s["n_rows"],
                     "origin_row": s["eval_end"],
                     "origin_ts": str(s["origin_ts"]),
                     "frac": round(s["eval_end"] / s["n_rows"], 3),
                     "win_full": s["n_rows"] // W, "win_dep": n_dep,
                     "folds_ok": n_dep >= 6, "coincide": s["coincides"],
                     "channels": 1})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    bad = df[~df.folds_ok]
    if len(bad):
        print(f"\nWARNING: {len(bad)} series have <6 pre-origin windows: "
              f"{list(bad.series)}")
    coin = df[df.coincide]
    if len(coin):
        print(f"\nNOTE: variants coincide by construction on: {list(coin.series)}")
    print(f"\n{len(df)} series resolved; no model was fitted.")


def cmd_check_legacy() -> None:
    """The full variant must still reproduce the frozen CSVs to 4 dp."""
    frozen = pd.read_csv(FROZEN_PANEL).set_index("dataset").lps.to_dict()
    ok = True
    for name in MULTIVARIATE + CURATED:
        got = round(score(panel_series(name), None)["lps"], 4)
        good = abs(got - frozen[name]) <= 1e-4
        ok &= good
        print(f"{'OK ' if good else 'XX '}{name:14s} frozen {frozen[name]:+.4f}  "
              f"recomputed {got:+.4f}")
    fz = pd.read_csv(FROZEN_ZONES).set_index("name").lps.to_dict()
    for path in sorted(glob.glob(ZONE_GLOB)):
        s = zone_series(path)
        got = round(score(s, None)["lps"], 4)
        good = abs(got - fz[s["name"]]) <= 1e-4
        ok &= good
        print(f"{'OK ' if good else 'XX '}{s['name']:14s} frozen {fz[s['name']]:+.4f}  "
              f"recomputed {got:+.4f}")
    print("\nlegacy path intact" if ok else "\nLEGACY PATH CHANGED -- STOP")
    raise SystemExit(0 if ok else 1)


def _summarize(df: pd.DataFrame, label: str) -> None:
    print(f"\n{label}: {int(df.side_agrees.sum())}/{len(df)} series keep their "
          f"side of tau={TAU} under the deployment variant.")
    flip = df[~df.side_agrees]
    if len(flip):
        print("SIDE FLIPS (report these in the body, do not bury them):")
        print(flip[["dataset", "lps_full", "lps_deployment", "side_full",
                    "side_deployment"]].to_string(index=False))
    near = df[(df.lps_deployment.abs() - TAU).abs() < 0.1]
    if len(near):
        print(f"\nWithin 0.1 of tau under the deployment variant: "
              f"{list(near.dataset)}")


def cmd_panel() -> pd.DataFrame:
    rows = [row_for(panel_series(n)) for n in MULTIVARIATE + CURATED]
    for r in rows:
        print(r, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_PANEL, index=False)
    print("\n" + df[["dataset", "lps_full", "lps_deployment", "delta",
                     "side_full", "side_deployment", "prereg_side",
                     "deployment_matches_prereg"]].to_string(index=False))
    _summarize(df, "Panel")
    print(f"\nwrote {OUT_PANEL}")
    return df


def cmd_zones() -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(ZONE_GLOB)):
        rows.append(row_for(zone_series(path)))
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows).sort_values("lps_full")
    df.to_csv(OUT_ZONES, index=False)
    print("\n" + df[["dataset", "lps_full", "lps_deployment", "delta",
                     "side_full", "side_deployment"]].to_string(index=False))
    _summarize(df, "Zones")
    print(f"\nwrote {OUT_ZONES}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve forecast origins and window budgets, fit nothing")
    ap.add_argument("--check-legacy", action="store_true",
                    help="assert the full variant still reproduces the frozen CSVs")
    ap.add_argument("--zones", action="store_true", help="ten GEFCom-Wind zones")
    ap.add_argument("--all", action="store_true", help="panel and zones")
    a = ap.parse_args()
    if a.dry_run:
        return cmd_dry_run()
    if a.check_legacy:
        return cmd_check_legacy()
    if a.zones and not a.all:
        cmd_zones()
        return
    cmd_panel()
    if a.all:
        cmd_zones()


if __name__ == "__main__":
    main()
