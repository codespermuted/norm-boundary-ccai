"""Graded-LPS experiment — disaggregate GEFCom-Wind into its 10 zones.

Purpose. Our aggregated panel is bimodal in LPS, so the decision rule is only
validated as a two-regime CLASSIFIER; we cannot say whether LPS orders the
MAGNITUDE of the instance-normalization penalty (within the exogenous regime the
aggregated Spearman is even negative, rho=-0.75 on 4 points). GEFCom-Wind is a
mean over 10 wind zones with genuinely different NWP skill; disaggregating gives a
HOMOGENEOUS family (same physical process, same covariate structure) with a graded
spread of level predictability. Regressing the realized RevIN-CondNorm gap on the
per-zone LPS is the clean test the aggregated data cannot provide:
  - if the gap increases with LPS across zones (Spearman > 0), LPS is a DIAL, not
    just a classifier -> fills the paper's largest gap;
  - if it is flat/negative, that is an honest negative result that resolves the
    question and is reported as-is.
Any zone landing at intermediate LPS (0.3-0.7) also stress-tests the threshold in
the empty gap of the aggregated panel.

Modeling is IDENTICAL to the frozen main grid: same LPS protocol (w=96, LightGBM,
expanding CV), same first stage (train-only LightGBM on target-time covariates),
same torch_run / lgbm_run arms; only the series changes. GEFCom covariates are
competition-provided forecasts valid at target time (no reanalysis, within the
availability envelope -> no lead-matching defect), so the exogenous handling is
sound at every horizon.

Governance: run `--lps` first, commit the per-zone LPS + predictions
(evidence/prereg_graded_lps.md) BEFORE the forecasting grid, then run `--grid`.

Usage:
  CUDA_VISIBLE_DEVICES=1 uv run python -m experiments.graded_lps --lps
  CUDA_VISIBLE_DEVICES=1 uv run python -m experiments.graded_lps --grid
"""
import argparse
import csv
import glob
import os

import numpy as np
import pandas as pd

from experiments.g4_grid import lgbm_run, torch_run
from src.data.covariate import segment_ids
from src.norms.condnorm import first_stage_level
from src.theory.lps import calendar_features, lps

ROOT = os.path.join(os.path.dirname(__file__), "..")
ZONE_GLOB = os.path.join(
    ROOT, "curated", "raw", "gefcom", "GEFCom2014 Data", "Wind",
    "Task 15", "Task15_W_Zone1_10", "Task15_W_Zone*.csv")
LPS_CSV = os.path.join(ROOT, "results", "graded_lps_lps.csv")
GRID_CSV = os.path.join(ROOT, "results", "graded_lps.csv")
W = 96
HORIZONS = (24, 96, 336)
L_RLINEAR = 336            # fixed across arms & zones (arms differ only in norm)
SEEDS = range(5)
FS_DIR = os.path.join(ROOT, "curated", "firststage", "graded")


def zone_frame(path: str) -> dict:
    """Per-zone frame in the exact shape build_frame() returns (curated kind)."""
    z = pd.read_csv(path)
    z["date"] = pd.to_datetime(z["TIMESTAMP"], format="%Y%m%d %H:%M")
    z["ws10"] = np.hypot(z["U10"], z["V10"])
    z["ws100"] = np.hypot(z["U100"], z["V100"])
    df = (z.set_index("date")[["TARGETVAR", "ws10", "ws100"]]
          .rename(columns={"TARGETVAR": "y"}).sort_index().dropna())
    df = df[~df.index.duplicated(keep="first")]
    T = len(df)
    t1, t2 = int(T * 0.6), int(T * 0.8)
    index = df.index
    values = df[["y"]].values.astype(np.float64)
    exog = df[["ws10", "ws100"]].values
    zid = int(os.path.basename(path).split("Zone")[1].split(".")[0])
    return {"name": f"gwind_z{zid:02d}", "zid": zid, "values": values,
            "seg": segment_ids(index), "t1": t1, "t2": t2,
            "index": index, "exog": exog}


def zone_level(frame: dict) -> np.ndarray:
    """CondNorm first-stage level (T,1), train-only, cached — same as firststage()."""
    os.makedirs(FS_DIR, exist_ok=True)
    p = os.path.join(FS_DIR, f"{frame['name']}.npy")
    if os.path.exists(p):
        return np.load(p)
    cal = calendar_features(frame["index"])
    feats = np.column_stack([frame["exog"], cal])
    lvl = first_stage_level(feats, frame["values"][:, 0],
                            train_end=frame["t1"])[:, None]
    np.save(p, lvl)
    return lvl


def per_zone_lps(frame: dict) -> float:
    cal = calendar_features(frame["index"])
    X = np.column_stack([frame["exog"], cal])
    return lps(frame["values"][:, 0], X, W, model="lgbm")["lps"]


def cmd_lps():
    rows = []
    for path in sorted(glob.glob(ZONE_GLOB)):
        fr = zone_frame(path)
        rows.append({"zone": fr["zid"], "name": fr["name"],
                     "lps": round(per_zone_lps(fr), 4), "T": len(fr["values"])})
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows).sort_values("lps")
    df.to_csv(LPS_CSV, index=False)
    print("\n" + df.to_string(index=False))
    print(f"\nLPS range: {df.lps.min():.3f} .. {df.lps.max():.3f}  "
          f"(aggregated gefcom_wind = 0.744)")


def cmd_grid():
    import torch
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    new = not os.path.exists(GRID_CSV)
    f = open(GRID_CSV, "a", newline="")
    fields = ["zone", "backbone", "norm", "h", "seed", "L", "mse", "lps"]
    w = csv.DictWriter(f, fieldnames=fields)
    if new:
        w.writeheader()
    done = set()
    if not new:
        for r in csv.DictReader(open(GRID_CSV)):
            done.add((r["zone"], r["backbone"], r["norm"], r["h"], r["seed"]))

    for path in sorted(glob.glob(ZONE_GLOB)):
        fr = zone_frame(path)
        lvl = zone_level(fr)
        lp = round(per_zone_lps(fr), 4)
        for h in HORIZONS:
            # RLinear: raw / revin / condnorm, 5 seeds, fixed L
            for norm in ("raw", "revin", "condnorm"):
                for seed in SEEDS:
                    key = (str(fr["zid"]), "rlinear", norm, str(h), str(seed))
                    if key in done:
                        continue
                    r = torch_run(fr, L_RLINEAR, h, "rlinear", norm, seed, dev,
                                  level=lvl if norm == "condnorm" else None)
                    w.writerow({"zone": fr["zid"], "backbone": "rlinear",
                                "norm": norm, "h": h, "seed": seed,
                                "L": L_RLINEAR, "mse": round(r["mse"], 6),
                                "lps": lp})
                    f.flush()
            # LightGBM-DMS: winz (RevIN analogue) / raw / condnorm, deterministic
            for arm in ("winz", "raw", "condnorm"):
                key = (str(fr["zid"]), "lgbm_dms", arm, str(h), "0")
                if key in done:
                    continue
                r = lgbm_run(fr, h, arm, lvl if arm == "condnorm" else None)
                w.writerow({"zone": fr["zid"], "backbone": "lgbm_dms",
                            "norm": arm, "h": h, "seed": 0, "L": 336,
                            "mse": round(r["mse"], 6), "lps": lp})
                f.flush()
        print(f"done {fr['name']} (lps={lp})", flush=True)
    f.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lps", action="store_true", help="per-zone LPS only (pre-reg)")
    ap.add_argument("--grid", action="store_true", help="forecasting grid")
    a = ap.parse_args()
    if a.lps:
        cmd_lps()
    if a.grid:
        cmd_grid()
