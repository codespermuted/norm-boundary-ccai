"""G12 -- does the audited layer cost concentrate in the ramps?

Pre-registered in evidence/prereg_ramp_footing.md. NO MODEL IS TRAINED HERE:
the per-origin losses have been on disk since Block B
(results/g4_errors/{dataset}_{arm}+cov_{backbone}_{h}_{seed}.npy, aligned
element-for-element with split_starts(frame, L, h)["test"]).

The paper's climate pathway rests on the claim that the stale window mean goes
wrong "exactly at the ramps whose errors fossil-fuelled reserves absorb". That
has never been measured. Under docs/philosophy.md it becomes a number or it
leaves the paper.

Ramp statistic, fixed before any loss array was opened -- deliberately NOT the
staleness quantity, so the test cannot be circular:

    ramp(t) = max_{k=1..h-1} | y[t+L+k] - y[t+L+k-1] |

the largest realized one-hour change inside the forecast horizon. It never
references the lookback window, the window mean, or any model output. Bins are
terciles of ramp(t) computed within each (dataset, backbone, h) cell, so bin
membership cannot depend on an arm.

Output: results/g12_ramp.csv (per bin) + results/g12_ramp_cells.csv (per cell)
Usage: uv run python -m experiments.g12_ramp
"""

import csv
import os

import numpy as np
import pandas as pd

from experiments.g4_grid import build_frame, firststage, split_starts

ROOT = os.path.join(os.path.dirname(__file__), "..")
ERR = os.path.join(ROOT, "results", "g4_errors")
PARITY = os.path.join(ROOT, "results", "g4_covfair_full.csv")
OUT_BIN = os.path.join(ROOT, "results", "g12_ramp.csv")
OUT_CELL = os.path.join(ROOT, "results", "g12_ramp_cells.csv")

BACKBONES = ("linmix", "mlpmix", "lgbmcov", "segrnncov", "patchtstcov")
SEEDS = range(5)
NBINS = 3
MIN_ORIGINS = 60          # pre-registered: drop the cell and report the drop
# the seven cells with no disclosed provenance defect (GEFCom-Load's covariate
# is realized temperature; Jeju h=48 consumes the h<=24 forecast band)
CLEAN = {("gefcom_wind", 24), ("gefcom_wind", 96), ("gefcom_wind", 336),
         ("gefcom_solar", 24), ("gefcom_solar", 96), ("gefcom_solar", 336),
         ("jeju_wind", 24)}


def seed_mean_losses(dataset, arm, backbone, h):
    arrs = []
    for s in SEEDS:
        p = os.path.join(ERR, f"{dataset}_{arm}+cov_{backbone}_{h}_{s}.npy")
        if os.path.exists(p):
            arrs.append(np.load(p))
    if not arrs:
        return None
    n = min(map(len, arrs))
    return np.mean([a[:n] for a in arrs], axis=0)


def ramp_of(frame, starts, L, h):
    """max realized 1-step change inside the horizon, per origin."""
    y = frame["values"][:, 0]
    idx = starts[:, None] + L + np.arange(h)[None, :]
    seg = y[idx]
    return np.abs(np.diff(seg, axis=1)).max(axis=1)


def boot_ci(vals, B=100_000, seed=0):
    v = np.asarray(vals, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(B, len(v)))
    b = v[idx].mean(axis=1)
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    parity = pd.read_csv(PARITY)
    lb = (parity.groupby(["dataset", "backbone", "h"])["L"]
          .agg(lambda c: int(c.mode().iloc[0])).to_dict())
    cells = sorted({(d, int(h)) for d, h in
                    zip(parity.dataset, parity.h)})
    frames = {}

    bin_rows, cell_rows, dropped = [], [], []
    for backbone in BACKBONES:
        for dataset, h in cells:
            key = (dataset, backbone, h)
            if key not in lb:
                continue
            L = int(lb[key])
            lr = seed_mean_losses(dataset, "raw", backbone, h)
            lv = seed_mean_losses(dataset, "revin", backbone, h)
            if lr is None or lv is None:
                continue
            if dataset not in frames:
                fr = build_frame(dataset)
                frames[dataset] = (fr, firststage(fr))
            frame, level = frames[dataset]
            starts = np.asarray(split_starts(frame, L, h)["test"])
            n = min(len(starts), len(lr), len(lv))
            if n < MIN_ORIGINS:
                dropped.append((dataset, backbone, h, n))
                continue
            starts, lr, lv = starts[:n], lr[:n], lv[:n]
            cost = lv - lr

            ramp = ramp_of(frame, starts, L, h)
            # Equal-count terciles by rank. Value-cut terciles collapse on the
            # solar set, where a large share of origins are all-night and have
            # ramp exactly 0, leaving an empty bin. Ranking ties by origin
            # index is arbitrary but independent of every arm, so bin
            # membership still cannot depend on which arm is being scored.
            order = np.argsort(ramp, kind="stable")
            b = np.empty(n, dtype=int)
            b[order] = (np.arange(n) * NBINS) // n
            tie_frac = float((ramp == np.median(ramp)).mean())

            # secondary 3: do the bins mean what they should? level error of
            # the window mean (what RevIN restores) vs the covariate-implied
            # level, on the same origins.
            # level errors are put on the global-z scale (train sd), the same
            # scale as the losses, so they can be pooled across datasets whose
            # raw units differ (capacity factor vs MW).
            y = frame["values"][:, 0]
            sd_g = float(y[:frame["t1"]].std())
            hidx = starts[:, None] + L + np.arange(h)[None, :]
            hmean = y[hidx].mean(axis=1)
            widx = starts[:, None] + np.arange(L)[None, :]
            wmean = y[widx].mean(axis=1)
            fmean = level[:, 0][hidx].mean(axis=1)
            lvl_win = np.abs(hmean - wmean) / sd_g
            lvl_cov = np.abs(hmean - fmean) / sd_g

            per_bin = {}
            for k in range(NBINS):
                m = b == k
                per_bin[k] = float(cost[m].mean())
                bin_rows.append({
                    "dataset": dataset, "backbone": backbone, "h": h,
                    "bin": k, "n": int(m.sum()),
                    "ramp_lo": float(ramp[m].min()), "ramp_hi": float(ramp[m].max()),
                    "cost": round(float(cost[m].mean()), 6),
                    "mse_raw": round(float(lr[m].mean()), 6),
                    "mse_revin": round(float(lv[m].mean()), 6),
                    "lvl_err_windowmean": round(float(lvl_win[m].mean()), 6),
                    "lvl_err_covariate": round(float(lvl_cov[m].mean()), 6),
                })
            cell_rows.append({
                "dataset": dataset, "backbone": backbone, "h": h,
                "clean": (dataset, h) in CLEAN, "n_origins": n,
                "tie_frac_at_median_ramp": round(tie_frac, 4),
                "cost_all": round(float(cost.mean()), 6),
                "cost_lo": round(per_bin[0], 6),
                "cost_mid": round(per_bin[1], 6),
                "cost_hi": round(per_bin[NBINS - 1], 6),
                "hi_minus_lo": round(per_bin[NBINS - 1] - per_bin[0], 6),
                "monotone": bool(per_bin[0] <= per_bin[1] <= per_bin[NBINS - 1]),
            })

    pd.DataFrame(bin_rows).to_csv(OUT_BIN, index=False)
    cd = pd.DataFrame(cell_rows)
    cd.to_csv(OUT_CELL, index=False)

    # ------------------------------------------------------- endpoints
    print(f"cells analysed: {len(cd)}   dropped (<{MIN_ORIGINS} origins): {dropped}")
    for label, sel in [("CLEAN (7 cells)", cd.clean), ("ALL (11 cells)", cd.clean | True)]:
        print(f"\n===== {label} =====")
        for backbone in BACKBONES:
            s = cd[(cd.backbone == backbone) & sel]
            if s.empty:
                continue
            m, lo, hi = boot_ci(s.hi_minus_lo.values)
            star = "  <-- PRIMARY" if backbone == "linmix" and label.startswith("CLEAN") else ""
            print(f"{backbone:12s} n={len(s):2d} "
                  f"lo={s.cost_lo.mean():+.4f} mid={s.cost_mid.mean():+.4f} "
                  f"hi={s.cost_hi.mean():+.4f} | hi-lo={m:+.4f} "
                  f"[{lo:+.4f},{hi:+.4f}] mono={int(s.monotone.sum())}/{len(s)}{star}")

    print("\n--- PRIMARY, per cell (linmix, clean) ---")
    s = cd[(cd.backbone == "linmix") & cd.clean]
    print(s[["dataset", "h", "n_origins", "tie_frac_at_median_ramp",
             "cost_lo", "cost_mid", "cost_hi", "hi_minus_lo"]].to_string(index=False))

    print("\n--- secondary 3: do the bins mean what they should? "
          "(linmix, clean, global-z units) ---")
    bd = pd.DataFrame(bin_rows)
    s = bd[(bd.backbone == "linmix")
           & bd.set_index(["dataset", "h"]).index.isin(CLEAN)]
    g = s.groupby("bin")[["lvl_err_windowmean", "lvl_err_covariate"]].mean()
    g["gap"] = g.lvl_err_windowmean - g.lvl_err_covariate
    g["ratio"] = g.lvl_err_windowmean / g.lvl_err_covariate
    print(g.round(4).to_string())


if __name__ == "__main__":
    main()
