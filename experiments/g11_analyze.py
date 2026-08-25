"""G11 analysis -- the pre-registered footing endpoints.

Reads results/g11_footing.csv and evaluates the endpoints fixed in
evidence/prereg_ramp_footing.md (as amended 2026-07-29).

Usage: uv run python -m experiments.g11_analyze
"""

import os

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "results", "g11_footing.csv")
PARITY = os.path.join(ROOT, "results", "g4_covfair_full.csv")
FOOT_ORDER = ("global", "window", "window_floor", "scale")
B = 100_000


def boot(vals, seed=0):
    v = np.asarray(vals, float)
    rng = np.random.default_rng(seed)
    b = v[rng.integers(0, len(v), size=(B, len(v)))].mean(axis=1)
    return v.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def main():
    d = pd.read_csv(OUT)
    cell = (d.groupby(["backbone", "dataset", "h", "arm", "footing"]).mse
            .mean().unstack(["arm", "footing"]))
    parity = pd.read_csv(PARITY)
    cn = (parity[parity.arm == "condnorm"]
          .groupby(["backbone", "dataset", "h"]).mse.mean())

    for bb in cell.index.get_level_values(0).unique():
        c = cell.loc[bb]
        print(f"\n{'='*78}\n{bb}   ({len(c)} cells)\n{'='*78}")

        print("  absolute MSE")
        show = pd.DataFrame({f"{a[:4]}/{f[:6]}": c[(a, f)]
                             for a in ("raw", "revin") for f in FOOT_ORDER
                             if (a, f) in c.columns})
        show["CN"] = [cn.get((bb, ds, h), np.nan) for ds, h in c.index]
        print(show.round(4).to_string())

        print("\n  isolated layer toggle within each footing (revin - raw)")
        for f in FOOT_ORDER:
            if ("revin", f) not in c.columns:
                continue
            v = (c[("revin", f)] - c[("raw", f)]).dropna()
            m, lo, hi = boot(v.values)
            print(f"    {f:13s} {m:+.4f} [{lo:+.4f},{hi:+.4f}]  "
                  f"pos={int((v > 0).sum())}/{len(v)}")

        if ("revin", "scale") in c.columns and ("raw", "scale") in c.columns:
            v = (c[("revin", "scale")] - c[("raw", "scale")]).dropna()
            m, lo, hi = boot(v.values)
            print(f"\n  ORIGINAL endpoint  RevIN_S - RAW_S : {m:+.4f} "
                  f"[{lo:+.4f},{hi:+.4f}] pos={int((v > 0).sum())}/{len(v)}")

        rev_cols = [("revin", f) for f in FOOT_ORDER if ("revin", f) in c.columns]
        if rev_cols and ("raw", "global") in c.columns:
            best = c[rev_cols].min(axis=1)
            which = c[rev_cols].idxmin(axis=1).map(lambda t: t[1])
            v = (best - c[("raw", "global")]).dropna()
            m, lo, hi = boot(v.values)
            print(f"  PRIMARY (amended)  min_f RevIN_f - RAW_G : {m:+.4f} "
                  f"[{lo:+.4f},{hi:+.4f}] pos={int((v > 0).sum())}/{len(v)}")
            print("    best footing per cell:",
                  dict(which.value_counts()))

        if not cn.empty:
            cnv = pd.Series([cn.get((bb, ds, h), np.nan) for ds, h in c.index],
                            index=c.index)
            if rev_cols:
                best_endo = pd.concat([c[rev_cols].min(axis=1),
                                       c[("raw", "global")]], axis=1).min(axis=1)
                w = (best_endo - cnv).dropna()
                print(f"  SECONDARY 1  best endogenous (any footing) - CondNorm: "
                      f"{w.mean():+.4f}, CN better in {int((w > 0).sum())}/{len(w)}")

        print("\n  SECONDARY 2  what destroying the covariate level costs each arm")
        for a in ("raw", "revin"):
            for f in ("window", "window_floor"):
                if (a, f) not in c.columns:
                    continue
                v = (c[(a, f)] - c[(a, "global")]).dropna()
                print(f"    {a:5s} {f:13s} - global : {v.mean():+.4f}  "
                      f"per cell {list(np.round(v.values, 3))}")


if __name__ == "__main__":
    main()
