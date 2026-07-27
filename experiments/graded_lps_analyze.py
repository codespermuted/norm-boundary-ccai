"""Analyze graded-LPS results against the frozen pre-registration.

PRIMARY: Spearman(per-zone LPS, per-zone RevIN-CondNorm gap) across the 10 zones.
Also reports the sign-rule count, the RevIN-Raw (layer) gap, and horizon trend.
Reads results/graded_lps.csv + results/graded_lps_lps.csv. Reported as-is.
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.join(os.path.dirname(__file__), "..")
G = os.path.join(ROOT, "results", "graded_lps.csv")

# instance-normalization arm per backbone (RevIN analogue for LightGBM = winz)
IN_ARM = {"rlinear": "revin", "lgbm_dms": "winz"}


def main():
    d = pd.read_csv(G)
    lps = d.groupby("zone")["lps"].first()

    # per-(zone,backbone,h) cell means over seeds, then per-zone mean over cells
    cell = d.groupby(["zone", "backbone", "h", "norm"])["mse"].mean().reset_index()

    def gap_in_minus_cn(cells):
        # within each (zone,backbone,h): IN arm minus condnorm, then zone-mean
        rows = []
        for (z, bb, h), g in cells.groupby(["zone", "backbone", "h"]):
            m = g.set_index("norm")["mse"]
            if "condnorm" not in m or IN_ARM[bb] not in m:
                continue
            rows.append({"zone": z, "backbone": bb, "h": h,
                         "gap": m[IN_ARM[bb]] - m["condnorm"]})
        return pd.DataFrame(rows)

    # PRIMARY is RLinear-only: LightGBM-DMS was deferred (disclosed deviation in
    # RESULTS.md §4a), so pooling its partial rows would mix ragged zone coverage
    # into the pre-registered endpoint. Matches results/graded_lps_zonegaps.csv.
    gaps = gap_in_minus_cn(cell[cell.backbone == "rlinear"])
    per_zone_gap = gaps.groupby("zone")["gap"].mean()
    pooled = gap_in_minus_cn(cell)
    if len(pooled) > len(gaps):
        rho_pool, p_pool = spearmanr(
            lps, pooled.groupby("zone")["gap"].mean().reindex(lps.index))
        print(f"[info] non-RLinear rows present; pooled-backbone Spearman = "
              f"{rho_pool:+.3f} (p={p_pool:.3f}) — NOT the primary endpoint")

    # RevIN - Raw (the layer, RLinear only)
    rl = cell[cell.backbone == "rlinear"]
    lay = []
    for (z, h), g in rl.groupby(["zone", "h"]):
        m = g.set_index("norm")["mse"]
        if "revin" in m and "raw" in m:
            lay.append({"zone": z, "gap": m["revin"] - m["raw"]})
    per_zone_layer = pd.DataFrame(lay).groupby("zone")["gap"].mean()

    tab = pd.DataFrame({"lps": lps, "gap_in_cn": per_zone_gap,
                        "gap_revin_raw": per_zone_layer}).dropna().sort_values("lps")

    print("=== per-zone (sorted by LPS) ===")
    print(tab.round(4).to_string())

    rho, p = spearmanr(tab["lps"], tab["gap_in_cn"])
    rho_l, p_l = spearmanr(tab["lps"], tab["gap_revin_raw"])
    print(f"\n=== PRIMARY: does LPS order the magnitude? (n={len(tab)} zones) ===")
    print(f"Spearman(LPS, RevIN-CondNorm gap) = {rho:+.3f}  (p={p:.3f})")
    print(f"  pre-registered prediction: > 0 (M1).  "
          f"{'CONFIRMED -> LPS is a dial within a homogeneous family' if rho > 0 and p < 0.10 else 'NOT confirmed -> LPS stays a classifier'}")
    print(f"Spearman(LPS, RevIN-Raw layer gap) = {rho_l:+.3f}  (p={p_l:.3f})")

    print("\n=== SECONDARY: sign rule (all zones LPS>tau -> CondNorm wins) ===")
    print(f"RevIN-CondNorm gap > 0: {(tab.gap_in_cn > 0).sum()}/{len(tab)} zones")
    print(f"RevIN-Raw   gap > 0 (layer): {(tab.gap_revin_raw > 0).sum()}/"
          f"{len(tab)} zones")

    print("\n=== horizon trend (IN-CondNorm gap by h, zone-mean) ===")
    ht = gaps.groupby("h")["gap"].mean()
    print(ht.round(4).to_string())
    mono = all(ht.get(a, 0) <= ht.get(b, 0) for a, b in zip([24, 96], [96, 336]))
    print(f"monotone increasing in h: {mono}")

    # reference: aggregated within-exogenous rho was -0.75 on 4 datasets
    print(f"\n[reference] aggregated cross-dataset within-exogenous Spearman = -0.75 "
          f"(4 heterogeneous points); this is {len(tab)} homogeneous zones.")


if __name__ == "__main__":
    main()
