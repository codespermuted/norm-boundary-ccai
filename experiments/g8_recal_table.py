"""G8 recal — summary table + the numbers quoted in the paper.

Reads results/g8_recal.csv, writes paper/tables/tabG8_recal.md, and prints
the aggregates the manuscript cites:
  - CN cov80 before/after (mean and per-(dataset,h) range, rlinear_q),
  - whether CN retains the best pinball per cell once EVERY arm is
    recalibrated (the fairness condition),
  - val_cov80 sanity (pooled variant must sit at ~0.80 by construction),
  - lgbm_q cross-check cells (h<=48).
"""

import os

import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
CSV = os.path.join(ROOT, "results", "g8_recal.csv")
OUT = os.path.join(ROOT, "paper", "tables", "tabG8_recal.md")

ARM_ORDER = ["raw", "revin", "san", "fan", "condnorm", "winz"]


def main():
    df = pd.read_csv(CSV).drop_duplicates(
        subset=["dataset", "arm", "backbone", "h", "seed", "variant"],
        keep="last")
    cell = (df.groupby(["backbone", "dataset", "h", "arm", "variant"])
              [["pinball", "cov80", "cov_lo", "cov_hi", "val_cov80"]]
              .mean().reset_index())

    lines = ["# Tab G8 — split-conformal recalibration (validation-fitted, "
             "test-evaluated)", ""]
    for bk, sub in cell.groupby("backbone"):
        lines += [f"## {bk}", "",
                  "| dataset | h | arm | pinball none→pooled | "
                  "cov80 none→pooled | perstep cov80 | val cov80 (pooled) |",
                  "|---|---|---|---|---|---|---|"]
        for (ds, h), g in sub.groupby(["dataset", "h"]):
            p = g.pivot(index="arm", columns="variant",
                        values=["pinball", "cov80", "val_cov80"])
            for arm in [a for a in ARM_ORDER if a in p.index]:
                lines.append(
                    f"| {ds} | {h} | {arm} "
                    f"| {p.loc[arm, ('pinball', 'none')]:.4f}"
                    f"→{p.loc[arm, ('pinball', 'pooled')]:.4f} "
                    f"| {p.loc[arm, ('cov80', 'none')]:.3f}"
                    f"→{p.loc[arm, ('cov80', 'pooled')]:.3f} "
                    f"| {p.loc[arm, ('cov80', 'perstep')]:.3f} "
                    f"| {p.loc[arm, ('val_cov80', 'pooled')]:.3f} |")
        lines.append("")

    # ------------------------- paper aggregates (rlinear_q primary) -------
    rl = cell[cell.backbone == "rlinear_q"]
    cn = rl[rl.arm == "condnorm"]
    none_ = cn[cn.variant == "none"].set_index(["dataset", "h"])
    pooled = cn[cn.variant == "pooled"].set_index(["dataset", "h"])
    summary = {
        "cells": len(none_),
        "cn_cov80_before": (none_.cov80.mean(),
                            none_.cov80.min(), none_.cov80.max()),
        "cn_cov80_after": (pooled.cov80.mean(),
                           pooled.cov80.min(), pooled.cov80.max()),
        "cn_pinball_before": none_.pinball.mean(),
        "cn_pinball_after": pooled.pinball.mean(),
        "val_cov80_pooled_range": (pooled.val_cov80.min(),
                                   pooled.val_cov80.max()),
    }
    # fairness: best pinball per cell with every arm recalibrated (pooled)
    pl = rl[rl.variant == "pooled"]
    best = pl.loc[pl.groupby(["dataset", "h"]).pinball.idxmin()]
    summary["cn_best_after_recal"] = (
        int((best.arm == "condnorm").sum()), len(best))
    # pinball improvement direction per arm (does recal ever hurt?)
    for arm in sorted(rl.arm.unique()):
        a = rl[rl.arm == arm].pivot_table(
            index=["dataset", "h"], columns="variant", values="pinball")
        summary[f"{arm}_pinball_worse_cells"] = int(
            (a["pooled"] > a["none"] + 1e-6).sum())

    lg = cell[(cell.backbone == "lgbm_q") & (cell.arm == "condnorm")]
    if len(lg):
        ln = lg[lg.variant == "none"]
        lp = lg[lg.variant == "pooled"]
        summary["lgbm_cn_cov80"] = (ln.cov80.mean(), lp.cov80.mean())
        summary["lgbm_cells"] = len(ln)

    lines += ["## Paper aggregates", "", "```"]
    for k, v in summary.items():
        lines.append(f"{k}: {v}")
    lines += ["```", ""]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines[-len(summary) - 4 :]))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
