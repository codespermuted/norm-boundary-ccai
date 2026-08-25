"""Emit the CCAI footing table (Table 1) straight from results/g11_footing.csv.

The paper \\input{}s the file this writes, so the printed numbers cannot drift
from the frozen CSV.

One row per (dataset, h) cell rather than a pooled mean, deliberately: the
unfloored -- and even the floored -- window footing is numerically degenerate on
the solar set, where a large share of lookback windows are all-night and have
almost no covariate variance to divide by. Pooling would either hide that or
force a post-hoc exclusion rule. Per-cell rows let the reader see it.

Output: paper/tables/tab_footing.tex  (+ a markdown twin for the ledger)
Usage: uv run python -m experiments.g11_table [--backbone linmix]
"""

import argparse
import os

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
SRC = os.path.join(ROOT, "results", "g11_footing.csv")
TEX = os.path.join(ROOT, "paper", "tables", "tab_footing.tex")
MD = os.path.join(ROOT, "paper", "tables", "tab_footing.md")

PRETTY = {"gefcom_wind": "GEFCom-Wind", "gefcom_solar": "GEFCom-Solar",
          "jeju_wind": "Jeju Wind"}
ORDER = ["gefcom_wind", "jeju_wind", "gefcom_solar"]
TOGGLE_FOOTINGS = [("global", "global"), ("window_floor", "window"),
                   ("scale", "scale")]
B = 100_000


def boot(v, seed=0):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    b = v[rng.integers(0, len(v), size=(B, len(v)))].mean(axis=1)
    return v.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def num(x, d=3):
    return "---" if not np.isfinite(x) else f"{x:.{d}f}"


def signed(x, d=3):
    return "---" if not np.isfinite(x) else f"${x:+.{d}f}$"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="linmix")
    args = ap.parse_args()

    d = pd.read_csv(SRC)
    d = d[d.backbone == args.backbone]
    cell = (d.groupby(["dataset", "h", "arm", "footing"]).mse.mean()
            .unstack(["arm", "footing"]))
    idx = sorted(cell.index, key=lambda t: (ORDER.index(t[0]), int(t[1])))
    cell = cell.loc[idx]

    # The global-footing CondNorm cells are the frozen parity block itself
    # (same runner, same settings), so take them from there rather than
    # re-running them; g11's own condnorm rows cover the other footings.
    par = pd.read_csv(os.path.join(ROOT, "results", "g4_covfair_full.csv"))
    par = (par[(par.backbone == args.backbone) & (par.arm == "condnorm")]
           .groupby(["dataset", "h"]).mse.mean())
    if ("condnorm", "global") not in cell.columns:
        cell[("condnorm", "global")] = np.nan
    cell[("condnorm", "global")] = cell[("condnorm", "global")].fillna(
        pd.Series({k: par.get(k, np.nan) for k in cell.index}))

    def col(arm, foot):
        return cell[(arm, foot)] if (arm, foot) in cell.columns \
            else pd.Series(np.nan, index=cell.index)

    SHOW = [("global", "global"), ("window_floor", "window"), ("scale", "scale")]
    tog = {f: (col("revin", f) - col("raw", f)) for f, _ in SHOW}

    lines, md = [], []
    md.append("| cell | " + " | ".join(
        f"RAW/{lab} | RevIN/{lab}" for _, lab in SHOW) + " | CN |")
    md.append("|" + "---|" * (2 * len(SHOW) + 2))
    for ds, h in idx:
        k = (ds, h)
        cells = []
        for f, _ in SHOW:
            cells += [num(col("raw", f)[k]), num(col("revin", f)[k])]
        lines.append(f"{PRETTY[ds]} $h{{=}}{h}$ & " + " & ".join(cells)
                     + " & " + num(col("condnorm", "global")[k]) + r" \\")
        md.append(f"| {PRETTY[ds]} h={h} | " + " | ".join(cells)
                  + " | " + num(col("condnorm", "global")[k]) + " |")

    stats = {f: boot(tog[f].values) for f, _ in SHOW}
    lines.append(r"\midrule")
    mean_cells = []
    for f, _ in SHOW:
        mean_cells += [num(col("raw", f).mean()), num(col("revin", f).mean())]
    lines.append("mean & " + " & ".join(mean_cells) + " & "
                 + num(col("condnorm", "global").mean()) + r" \\")
    lines.append(r"\addlinespace[1pt]")
    lines.append("RevIN $-$ \\rawm{} & \\multicolumn{2}{c}{"
                 + f"${stats['global'][0]:+.3f}$" + "} & \\multicolumn{2}{c}{"
                 + f"${stats['window_floor'][0]:+.3f}$"
                 + "} & \\multicolumn{2}{c}{"
                 + f"${stats['scale'][0]:+.3f}$" + "} & \\\\")
    md.append("| **mean** | " + " | ".join(mean_cells) + " | "
              + num(col("condnorm", "global").mean()) + " |")
    md.append("| **RevIN-RAW** | " + " | ".join(
        f"{stats[f][0]:+.3f} [{stats[f][1]:+.3f},{stats[f][2]:+.3f}]"
        + (" | " if i < len(SHOW) - 1 else "")
        for i, (f, _) in enumerate(SHOW)) + " | |")

    # ---- endpoints
    revc = [c for c in cell.columns if c[0] == "revin"]
    best_rev = cell[revc].min(axis=1)
    raw_g = col("raw", "global")
    prim = (best_rev - raw_g).dropna()
    pm, plo, phi = boot(prim.values)
    cnc = [c for c in cell.columns if c[0] == "condnorm"]
    best_cn = cell[cnc].min(axis=1) if cnc else pd.Series(np.nan, cell.index)
    endo = pd.concat([best_rev, raw_g], axis=1).min(axis=1)
    sec = (endo - best_cn).dropna()

    # The whole tabular is emitted here, not just its body: \input-ing a
    # fragment into an open tabular leaves a trailing space token that makes
    # the following \bottomrule a misplaced \noalign.
    head = [r"\begin{tabular}{lcccccc@{\hskip 10pt}c}", r"\toprule",
            r"& \multicolumn{2}{c}{global $z$} & "
            r"\multicolumn{2}{c}{window$^\dagger$} & "
            r"\multicolumn{2}{c}{scale-matched} & \\",
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
            r"Cell & \rawm{} & RevIN & \rawm{} & RevIN & \rawm{} & RevIN "
            r"& \cnm{} \\",
            r"\midrule"]
    with open(TEX, "w") as f:
        f.write("% generated by experiments/g11_table.py -- do not edit\n")
        f.write("\n".join(head + lines[:-1] + [r"\bottomrule",
                                               r"\end{tabular}"]) + "\n")

    with open(MD, "w") as f:
        f.write(f"# Footing table ({args.backbone}, {len(idx)} cells)\n\n")
        f.write("\n".join(md) + "\n\n")
        f.write("## endpoints (pre-registered in evidence/prereg_ramp_footing.md)\n\n")
        f.write(f"- PRIMARY (amended): min_f RevIN_f - RAW_global = {pm:+.4f} "
                f"[{plo:+.4f},{phi:+.4f}], positive in "
                f"{int((prim > 0).sum())}/{len(prim)} -> "
                f"{'CONFIRMED' if plo > 0 else 'NOT CONFIRMED'}\n")
        orig = (col("revin", "scale") - col("raw", "scale")).dropna()
        if len(orig):
            om, olo, ohi = boot(orig.values)
            f.write(f"- ORIGINAL endpoint: RevIN_scale - RAW_scale = {om:+.4f} "
                    f"[{olo:+.4f},{ohi:+.4f}], positive in "
                    f"{int((orig > 0).sum())}/{len(orig)}\n")
        if len(sec):
            f.write(f"- SECONDARY 1: best endogenous (any footing) - best CN = "
                    f"{sec.mean():+.4f}, CN better in "
                    f"{int((sec > 0).sum())}/{len(sec)} cells\n")
        f.write("\n## SECONDARY 2 -- destroying the covariate level "
                "(window_floor - global), per cell\n\n")
        for arm in ("raw", "revin"):
            v = col(arm, "window_floor") - col(arm, "global")
            f.write(f"- {arm}: " + ", ".join(
                f"{PRETTY[ds]} h={h} {v[(ds, h)]:+.3f}" for ds, h in idx
                if np.isfinite(v[(ds, h)])) + "\n")
        f.write("\n## unfloored window footing (degenerate on solar: "
                "all-night windows have ~no covariate variance)\n\n")
        for arm in ("raw", "revin"):
            v = col(arm, "window")
            f.write(f"- {arm}: " + ", ".join(
                f"{PRETTY[ds]} h={h} {v[(ds, h)]:.3f}" for ds, h in idx
                if np.isfinite(v[(ds, h)])) + "\n")

    print(open(MD).read())


if __name__ == "__main__":
    main()
