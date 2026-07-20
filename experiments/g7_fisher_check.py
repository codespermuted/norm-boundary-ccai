"""G7 item 7 — dependence-robust re-analysis of the Tab 1 Fisher stars.

Recomputes, per (dataset, non-best arm), the per-(backbone, h) DM p-values
exactly as experiments/g4_table1.py does (Harvey correction, seed-mean losses,
original four backbones), then compares three combinations over the K cells:

- Fisher chi^2 (the frozen Tab 1 rule; assumes independent cells)
- Simes (valid under positive regression dependence)
- harmonic-mean p (Wilson 2019 PNAS; valid under arbitrary dependence)

plus the assumption-free per-cell count k_sig = #{cells with p < 0.05 and
mean_diff > 0}. Reads only frozen inputs (results/g4_grid.csv,
results/g4_errors/); writes results/g7_fisher_robustness.csv.

Usage: uv run python -m experiments.g7_fisher_check
"""

import os

import numpy as np
import pandas as pd
from scipy import stats

from src.eval.dm_test import dm_test

ROOT = os.path.join(os.path.dirname(__file__), "..")
CSV_PATH = os.path.join(ROOT, "results", "g4_grid.csv")
ERR_DIR = os.path.join(ROOT, "results", "g4_errors")
OUT_PATH = os.path.join(ROOT, "results", "g7_fisher_robustness.csv")


def load_losses(dataset, norm, backbone, h, seeds):
    arrs = []
    for s in seeds:
        p = os.path.join(ERR_DIR, f"{dataset}_{norm}_{backbone}_{h}_{s}.npy")
        if os.path.exists(p):
            arrs.append(np.load(p))
    if not arrs:
        return None
    n = min(map(len, arrs))
    return np.mean([a[:n] for a in arrs], axis=0)


def main():
    df = pd.read_csv(CSV_PATH)
    # same filtering as g4_table1.py: pre-registered four backbones, de-duped
    df = df[df["backbone"].isin(("rlinear", "patchtst", "segrnn", "lgbm_dms"))]
    df = df.drop_duplicates(["dataset", "norm", "backbone", "h", "seed"])
    seeds = sorted(df["seed"].unique())

    rows = []
    for d in sorted(df["dataset"].unique()):
        sub = df[df["dataset"] == d]
        means = sub.groupby("norm")["mse"].mean()
        best = means.idxmin()
        for norm in means.index:
            if norm == best:
                continue
            pvals = []
            for (bb, h), _ in sub.groupby(["backbone", "h"]):
                l_best = load_losses(d, best, bb, h, seeds)
                l_arm = load_losses(d, norm, bb, h, seeds)
                if l_best is None or l_arm is None:
                    continue
                n = min(len(l_best), len(l_arm))
                r = dm_test(l_arm[:n], l_best[:n], h=int(h))
                pvals.append((r["p"], r["mean_diff"]))
            if not pvals:
                continue
            ps = np.array([p for p, _ in pvals])
            K = len(ps)
            chi = -2 * np.sum(np.log(np.maximum(ps, 1e-300)))
            p_fisher = float(stats.chi2.sf(chi, df=2 * K))
            p_simes = float(np.min(np.sort(ps) * K / np.arange(1, K + 1)))
            p_hmp = float(K / np.sum(1.0 / ps))
            worse = np.mean([md for _, md in pvals]) > 0
            rows.append({
                "dataset": d, "arm": norm, "best": best, "K": K,
                "k_sig": int(sum(1 for p, md in pvals
                                 if p < 0.05 and md > 0)),
                "p_fisher": p_fisher, "p_simes": p_simes, "p_hmp": p_hmp,
                "star_fisher": bool(p_fisher < 0.05 and worse),
                "star_simes": bool(p_simes < 0.05 and worse),
                "star_hmp": bool(p_hmp < 0.05 and worse),
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    n_f, n_s, n_h = (int(out[c].sum()) for c in
                     ("star_fisher", "star_simes", "star_hmp"))
    print(f"stars: Fisher {n_f} / Simes {n_s} / HMP {n_h}  -> {OUT_PATH}")
    flip = out[out["star_fisher"] & ~out["star_simes"]]
    if len(flip):
        print("flips under Simes:")
        print(flip.to_string(index=False))


if __name__ == "__main__":
    main()
