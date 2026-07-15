"""Tab 1 draft + DM significance + MCS + GATE 2 verdict (plan §5.4, §8).

- Tab 1: dataset x norm mean test MSE (backbone-, h-, seed-averaged), best
  per dataset bold, '*' when DM says the arm is significantly worse than the
  dataset's best arm (p < 0.05, per (h, backbone) losses averaged over seeds,
  Harvey-corrected, then Fisher-combined across cells)
- MCS (alpha=.10) per (dataset, h) over norm arms on backbone-averaged
  per-origin losses
- GATE 2: sign(mean RevIN - CondNorm gap) per dataset vs paper/predictions.md
  (hit rate >= 70% -> GO)

Usage: uv run python -m experiments.g4_table1 [--partial]
"""

import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

from src.eval.dm_test import dm_test
from src.eval.mcs import mcs

ROOT = os.path.join(os.path.dirname(__file__), "..")
CSV_PATH = os.path.join(ROOT, "results", "g4_grid.csv")
ERR_DIR = os.path.join(ROOT, "results", "g4_errors")
TAB_PATH = os.path.join(ROOT, "paper", "tables", "tab1_draft.md")
GATE_PATH = os.path.join(ROOT, "results", "gate2.md")

PREDICTIONS = {  # paper/predictions.md (commit cab17c1)
    "jeju_wind": +1, "gefcom_wind": +1, "gefcom_load": +1, "gefcom_solar": +1,
    "etth1": -1, "etth2": -1, "electricity": -1, "weather": -1,
}
NORMS = ("raw", "revin", "san", "fan", "condnorm")


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", action="store_true",
                    help="analyze whatever is complete so far")
    args = ap.parse_args()

    df = pd.read_csv(CSV_PATH)
    datasets = [d for d in PREDICTIONS if d in set(df["dataset"])]

    # ---------------- GATE 2: pre-registered sign hits -----------------
    gate_rows, hits = [], []
    for d in datasets:
        sub = df[df["dataset"] == d]
        piv = sub.pivot_table(index=["backbone", "h", "seed"], columns="norm",
                              values="mse")
        if not {"revin", "condnorm"} <= set(piv.columns):
            continue
        paired = piv.dropna(subset=["revin", "condnorm"])
        gap = float((paired["revin"] - paired["condnorm"]).mean())
        pred = PREDICTIONS[d]
        hit = int(np.sign(gap) == pred)
        hits.append(hit)
        gate_rows.append({"dataset": d, "gap_revin_minus_cn": round(gap, 4),
                          "n_pairs": len(paired), "predicted": pred,
                          "hit": "✅" if hit else "❌"})
    hit_rate = np.mean(hits) if hits else float("nan")
    verdict = "GO ✅" if (len(hits) == len(PREDICTIONS) and hit_rate >= 0.7) \
        else ("PARTIAL" if len(hits) < len(PREDICTIONS)
              else "NO-GO ❌")

    # ---------------- Tab 1: dataset x norm + DM marks -----------------
    seeds = sorted(df["seed"].unique())
    tab, marks = {}, {}
    for d in datasets:
        sub = df[df["dataset"] == d]
        means = sub.groupby("norm")["mse"].mean()
        tab[d] = means
        best = means.idxmin()
        marks[d] = {}
        for norm in means.index:
            if norm == best:
                marks[d][norm] = "best"
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
            if pvals:
                chi = -2 * sum(np.log(max(p, 1e-300)) for p, _ in pvals)
                p_comb = float(stats.chi2.sf(chi, df=2 * len(pvals)))
                worse = np.mean([md for _, md in pvals]) > 0
                marks[d][norm] = "*" if (p_comb < 0.05 and worse) else ""
            else:
                marks[d][norm] = ""

    lines = ["# Tab 1 초안 — dataset × normalization (backbone·h·seed 평균, "
             "std-scale MSE)", "",
             "| dataset | " + " | ".join(NORMS) + " |",
             "|---|" + "---|" * len(NORMS)]
    for d in datasets:
        cells = []
        for nm in NORMS:
            if nm not in tab[d]:
                cells.append("—")
                continue
            v = f"{tab[d][nm]:.4f}"
            if marks[d].get(nm) == "best":
                v = f"**{v}**"
            elif marks[d].get(nm) == "*":
                v += "*"
            cells.append(v)
        lines.append(f"| {d} | " + " | ".join(cells) + " |")
    lines += ["", "`*` = DM 검정(Harvey 보정, (backbone,h) 셀별 시드 평균 손실, "
              "Fisher 결합)에서 해당 데이터셋 최적 arm보다 유의하게 나쁨 (p<0.05). "
              "**굵게** = 최적 arm."]

    # ---------------- MCS per (dataset, h) over norms -------------------
    lines += ["", "## MCS (α=0.10) — (dataset, h)별 정규화 arm 잔존 집합", ""]
    for d in datasets:
        sub = df[df["dataset"] == d]
        for h in sorted(sub["h"].unique()):
            loss_mat, names = [], []
            for nm in NORMS:
                per_bb = [load_losses(d, nm, bb, h, seeds)
                          for bb in sub["backbone"].unique()]
                per_bb = [x for x in per_bb if x is not None]
                if per_bb:
                    n = min(map(len, per_bb))
                    loss_mat.append(np.mean([x[:n] for x in per_bb], axis=0))
                    names.append(nm)
            if len(names) < 2:
                continue
            n = min(map(len, loss_mat))
            res = mcs(np.stack([x[:n] for x in loss_mat], axis=1), names)
            lines.append(f"- {d} h={h}: {{{', '.join(res['included'])}}}")

    os.makedirs(os.path.dirname(TAB_PATH), exist_ok=True)
    with open(TAB_PATH, "w") as f:
        f.write("\n".join(lines))

    gate_md = ["# GATE 2 판정 — 사전 등록 적중률 (predictions.md, commit cab17c1)",
               "", f"**판정: {verdict}** — 적중 {sum(hits)}/{len(hits)} "
               f"(판정 대상 {len(PREDICTIONS)}개 중 {len(hits)}개 산출됨)", "",
               "| dataset | 격차(RevIN−CN) | n쌍 | 예측 | 적중 |",
               "|---|---|---|---|---|"]
    for r in gate_rows:
        gate_md.append(f"| {r['dataset']} | {r['gap_revin_minus_cn']:+.4f} | "
                       f"{r['n_pairs']} | {'+' if r['predicted'] > 0 else '−'} |"
                       f" {r['hit']} |")
    with open(GATE_PATH, "w") as f:
        f.write("\n".join(gate_md))

    print("\n".join(gate_md))
    print(f"\nTab1 -> {TAB_PATH}")


if __name__ == "__main__":
    main()
