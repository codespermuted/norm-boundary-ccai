"""G2 analysis: Fig 2 (theory curves + empirical points) and GATE 1 verdict.

Theory curves are the closed-form structure of model M1 with parameters
MEASURED from the DGP (docs/theory_g1.md §5 mapping):
  theory_mse_arm(lam, L, h) = th_level_term(arm)  [logged per run, oracle truth]
                              + sigma_eps_hat(lam, L, h)  [common shape residual,
                                calibrated once from the CN-oracle arm]
The common offset cancels in every IN-vs-CN comparison, so the GATE 1
crossover test is calibration-free.

GATE 1 (plan §4.3):
  1. empirical IN/CN-est crossover lambda within +-0.1 of theory crossover
  2. IN - CN gap widens with h
  3. CN-est minus CN-oracle explained by the measured first-stage error term

Usage: uv run python -m src.synth.analyze
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.theory.figstyle import METHOD_COLORS, apply_paper_style

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
CSV_PATH = os.path.join(ROOT, "results", "synth_grid.csv")
FIG_DIR = os.path.join(ROOT, "paper", "figures")
GATE_PATH = os.path.join(ROOT, "results", "gate1.md")

ARM_COLOR = {"raw": METHOD_COLORS["raw"], "revin": METHOD_COLORS["in"],
             "cn_est": METHOD_COLORS["cn"], "cn_oracle": METHOD_COLORS["cn_oracle"]}
ARM_LABEL = {"raw": "RAW", "revin": "IN (RevIN)", "cn_est": "CN-est",
             "cn_oracle": "CN-oracle"}


def crossover_lam(lams: np.ndarray, diff: np.ndarray) -> float:
    """First sign change of diff (IN - CN) along lams, linear interpolation."""
    s = np.sign(diff)
    for i in range(len(lams) - 1):
        if s[i] >= 0 > s[i + 1] or s[i] > 0 >= s[i + 1]:
            x0, x1, d0, d1 = lams[i], lams[i + 1], diff[i], diff[i + 1]
            return float(x0 + (x1 - x0) * d0 / (d0 - d1))
        if s[i] <= 0 < s[i + 1] or s[i] < 0 <= s[i + 1]:
            x0, x1, d0, d1 = lams[i], lams[i + 1], diff[i], diff[i + 1]
            return float(x0 + (x1 - x0) * d0 / (d0 - d1))
    return float("nan")


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df["lam"] = df["lam"].round(1)
    return df


def make_fig2(df: pd.DataFrame) -> str:
    apply_paper_style()
    lbs = sorted(df["L"].unique())
    hs = sorted(df["h"].unique())
    fig, axes = plt.subplots(len(lbs), len(hs), figsize=(7.0, 4.6),
                             sharex=True, sharey="row", constrained_layout=True)
    axes = np.atleast_2d(axes)

    for i, L in enumerate(lbs):
        for j, h in enumerate(hs):
            ax = axes[i, j]
            sub = df[(df["L"] == L) & (df["h"] == h)]
            # common residual calibrated from CN-oracle arm (per lam)
            eps = sub[sub["norm"] == "cn_oracle"].groupby("lam")["mse"].mean()
            for arm in ("raw", "revin", "cn_est", "cn_oracle"):
                g = sub[sub["norm"] == arm].groupby("lam")
                mean, sd, n = g["mse"].mean(), g["mse"].std(), g["mse"].count()
                ci = 1.96 * sd / np.sqrt(n)
                ax.errorbar(mean.index, mean, yerr=ci, fmt="o", ms=3,
                            lw=0, elinewidth=0.8, capsize=1.5,
                            color=ARM_COLOR[arm],
                            label=ARM_LABEL[arm] if (i, j) == (0, 0) else None)
                th = (sub[sub["norm"] == arm].groupby("lam")["th_level_term"]
                      .mean() + eps)
                ax.plot(th.index, th.values, "-", lw=1.4, color=ARM_COLOR[arm],
                        alpha=0.85)
            ax.set_title(f"$L={L}$, $h={h}$")
            if i == len(lbs) - 1:
                ax.set_xlabel(r"$\lambda$")
            if j == 0:
                ax.set_ylabel("test MSE (global-z scale)")
    fig.legend(loc="outside upper center", ncol=4, frameon=False)
    os.makedirs(FIG_DIR, exist_ok=True)
    base = os.path.join(FIG_DIR, "fig2_synth")
    fig.savefig(base + ".pdf")
    fig.savefig(base + ".png")
    plt.close(fig)
    return base


def gate1(df: pd.DataFrame) -> str:
    lines = ["# GATE 1 판정 (G2, plan §4.3)", ""]
    ok1_all, ok2_all, ok3_all = [], [], []

    for L in sorted(df["L"].unique()):
        lines.append(f"## L = {L}\n")
        lines.append("| h | λ*_emp | λ*_theory | \\|diff\\| ≤ 0.1 | gap(λ=0.8) |")
        lines.append("|---|---|---|---|---|")
        gaps = []
        for h in sorted(df["h"].unique()):
            sub = df[(df["L"] == L) & (df["h"] == h)]
            m = sub.groupby(["norm", "lam"])["mse"].mean().unstack("lam")
            t = sub.groupby(["norm", "lam"])["th_level_term"].mean().unstack("lam")
            lams = m.columns.values.astype(float)
            emp = crossover_lam(lams, (m.loc["revin"] - m.loc["cn_est"]).values)
            th = crossover_lam(lams, (t.loc["revin"] - t.loc["cn_est"]).values)
            ok = (abs(emp - th) <= 0.1) if np.isfinite(emp) and np.isfinite(th) else False
            ok1_all.append(ok)
            gap = float(m.loc["revin", 0.8] - m.loc["cn_est", 0.8])
            gaps.append(gap)
            lines.append(f"| {h} | {emp:.3f} | {th:.3f} | "
                         f"{'✅' if ok else '❌'} | {gap:.4f} |")
        ok2 = all(a < b for a, b in zip(gaps, gaps[1:]))
        ok2_all.append(ok2)
        lines.append(f"\n- gap(λ=0.8) h-단조 증가: {'✅' if ok2 else '❌'} "
                     f"({['%.4f' % g for g in gaps]})\n")

    # criterion 3: CN-est degradation vs CN-oracle explained by first-stage term
    sub = df[df["norm"].isin(["cn_est", "cn_oracle"])]
    m = sub.groupby(["norm", "lam", "L", "h"])["mse"].mean().unstack("norm")
    degr = (m["cn_est"] - m["cn_oracle"]).clip(lower=0)
    fs = (df[df["norm"] == "cn_est"]
          .groupby(["lam", "L", "h"])["th_level_term"].mean())
    ratio = (degr / fs.replace(0, np.nan)).dropna()
    med = float(ratio.median())
    ok3 = 0.5 <= med <= 2.0
    ok3_all.append(ok3)
    lines.append("## 기준 3 — CN-est 저하의 1단계 오차 설명력\n")
    lines.append(f"- (CN-est − CN-oracle) / 측정된 1단계 수준 오차항: "
                 f"중앙값 {med:.2f} (0.5–2.0 내: {'✅' if ok3 else '❌'})\n")

    verdict = all(ok1_all) and all(ok2_all) and all(ok3_all)
    lines.insert(1, f"**판정: {'GO ✅' if verdict else 'NO-GO ❌'}** — "
                 f"기준1 교차±0.1: {sum(ok1_all)}/{len(ok1_all)}, "
                 f"기준2 h-단조: {sum(ok2_all)}/{len(ok2_all)}, "
                 f"기준3 1단계 설명: {sum(ok3_all)}/{len(ok3_all)}\n")
    text = "\n".join(lines)
    with open(GATE_PATH, "w") as f:
        f.write(text)
    return text


def main():
    df = load()
    n_expected = 11 * 10 * 2 * 3 * 4
    print(f"rows: {len(df)} / expected {n_expected}")
    base = make_fig2(df)
    print(f"fig2 saved: {base}.png")
    print(gate1(df))


if __name__ == "__main__":
    main()
