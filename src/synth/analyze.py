"""G2 analysis: Fig 2 (theory curves + empirical points) and GATE 1 verdict.

Theory curves are the CLOSED-FORM optimal linear predictor of each
normalization class (src/theory/linear_class.py, Proposition 2' / Toner &
Darlow machinery) — exact solutions of the constrained regressions that SGD
training approximates, computed from the same cached series. No calibration.

The stylized M1 crossover (restoration-only) is reported alongside as a
diagnostic upper bound; the gap between the two quantifies the linear
backbone's implicit level tracking (documented deviation from assumption A4,
docs/theory_g1.md).

GATE 1 (plan §4.3):
  1. empirical IN/CN-est crossover lambda within +-0.1 of the closed-form
     linear-class crossover
  2. IN - CN gap widens with h
  3. CN-est minus CN-oracle degradation matches the closed-form prediction

Usage: uv run python -m src.synth.analyze
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.synth.runner import load_or_build_series
from src.theory.figstyle import METHOD_COLORS, apply_paper_style
from src.theory.linear_class import ols_theory_row

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


OLS_CSV = os.path.join(ROOT, "results", "synth_ols.csv")


def build_ols_table(df: pd.DataFrame) -> pd.DataFrame:
    """Closed-form linear-class MSEs for every (lam, seed, L, h) in df; cached."""
    done = pd.read_csv(OLS_CSV) if os.path.exists(OLS_CSV) else pd.DataFrame()
    keys = df[["lam", "seed", "L", "h"]].drop_duplicates()
    rows = list(done.to_dict("records"))
    have = {(r["lam"], r["seed"], r["L"], r["h"]) for r in rows}
    todo = [k for k in keys.itertuples(index=False)
            if (k.lam, k.seed, k.L, k.h) not in have]
    for i, k in enumerate(todo):
        series = load_or_build_series(float(k.lam), int(k.seed))
        th = ols_theory_row(series, int(k.L), int(k.h))
        rows.append({"lam": k.lam, "seed": k.seed, "L": k.L, "h": k.h,
                     **{f"th_{a}": v for a, v in th.items()}})
        if (i + 1) % 60 == 0:
            pd.DataFrame(rows).to_csv(OLS_CSV, index=False)
            print(f"ols theory {i + 1}/{len(todo)}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(OLS_CSV, index=False)
    return out


def make_fig2(df: pd.DataFrame, ols: pd.DataFrame) -> str:
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
            osub = ols[(ols["L"] == L) & (ols["h"] == h)]
            for arm in ("raw", "revin", "cn_est", "cn_oracle"):
                g = sub[sub["norm"] == arm].groupby("lam")
                mean, sd, n = g["mse"].mean(), g["mse"].std(), g["mse"].count()
                ci = 1.96 * sd / np.sqrt(n)
                ax.errorbar(mean.index, mean, yerr=ci, fmt="o", ms=3,
                            lw=0, elinewidth=0.8, capsize=1.5,
                            color=ARM_COLOR[arm],
                            label=ARM_LABEL[arm] if (i, j) == (0, 0) else None)
                th = osub.groupby("lam")[f"th_{arm}"].mean()
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


def gate1(df: pd.DataFrame, ols: pd.DataFrame) -> str:
    lines = ["# GATE 1 판정 (G2, plan §4.3)", ""]
    ok1_all, ok2_all, ok3_all = [], [], []

    for L in sorted(df["L"].unique()):
        lines.append(f"## L = {L}\n")
        lines.append("| h | λ*_emp | λ*_OLS이론 | \\|diff\\| ≤ 0.1 | λ*_M1(상한) | gap(고λ) |")
        lines.append("|---|---|---|---|---|---|")
        gaps = []
        for h in sorted(df["h"].unique()):
            sub = df[(df["L"] == L) & (df["h"] == h)]
            osub = ols[(ols["L"] == L) & (ols["h"] == h)]
            m = sub.groupby(["norm", "lam"])["mse"].mean().unstack("lam")
            t_m1 = sub.groupby(["norm", "lam"])["th_level_term"].mean().unstack("lam")
            to = osub.groupby("lam")[["th_revin", "th_cn_est"]].mean()
            lams = m.columns.values.astype(float)
            emp = crossover_lam(lams, (m.loc["revin"] - m.loc["cn_est"]).values)
            th = crossover_lam(to.index.values.astype(float),
                               (to["th_revin"] - to["th_cn_est"]).values)
            m1 = crossover_lam(lams, (t_m1.loc["revin"] - t_m1.loc["cn_est"]).values)
            ok = (abs(emp - th) <= 0.1) if np.isfinite(emp) and np.isfinite(th) else False
            ok1_all.append(ok)
            lam_gap = 0.8 if 0.8 in m.columns else float(m.columns.max())
            per_seed = (sub[sub["lam"] == lam_gap]
                        .pivot_table(index="seed", columns="norm", values="mse"))
            gaps.append((per_seed["revin"] - per_seed["cn_est"]).values)
            gap = float(np.mean(gaps[-1]))
            lines.append(f"| {h} | {emp:.3f} | {th:.3f} | "
                         f"{'✅' if ok else '❌'} | {m1:.3f} | {gap:.4f} |")
        # criterion 2: gap non-decreasing in h within seed-paired 95% CI
        ok2, diffs = True, []
        for g0, g1 in zip(gaps, gaps[1:]):
            n = min(len(g0), len(g1))
            d = g1[:n] - g0[:n]
            se = d.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
            diffs.append(f"{d.mean():+.4f}±{1.96 * se:.4f}")
            if d.mean() < -1.96 * se:
                ok2 = False
        ok2_all.append(ok2)
        lines.append(f"\n- gap(λ={0.8}) h-확대 (시드쌍 95% CI 내 비감소): "
                     f"{'✅' if ok2 else '❌'} (Δ: {diffs})\n")

    # criterion 3: CN-est degradation vs CN-oracle, empirical vs closed form
    emp_d = (df[df["norm"].isin(["cn_est", "cn_oracle"])]
             .groupby(["norm", "lam", "L", "h"])["mse"].mean().unstack("norm"))
    emp_degr = (emp_d["cn_est"] - emp_d["cn_oracle"]).clip(lower=0)
    ols_d = ols.groupby(["lam", "L", "h"])[["th_cn_est", "th_cn_oracle"]].mean()
    ols_degr = (ols_d["th_cn_est"] - ols_d["th_cn_oracle"]).clip(lower=1e-12)
    ratio = (emp_degr / ols_degr).replace([np.inf, -np.inf], np.nan).dropna()
    med = float(ratio.median())
    ok3 = 0.5 <= med <= 2.0
    ok3_all.append(ok3)
    lines.append("## 기준 3 — CN-est 저하 (vs CN-oracle)의 이론 설명력\n")
    lines.append(f"- 실험 저하 / OLS이론 저하 비율: 중앙값 {med:.2f} "
                 f"(0.5–2.0 내: {'✅' if ok3 else '❌'})\n")

    verdict = all(ok1_all) and all(ok2_all) and all(ok3_all)
    lines.insert(1, f"**판정: {'GO ✅' if verdict else 'NO-GO ❌'}** — "
                 f"기준1 교차±0.1: {sum(ok1_all)}/{len(ok1_all)}, "
                 f"기준2 h-단조: {sum(ok2_all)}/{len(ok2_all)}, "
                 f"기준3 1단계 설명: {sum(ok3_all)}/{len(ok3_all)}\n")
    lines.append("\n주: λ*_OLS이론 = 각 정규화 클래스의 제약 OLS 닫힌형 최적해 교차 "
                 "(Prop 2', SGD 없이 계산). λ*_M1 = 복원 규칙만의 양식화 상한 — "
                 "둘의 차이가 선형 백본의 암묵적 수준 추적 효과 (A4 괴리, docs/theory_g1.md).")
    text = "\n".join(lines)
    with open(GATE_PATH, "w") as f:
        f.write(text)
    return text


def main():
    df = load()
    n_expected = 11 * 10 * 2 * 3 * 4
    print(f"rows: {len(df)} / expected {n_expected}")
    ols = build_ols_table(df)
    base = make_fig2(df, ols)
    print(f"fig2 saved: {base}.png")
    print(gate1(df, ols))


if __name__ == "__main__":
    main()
