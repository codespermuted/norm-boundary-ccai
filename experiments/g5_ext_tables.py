"""G5 — extension-block final tables (Blocks B and C), reported separately
from the pre-registered analysis.

  Tab B  covfair-full: arm x backbone mean MSE (exogenous datasets),
         DM(revin+cov vs condnorm+cov) per backbone
  Tab C  SOTA: (i) MS mode (covariate path) arm means + DM;
         (ii) endogenous iTransformer/TimeXer rows alongside Block-A
              backbones for the same datasets (robustness view)

Usage: uv run python -m experiments.g5_ext_tables
"""

import os

import numpy as np
import pandas as pd

from src.eval.dm_test import dm_test

ROOT = os.path.join(os.path.dirname(__file__), "..")
ERR = os.path.join(ROOT, "results", "g4_errors")
TABD = os.path.join(ROOT, "paper", "tables")
ARMS = ("raw", "revin", "san", "fan", "condnorm")
EXOG = ("jeju_wind", "gefcom_wind", "gefcom_load", "gefcom_solar")


def dm_pair(tag_a, tag_b, h):
    la, lb = [], []
    for s in range(5):
        pa = os.path.join(ERR, f"{tag_a}_{s}.npy")
        pb = os.path.join(ERR, f"{tag_b}_{s}.npy")
        if os.path.exists(pa) and os.path.exists(pb):
            la.append(np.load(pa))
            lb.append(np.load(pb))
    if not la:
        return None
    n = min(min(map(len, la)), min(map(len, lb)))
    return dm_test(np.mean([x[:n] for x in la], 0),
                   np.mean([x[:n] for x in lb], 0), h=h)


def tab_covfair() -> str:
    cf = pd.read_csv(os.path.join(ROOT, "results", "g4_covfair_full.csv"))
    cf = cf.drop_duplicates(["dataset", "arm", "backbone", "h", "seed"])
    lines = ["# Tab B 초안 — 정보-대등 ablation (전 arm 동일 공변량, 외생 4종)",
             "", "arm 평균 MSE (dataset·h·seed 평균):", ""]
    piv = (cf.groupby(["backbone", "arm"])["mse"].mean().unstack("arm")
           .reindex(columns=ARMS).round(4))
    lines += [piv.to_markdown(), ""]
    lines += ["DM (revin+cov vs condnorm+cov, (dataset,h) 셀별 시드평균 손실):",
              "", "| backbone | CN우세 유의 셀 | 전체 셀 |", "|---|---|---|"]
    from experiments.g4_grid import DATASETS

    for bb in piv.index:
        sig = tot = 0
        for d in EXOG:
            for h in DATASETS[d]["horizons"]:
                r = dm_pair(f"{d}_revin+cov_{bb}_{h}",
                            f"{d}_condnorm+cov_{bb}_{h}", h)
                if r is None:
                    continue
                tot += 1
                sig += int(r["p"] < 0.05 and r["stat"] > 0)
        lines.append(f"| {bb} | {sig} | {tot} |")
    lines += ["", "주: 블록 간 직접 비교 금지 원칙에 따라 이 표는 covfair 내부 "
              "비교만 담는다 (docs/design_audit.md §4)."]
    text = "\n".join(lines)
    open(os.path.join(TABD, "tabB_covfair.md"), "w").write(text)
    return text


def tab_sota() -> str:
    df = pd.read_csv(os.path.join(ROOT, "results", "g4_grid.csv"))
    df = df.drop_duplicates(["dataset", "norm", "backbone", "h", "seed"])
    lines = ["# Tab C 초안 — SOTA 확장 (사전 등록 외 robustness)", ""]

    for bb, title in (("timexer_ms", "TimeXer-MS (공식 exogenous 경로)"),
                      ("itransformer_ms", "iTransformer-MS (공변량 variate 토큰)")):
        sub = df[df.backbone == bb]
        piv = (sub.groupby(["dataset", "norm"])["mse"].mean().unstack("norm")
               .reindex(columns=ARMS).round(4))
        lines += [f"## {title}", "", piv.to_markdown(), ""]
        piv2 = sub.pivot_table(index=["dataset", "h", "seed"], columns="norm",
                               values="mse").dropna(subset=["revin", "condnorm"])
        gap = float((piv2["revin"] - piv2["condnorm"]).mean())
        lines += [f"평균 격차 RevIN−CondNorm = {gap:+.4f} "
                  f"({len(piv2)}쌍) — {'CN 우세' if gap > 0 else 'IN 우세'}", ""]

    lines += ["## Endogenous 확장 (iTransformer / TimeXer-M): "
              "정규화 arm 평균 MSE", ""]
    endo = df[df.backbone.isin(["itransformer", "timexer"])]
    piv = (endo.groupby(["backbone", "dataset", "norm"])["mse"].mean()
           .unstack("norm").reindex(columns=ARMS).round(4))
    lines += [piv.to_markdown(), "",
              "주: GATE 2는 사전 등록된 원 4백본으로만 판정; 본 표는 등록 후 "
              "확장으로 별도 보고 (docs/design_audit.md 블록 C/D)."]
    text = "\n".join(lines)
    open(os.path.join(TABD, "tabC_sota.md"), "w").write(text)
    return text


if __name__ == "__main__":
    print(tab_covfair()[:900])
    print()
    print(tab_sota()[:900])
