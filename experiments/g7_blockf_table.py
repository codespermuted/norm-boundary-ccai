"""Tab F draft — Block F probabilistic results (g7_blockf.csv).

Renders paper/tables/tabF_probabilistic.md: per backbone, dataset x arm mean
pinball with cov80 in parentheses (h- and seed-averaged), split into the
exogenous group vs the standard group sub-blocks. crps = 2 x pinball at a
fixed quantile set, so the pinball ordering IS the crps ordering — crps is
therefore not tabulated separately.

Also seeds docs/blockf_summary.md with the {TO_FILL} draft paragraph IF the
file does not exist yet (never overwrites — the draft is hand-finished).

Usage: uv run python -m experiments.g7_blockf_table [--partial]
"""

import argparse
import os

import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
CSV_PATH = os.path.join(ROOT, "results", "g7_blockf.csv")
TAB_PATH = os.path.join(ROOT, "paper", "tables", "tabF_probabilistic.md")
SUMMARY_PATH = os.path.join(ROOT, "docs", "blockf_summary.md")

EXOG = ("jeju_wind", "gefcom_wind", "gefcom_load", "gefcom_solar")
STANDARD = ("etth1", "etth2", "weather", "electricity")
ARMS = {"rlinear_q": ("raw", "revin", "san", "fan", "condnorm"),
        "lgbm_q": ("raw", "winz", "condnorm")}

SUMMARY_DRAFT = """\
# Block F 요약 초안 — 확률적 지표에서도 경계가 유지되는가

> DRAFT. 숫자는 results/g7_blockf.csv 완주 후 {TO_FILL} 자리에 채울 것.
> 표: paper/tables/tabF_probabilistic.md. 설계 근거: docs/blockf_design.md.

## 논문용 문단 초안 (영어)

Block F extends the point-forecast boundary to probabilistic forecasting:
the same eight datasets, grid horizons, and frozen Block-A lookbacks, with a
quantile head (pinball loss, CRPS approximated as twice the mean pinball
over the quantile grid) replacing the MSE objective. On the exogenous group,
CondNorm improves mean pinball over RevIN by {TO_FILL} on rlinear_q
({TO_FILL} of {TO_FILL} (dataset, h) cells), and the 80% interval coverage
moves from {TO_FILL} (RevIN) to {TO_FILL} (CondNorm) against the 0.80
nominal level — i.e., the boundary {TO_FILL: persists / amplifies /
attenuates} when the target is the predictive distribution rather than the
conditional mean. On the standard group the ordering {TO_FILL: mirrors /
reverses} the Block-A pattern ({TO_FILL} vs {TO_FILL} mean pinball). The
LightGBM quantile backbone shows {TO_FILL: the same / a different} contrast
(raw {TO_FILL} / winz {TO_FILL} / CondNorm {TO_FILL}), indicating the effect
is {TO_FILL: not / partly} an artifact of the neural training protocol.
Coverage decomposition (cov_lo/cov_hi) attributes miscoverage mainly to
{TO_FILL: the lower / the upper} tail on {TO_FILL}.

## 판정 메모 (한국어)

- 경계 유지 여부: {TO_FILL} (외생 4종에서 CN<RevIN pinball 셀 수 {TO_FILL}/11)
- 증폭/감쇠: Block A MSE 격차 대비 pinball 상대 격차 {TO_FILL}
- 커버리지: RevIN cov80 {TO_FILL}, CN cov80 {TO_FILL} (명목 0.80)
- lgbm_q 교차 확인: {TO_FILL}
"""


def block(df: pd.DataFrame, backbone: str, group: tuple) -> list[str]:
    arms = ARMS[backbone]
    sub = df[(df["backbone"] == backbone) & df["dataset"].isin(group)]
    lines = ["| dataset | " + " | ".join(arms) + " |",
             "|---|" + "---|" * len(arms)]
    for d in group:
        dd = sub[sub["dataset"] == d]
        if dd.empty:
            continue
        m = dd.groupby("arm")[["pinball", "cov80"]].mean()
        best = m["pinball"].idxmin() if len(m) else None
        cells = []
        for a in arms:
            if a not in m.index:
                cells.append("—")
                continue
            v = f"{m.loc[a, 'pinball']:.4f} ({m.loc[a, 'cov80']:.2f})"
            cells.append(f"**{v}**" if a == best else v)
        lines.append(f"| {d} | " + " | ".join(cells) + " |")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", action="store_true",
                    help="render whatever is complete so far")
    args = ap.parse_args()

    df = pd.read_csv(CSV_PATH)
    df = df.drop_duplicates(["dataset", "arm", "backbone", "h", "seed"],
                            keep="last")
    n_cells = df.groupby("backbone").size().to_dict()

    lines = ["# Tab F 초안 — Block F 확률 예측 (dataset × arm, h·seed 평균)",
             "",
             "셀 = 평균 pinball (평균 cov80), **굵게** = 해당 데이터셋 최소 "
             "pinball arm. CRPS = 2×pinball (고정 분위수 격자 근사)이므로 "
             "순위는 pinball과 동일하여 별도 표기하지 않음. 전 지표 전역 "
             "z-score 공간."]
    if args.partial:
        lines += ["", f"> PARTIAL — 현재 행수: {n_cells}"]
    for backbone in ("rlinear_q", "lgbm_q"):
        if backbone not in set(df["backbone"]):
            continue
        lines += ["", f"## {backbone}"
                  + (" (9분위수 0.1–0.9, seeds 0–4)" if backbone == "rlinear_q"
                     else " (3분위수 {0.1,0.5,0.9}, 결정적 seed 0)"), ""]
        lines += ["### 외생 그룹 (exogenous)", ""]
        lines += block(df, backbone, EXOG)
        lines += ["", "### 표준 그룹 (standard)", ""]
        lines += block(df, backbone, STANDARD)
    lines += ["", "주: cov80 명목값 0.80. lgbm_q의 SAN/FAN은 Block A와 동일하게 "
              "구조적 N/A, winz = 창 z-정규화 (pinball 1차 동차성에 맞춘 sd "
              "표본가중 — docs/blockf_design.md §4). 블록 간 직접 비교 금지 "
              "원칙 유지 (docs/design_audit.md §1.3)."]

    os.makedirs(os.path.dirname(TAB_PATH), exist_ok=True)
    with open(TAB_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"TabF -> {TAB_PATH}")

    if not os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH, "w") as f:
            f.write(SUMMARY_DRAFT)
        print(f"summary draft seeded -> {SUMMARY_PATH}")
    else:
        print(f"summary exists, untouched -> {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
