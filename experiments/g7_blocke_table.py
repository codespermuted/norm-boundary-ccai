"""Tab E — Block E baselines & published-default ablation table.

Renders paper/tables/tabE_baselines.md from results/g7_blocke.csv:
per dataset x arm mean test MSE and MAE (averaged over horizons and seeds,
global-z space fitted on train — Block A metric convention), plus an nMAE
row for jeju_wind (raw-scale MAE / train-split max of y, a capacity proxy).
Block E is self-contained — the note in the table says so explicitly.

Usage: uv run python -m experiments.g7_blocke_table
"""

import csv
import os
from collections import defaultdict

from experiments.g7_blocke import ARM_ORDER, CSV_PATH, DATASETS

TAB_PATH = os.path.join(os.path.dirname(__file__), "..", "paper", "tables",
                        "tabE_baselines.md")


def main():
    acc = defaultdict(lambda: defaultdict(list))
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            cell = acc[(r["dataset"], r["arm"])]
            cell["mse"].append(float(r["mse"]))
            cell["mae"].append(float(r["mae"]))
            if r["nmae"]:
                cell["nmae"].append(float(r["nmae"]))

    def fmt(dataset, arm, metric):
        vals = acc.get((dataset, arm), {}).get(metric)
        return f"{sum(vals) / len(vals):.4f}" if vals else "—"

    lines = [
        "# Tab E 초안 — Block E baselines & published-default ablation "
        "(dataset × arm, h·seed 평균, std-scale)", "",
        "Block E는 **자체 완결(self-contained)** 블록이다: 모든 비교는 블록 내 "
        "동일 세팅(동일 분할·lookback·예산)의 arm 사이에서만 유효하며, Block "
        "A/B 표의 수치와 직접 비교하지 않는다 (revin_all의 비교군 "
        "linmix_raw/linmix_revin/linmix_condnorm이 블록 내에 포함된 이유). "
        "지표는 본 그리드와 동일한 train-적합 global z-score 공간의 MSE/MAE. "
        "nMAE(jeju_wind 한정)는 원 스케일 MAE를 train 구간 y 최대값(용량 "
        "프록시)으로 나눈 값.", "",
        "| dataset | metric | " + " | ".join(ARM_ORDER) + " |",
        "|---|---|" + "---|" * len(ARM_ORDER),
    ]
    for d in DATASETS:
        rows = [("MSE", "mse"), ("MAE", "mae")]
        if d == "jeju_wind":
            rows.append(("nMAE", "nmae"))
        for label, metric in rows:
            cells = [fmt(d, a, metric) for a in ARM_ORDER]
            lines.append(f"| {d} | {label} | " + " | ".join(cells) + " |")
    os.makedirs(os.path.dirname(TAB_PATH), exist_ok=True)
    with open(TAB_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"TabE -> {TAB_PATH}")


if __name__ == "__main__":
    main()
