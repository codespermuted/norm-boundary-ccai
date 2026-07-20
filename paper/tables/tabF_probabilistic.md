# Tab F 초안 — Block F 확률 예측 (dataset × arm, h·seed 평균)

셀 = 평균 pinball (평균 cov80), **굵게** = 해당 데이터셋 최소 pinball arm. CRPS = 2×pinball (고정 분위수 격자 근사)이므로 순위는 pinball과 동일하여 별도 표기하지 않음. 전 지표 전역 z-score 공간.

> PARTIAL — 현재 행수: {'rlinear_q': 5}

## rlinear_q (9분위수 0.1–0.9, seeds 0–4)

### 외생 그룹 (exogenous)

| dataset | raw | revin | san | fan | condnorm |
|---|---|---|---|---|---|
| jeju_wind | **0.2046 (0.82)** | — | — | — | — |

### 표준 그룹 (standard)

| dataset | raw | revin | san | fan | condnorm |
|---|---|---|---|---|---|

주: cov80 명목값 0.80. lgbm_q의 SAN/FAN은 Block A와 동일하게 구조적 N/A, winz = 창 z-정규화 (pinball 1차 동차성에 맞춘 sd 표본가중 — docs/blockf_design.md §4). 블록 간 직접 비교 금지 원칙 유지 (docs/design_audit.md §1.3).
