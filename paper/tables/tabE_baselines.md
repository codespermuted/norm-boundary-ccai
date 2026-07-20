# Tab E 초안 — Block E baselines & published-default ablation (dataset × arm, h·seed 평균, std-scale)

Block E는 **자체 완결(self-contained)** 블록이다: 모든 비교는 블록 내 동일 세팅(동일 분할·lookback·예산)의 arm 사이에서만 유효하며, Block A/B 표의 수치와 직접 비교하지 않는다 (revin_all의 비교군 linmix_raw/linmix_revin/linmix_condnorm이 블록 내에 포함된 이유). 지표는 본 그리드와 동일한 train-적합 global z-score 공간의 MSE/MAE. nMAE(jeju_wind 한정)는 원 스케일 MAE를 train 구간 y 최대값(용량 프록시)으로 나눈 값.

| dataset | metric | seasonal_naive | climatology | first_stage_only | dynreg | revin_all | linmix_raw | linmix_revin | linmix_condnorm |
|---|---|---|---|---|---|---|---|---|---|
| jeju_wind | MSE | — | — | — | — | — | — | — | — |
| jeju_wind | MAE | — | — | — | — | — | — | — | — |
| jeju_wind | nMAE | — | — | — | — | — | — | — | — |
| gefcom_wind | MSE | — | — | — | — | — | — | — | — |
| gefcom_wind | MAE | — | — | — | — | — | — | — | — |
| gefcom_load | MSE | — | — | — | — | — | — | — | — |
| gefcom_load | MAE | — | — | — | — | — | — | — | — |
| gefcom_solar | MSE | — | — | — | — | — | — | — | — |
| gefcom_solar | MAE | — | — | — | — | — | — | — | — |
| etth1 | MSE | 0.6689 | 0.8848 | 1.4044 | 0.3553 | — | — | — | — |
| etth1 | MAE | 0.5106 | 0.7118 | 0.8680 | 0.3893 | — | — | — | — |
| etth2 | MSE | — | — | — | — | — | — | — | — |
| etth2 | MAE | — | — | — | — | — | — | — | — |
| weather | MSE | — | — | — | — | — | — | — | — |
| weather | MAE | — | — | — | — | — | — | — | — |
| electricity | MSE | — | — | — | — | — | — | — | — |
| electricity | MAE | — | — | — | — | — | — | — | — |
