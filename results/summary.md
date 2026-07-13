# Results Summary

## G2 합성 실험 — GATE 1: **GO ✅** (2026-07-13)

- 격자: λ 11 × h {24,96,336} × L {96,336} × 시드 10 × 정규화 4 = **2,640 runs** (RLinear 고정, `results/synth_grid.csv`, MLflow 기록)
- **기준 1** (교차점 ±0.1): 6/6 통과 — λ*_emp vs λ*_OLS이론 최대 편차 **0.015** (L=96: 0.009/0.007, 0.011/0.009, 0.022/0.022; L=336: 0.018/0.007, 0.023/0.008, 0.029/0.021)
- **기준 2** (h 격차 확대): 2/2 통과 — λ=0.8에서 시드쌍 Δgap: L=96 +0.170±0.034, +0.031±0.024 / L=336 +0.128±0.019, +0.040±0.016
- **기준 3** (CN-est 저하의 이론 설명): 실험/이론 저하 비율 중앙값 **1.12**
- 핵심 발견 (명제 2′): 선형 백본의 암묵적 수준 추적으로 λ*_M1(≈0.27, 복원 규칙 상한) ≫ λ*_실제(≈0.01–0.03) — 상세 `results/gate1.md`, `docs/theory_g1.md` §5.1
- Fig 2: `paper/figures/fig2_synth.{pdf,png}`

## G1 이론 수치화 (2026-07-13)

- 닫힌형 MSE (RAW/IN/CN-oracle/CN-est) vs 몬테카를로(n=2M): 전 파라미터 격자에서 **상대오차 < 1%** (pytest 고정, `tests/test_theory.py`)
- λ* 임계 함수 검증: 교차 항등식(MSE_CN(λ*)=MSE_IN), h 단조 감소, 드리프트 단조 증가 모두 pytest 고정
- 기준 파라미터(V=1, w=96, σ_z=1, σ_u²=0.0036, σ_est²=0.02, σ_Δ=0)에서 λ*: h=24 → **0.923**, h=96 → **0.664**, h=336 → **CN 전역 지배** (λ*<0)
- 명제 1 상호작용 gap: 닫힌형 κ²[Var(ȳ)Var(g)+Cov²] vs OLS MC 일치 (<1%)
- Fig 1 생성: `paper/figures/fig1_dominance.{pdf,png}` — 유도·명세는 `docs/theory_g1.md`

## G0 스모크 (2026-07-13)

| run | dataset | norm | backbone | L | h | seed | test MSE | test MAE |
|---|---|---|---|---|---|---|---|---|
| etth1_revin_rlinear_96_0 | ETTh1 | RevIN | RLinear | 336 | 96 | 0 | 0.4067 | 0.4193 |

- 원본: MLflow `sqlite:///mlflow.db`, experiment `ini/norm-boundary`
- 목적: 파이프라인 개통 확인 (하이퍼파라미터 미튜닝 — 문헌 대조는 G3에서)
