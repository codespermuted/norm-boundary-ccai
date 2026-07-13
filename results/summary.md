# Results Summary

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
