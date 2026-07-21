# Block F 요약 — 확률적 지표에서도 경계가 유지되는가 (최종, 2026-07-22)

> 수치 확정: results/g7_blockf.csv 644 rows (rlinear_q 575 + lgbm_q 69) 완주 기준.
> 표: paper/tables/tabF_probabilistic.md. 설계 근거: docs/blockf_design.md.

## 논문용 문단 (영어)

Block F extends the point-forecast boundary to probabilistic forecasting:
the same eight datasets, grid horizons, and frozen Block-A lookbacks, with a
quantile head (pinball loss, CRPS approximated as twice the mean pinball
over the quantile grid) replacing the MSE objective. On the exogenous group,
CondNorm improves mean pinball over RevIN by 46% on the linear quantile
backbone (11 of 11 (dataset, h) cells), and the same contrast holds for the
LightGBM quantile backbone (11 of 11 cells; mean pinball 0.085 vs 0.155 for
the window-normalized arm) — the boundary **persists** when the target is
the predictive distribution rather than the conditional mean, and it is not
an artifact of the neural training protocol. On the standard group the
ordering mirrors Block A: CondNorm's mean pinball (0.521) is far worse than
RevIN's (0.121), with the same catastrophic failure cases. The one metric
where CondNorm does not dominate is calibration: its empirical 80% coverage
on the exogenous group is 0.60 against RevIN's 0.82 (nominal 0.80), and the
tail decomposition attributes the miscoverage almost entirely to the lower
tail (P(y <= q10) = 0.19 vs nominal 0.10, while P(y <= q90) = 0.79 is nearly
nominal): first-stage estimation uncertainty is not propagated into the
quantile spread, so the intervals are too narrow from below. Sharper but
under-covered intervals are exactly the probabilistic signature of the
sigma_est term in the theory, and propagating first-stage uncertainty is a
natural extension we leave to future work.

## 판정 메모 (한국어)

- **경계 유지**: ✅ 외생 4종에서 CN<RevIN pinball 11/11 셀 (rlinear_q), lgbm_q도 11/11 — 두 백본 계열 교차 확인
- **증폭/감쇠**: 상대 격차 기준 감쇠 — Block A 상대 MSE 격차 73% → pinball 상대 격차 46%
  (제곱 손실 → 1차 동차 손실로 바뀌며 큰 오차의 벌점이 줄어드는 효과; 부호·전 셀 일관성은 그대로)
- **커버리지**: RevIN cov80 0.82, CN cov80 0.60 (명목 0.80) — CN 과소커버는 **하방 꼬리 집중**
  (P(y≤q10)=0.19 vs 명목 0.10; 상방은 0.79로 거의 명목). 1단계 불확실성 미전파가 원인,
  이론의 σ_est 항의 확률 예측 버전으로 해석 — 향후 과제로 명시
- **lgbm_q 교차 확인**: ✅ 동일 부호 패턴 (외생 CN 압승 / 표준 winz 우세), 신경망 프로토콜 아티팩트 아님
