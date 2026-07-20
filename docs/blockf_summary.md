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
