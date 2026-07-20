# Block F 설계 — 확률적 (분위수) 예측으로의 경계 확장 (G7, 2026-07-20)

> 질문: RevIN/CondNorm 경계가 조건부 평균(MSE)이 아니라 **예측 분포**(pinball/CRPS,
> 구간 커버리지)를 목표로 할 때도 유지·증폭되는가.
> 러너: `experiments/g7_blockf.py` → `results/g7_blockf.csv` + `results/g7_errors/`.
> 표: `experiments/g7_blockf_table.py` → `paper/tables/tabF_probabilistic.md`.
> Block A–D 산출물(`results/g4_*`)은 동결 — 여기서는 읽기 전용(튜닝된 L, 1단계 캐시).

## 1. 백본 2종 제한의 근거

정규화 효과 분리(normalization-effect isolation)가 목적이므로 백본 축은
**용량 축의 양 끝 대표 2종**으로 고정한다:

- `rlinear_q` — 선형(Prop-1 이론 함수 클래스와 동일 축), 9-분위수 헤드.
  Block A의 rlinear와 분위수당 용량 동일(각 분위수 = 창의 독립 선형 사상 1개).
- `lgbm_q` — 분위수 목적함수를 **네이티브**로 지원하는 GBM (objective='quantile').

백본을 더 늘리면 비용은 배수로 늘지만 비교 대상(정규화 arm 대비)은 변하지
않는다 — Block A에서 이미 정규화 효과의 백본 견고성을 4종으로 확인했으므로,
Block F의 추가 백본은 한계 정보가 없다. (블록 자기완결 원칙: 이 블록의 행은
블록 내부에서만 비교, docs/design_audit.md §1.3)

## 2. 그리드·예산 (Block A 상속)

- 데이터셋 8종, horizon은 grid 기본값(jeju_wind 24/48, 나머지 24/96/336),
  분할·전역 z-score도 `experiments/g4_grid.py`의 `build_frame`/`split_starts` 재사용.
- `rlinear_q`: epochs 12 / patience 3 / seeds 0–4 / lr 5e-3 (Block A rlinear 예산).
  lookback은 (dataset, rlinear, h)별 Block A 튜닝값을 `results/g4_grid.csv`의
  L 컬럼에서 복원해 **재튜닝 없이 동결** (arm 간 용량 동일 불변식 유지).
- `lgbm_q`: L=336 고정(Block A LGBM_L), 결정적이므로 seed 0 단일.
  행 캡 `G7_LGBM_MAX_ROWS`(기본 250k, arm 간 동일 — capacity-fair),
  스레드 `G7_LGBM_JOBS`(기본 8 — 소규모 데이터에서 전코어 LightGBM 스래싱
  방지, `experiments/compute_lps_delta.py`의 교훈).
- CondNorm 1단계는 Block A 캐시(`curated/firststage/*.npy`)를 그대로 읽는다
  (재적합 없음 → Block A와 동일한 m_hat).

## 3. 지표 정의 (전 지표 전역 z-score 공간)

- `pinball` = 분위수 집합 전체 평균 pinball loss.
- `crps` = **2 × pinball** — CRPS = 2∫₀¹ pinball_q dq 의 유한 격자 근사.
  9점(rlinear_q)/3점(lgbm_q) 격자의 리만 근사이므로 **절대값은 격자 의존적**이고,
  동일 격자 내 arm 간 비교로만 사용한다. 격자가 다른 두 백본의 crps는
  서로 비교하지 않는다.
- `cov80` = P(q̂₀.₁ ≤ y ≤ q̂₀.₉) (명목 0.80), `cov_lo` = P(y ≤ q̂₀.₁),
  `cov_hi` = P(y ≤ q̂₀.₉) — 꼬리별 miscoverage 분해용.
- 비교차(non-crossing): **추론 시 분위수 정렬**(rearrangement, Chernozhukov
  et al. 2010). 학습은 비정렬 헤드 그대로(표준 관행), 정렬은 pinball을
  악화시키지 않는 후처리다. 검증 조기종료 지표는 학습 목적함수와 동일한
  비정렬 pinball.

## 4. arm별 분위수 복원 규약

핵심 불변식: **모든 복원(denorm)이 원소별 단조 증가 사상**이므로 분위수
동변성(quantile equivariance)이 성립 — 정규화 공간의 q-분위수를 복원하면
원공간의 q-분위수가 된다. 따라서 Block A의 arm 구성을 그대로 이식할 수 있다.

### rlinear_q: raw / revin / san / fan / condnorm

- **raw**: 항등. **revin**: 창 통계 affine — 모든 분위수 채널에 동일 적용.
- **san**: 통계 사전학습 프로토콜 **무변경**(stats_loss는 백본 목적함수와
  독립). 복원은 예측된 슬라이스 (mean, std)로 분위수마다 적용 — pred_std는
  ReLU 출력 ≥ 0이라 단조성 보장.
- **fan**: 예측 주파수 성분을 더하는 위치 이동(단조) — 분위수마다 동일 적용.
  공식 aux loss의 잔차 MSE 항은 "백본 출력 ≈ 참 잔차" 회귀인데, 분위수
  설정에서의 자연스러운 대응은 **중앙값(q=0.5) 헤드**다(pinball@0.5가
  중앙값을 표적). 구현: `_pred_residual`을 중앙값 헤드의 정규화 공간
  출력으로 지정 후 공식 `aux_loss(y)` 호출. 주파수 예측 항
  `lf(pred_main, true_main)`은 무변경. — 프로토콜상 건전하다고 판단해
  N/A 처리하지 않음; 이 중앙값 대응이 유일한 각색이며 여기 명시로 고정.
- **condnorm**: 데이터 공간 변환(모델측은 raw) — 잔차 표준화 공간에서 분위수
  학습 후 `q̂·σ_r + μ_r + m̂(타깃 시점)` 복원. m̂ 가산은 이동이므로 단조.

### lgbm_q: raw / winz / condnorm (SAN/FAN 구조적 N/A — Block A와 동일)

- SAN/FAN은 torch 모듈(통계 MLP·주파수 MLP)이 백본과 공동학습되는 구조라
  GBM에 이식 불가 — Block A lgbm_dms와 동일한 사유로 N/A.
- **winz**: `src/models/lgbm_dms.window_znorm`의 sd-floor(√(var+eps))
  그대로. 표본가중은 **sd¹** — pinball은 스케일 1차 동차이므로 sd 가중이
  "정규화 공간 목적함수 = 원공간 pinball"을 만든다 (MSE의 sd² 규약은 2차
  동차성 전용이었음; 규약의 *의도*를 승계한 것).
- **condnorm**: Block A lgbm_run과 동일한 잔차 표준화·복원 경로.

## 5. lgbm_q 분위수 축소 {0.1, 0.5, 0.9}의 근거

h=336 직접 다단계(DMS) LightGBM은 Block A에서 확인된 비용 병목(스텝당
부스터 1개)이고, 분위수는 그 비용을 다시 |Q|배 한다(h=336 × 9 = 3024
부스터/셀은 비현실적). {0.1, 0.5, 0.9}면 cov80/cov_lo/cov_hi는 **정확히**
계산되고 pinball/crps는 성긴 격자 근사로 얻는다 — lgbm_q의 역할은 경계의
GBM 교차 확인이지 조밀한 분포 추정이 아니다. rlinear_q(9점)와 crps 절대값을
비교하지 않는 이유이기도 하다(§3).

## 6. 산출물·재개 규약

- `results/g7_blockf.csv`: `dataset,arm,backbone,h,seed,pinball,crps,cov80,cov_lo,cov_hi`
  — 기존 키 (dataset,arm,backbone,h,seed) 행은 스킵(재개 가능).
- `results/g7_errors/{dataset}_{arm}_{backbone}_{h}_{seed}.npy`: 원점별 평균
  pinball (사후 DM 검정용).
- MLflow(선택): experiment `ini/norm-boundary`, run 이름
  `g7_{dataset}_{arm}_{backbone}_{h}_{seed}`, 실패해도 실험은 계속.
- CLI: `--datasets --backbones --arms --horizons --max-runs` (워커 분할용).
