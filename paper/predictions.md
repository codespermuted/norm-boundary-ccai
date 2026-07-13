# 사전 등록 (Pre-registration) — G4 실데이터 grid 부호 예측

- **등록일**: 2026-07-13 (요인 격자 실행 **전** — 본 문서의 git commit 시각이 증빙)
- **근거 데이터**: `results/lps_official.csv` (이 시점까지 어떤 grid run도 실행되지 않음)
- **판정 기준 (GATE 2)**: 아래 부호 예측의 데이터셋 단위 적중률 ≥ 70% (RESEARCH_PLAN.md §8)

## LPS 공식 정의 (이 문서로 고정)

- 윈도우 w = 96, 1단계 g = LightGBM (CondNorm 1단계와 동일 계열), 시간순 확장 CV (min_train_frac 0.4, 5 fold)
- 다변량 데이터셋: **채널별 LPS의 평균** (grid MSE가 채널 평균이므로)
- 공변량: grid에서 CondNorm이 받는 것과 동일 집합 — 외생 구동 그룹은 도메인 공변량(리드 매칭 NWP·기온)+달력 조화항, 표준 LTSF 그룹은 달력 조화항만
- Ridge LPS는 민감도 보고용 (결정 규칙에 불사용)

## 결정 규칙과 τ

τ_prereg = **0.3**. 근거: G2 합성 실험에서 복원 규칙만의 양식화 상한 λ*_M1 ≈ 0.27–0.28 (h=24),
실측 교차점은 0.01–0.03. τ=0.3은 RevIN 쪽으로 보수적인 상한 채택 (드리프트·1단계 오차 마진 포함).

```
LPS ≥ 0.3  →  CondNorm 우세 예측 (RevIN − CondNorm 격차 부호: +)
LPS < 0.3  →  RevIN 우세 예측 (격차 부호: −)
```

## 데이터셋별 부호 예측

| 데이터셋 | LPS (공식) | LPS (ridge, 민감도) | **예측 부호 (RevIN−CondNorm)** |
|---|---|---|---|
| jeju_wind | 0.745 | 0.739 | **+ (CondNorm 우세)** |
| gefcom_wind | 0.744 | 0.896 | **+ (CondNorm 우세)** |
| gefcom_load | 0.894 | 0.118 | **+ (CondNorm 우세)** |
| gefcom_solar | 0.875 | 0.890 | **+ (CondNorm 우세)** |
| etth1 | −0.717 | −0.510 | **− (RevIN 우세)** |
| etth2 | −0.205 | 0.046 | **− (RevIN 우세)** |
| electricity | 0.283 | 0.266 | **− (RevIN 우세)** |
| weather | 0.110 | 0.347 | **− (RevIN 우세)** |

## 격차의 조작적 정의 (판정 방법 고정)

- 격차(dataset) = mean over {백본 4종(각자 밸리데이션 선택 lookback), h(데이터셋별 grid 값), 시드} 의
  [ test MSE(RevIN) − test MSE(CondNorm) ] (전역 z-score 스케일)
- 적중 = sign(격차) == 예측 부호. 8개 중 6개 이상 적중 시 GATE 2 통과.
- 보조 보고: (dataset × h) 셀 단위 적중률, DM 검정 유의성.

## 부수 예측 (보조, GATE 판정 비포함)

1. 격차의 크기는 LPS에 단조 증가 (Spearman ρ > 0, Fig 3)
2. jeju_wind에서 h=48 격차 > h=24 격차 (명제 3)
3. electricity(LPS 0.283, τ 근처)는 격차가 0 근방 — 부호 예측 신뢰도가 가장 낮은 셀로 사전 명시

## Grid 설계 고정 (§5.2 준수)

- 정규화 5: RAW / RevIN / SAN / FAN / CondNorm. 백본 4: RLinear / PatchTST / SegRNN / LightGBM-DMS
  (LGBM×SAN/FAN은 구조상 불가 — N/A로 사전 명시. LGBM의 RevIN 대응 = 윈도우 z-norm)
- h: 표준·gefcom {24, 96, 336} / jeju_wind {24, 48} (단기예보 리드 커버리지 제약, §5.2 "데이터 해상도 조정" 조항)
- lookback: {96, 192, 336, 720}에서 (dataset, backbone, h)별 RevIN seed0 val MSE로 선택 후
  **모든 정규화 arm에 동일 적용** (이론–실험 정합 규약: 용량은 arm 간 완전 동일)
- 시드 {0..4}, LGBM-DMS는 결정적 1회. drop_last=False, 전역 z-score는 train 통계만.
