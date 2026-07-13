# HANDOFF — 현재 상태와 다음 단계

- **최종 갱신**: 2026-07-13
- **완료 Phase**: G0 ✅ → G1 ✅ → G2 ✅ **[GATE 1: GO]** → G3 ✅
- **다음 Phase**: **G4 — 사전 등록 → 실데이터 grid (GATE 2)** (RESEARCH_PLAN.md §10, 부록 A [G4])

## G3 AC 자체 평가

| AC | 판정 | 근거 |
|---|---|---|
| SAN·FAN 공식 구현 이식 (커밋 고정) + 대표 수치 재현 | ✅ | SAN `7e1ca66`, FAN `838e1b0` 이식. ETTh1 h=96: SAN+RLinear 0.404 / FAN+RLinear 0.413 — 문헌 범위 내. 2단계 사전학습·보조손실 train.py 통합 |
| CondNorm (1단계 LightGBM + 가역 변환) + 가역성·누수 테스트 | ✅ | `src/norms/condnorm.py` — inverse(transform)==y 정확 가역, train-only fit 누수 테스트 |
| PatchTST/SegRNN/LightGBM-DMS + ETTh1 문헌 범위 | ✅ | PatchTST 0.383 / SegRNN 0.364 / LGBM-DMS (아래 참조) |
| 데이터셋 7종 로더 | ✅ | ETTh1·ETTh2 (`etth.py`) / electricity·weather (`ltsf.py`, 7:1:2) / jeju_wind·gefcom_wind·load·solar·kpx_demand (`covariate.py`+`curation.py`, 세그먼트 인지) |
| LPS 계산기 + results/lps.csv | ✅ | `src/theory/lps.py` (시간순 확장 CV, lgbm+ridge), 10개 데이터셋 산출 |
| 전체 pytest | ✅ | **56 passed** (가역성·누수·계약·세그먼트·구조 항등식) |

## LPS 요약 (w=96, LightGBM) — Fig 3의 x축 데이터

| 외생 구동 | LPS | 표준 LTSF | LPS |
|---|---|---|---|
| **jeju_wind** (KMA NWP) | **0.745** | etth1 | −0.23 |
| gefcom_load (기온) | 0.894 | etth2 | 0.89 ⚠️ |
| gefcom_solar (NWP 12종) | 0.875 | electricity | 0.58 |
| gefcom_wind (NWP 풍속) | 0.817 | weather | −0.54 |
| | | kpx_demand (달력만·단기) | −0.20/−0.0 |

## G4 착수 전 반드시 처리할 이슈

1. **LPS 정의 안정화** (사전 등록 전 필수): etth2가 달력만으로 0.89 (단 w·모델 간 0.03~0.93 불안정), gefcom 계열도 w=336 LightGBM에서 급락 (n_windows<60에서 GBM 과소적합; ridge는 안정). 대응 후보: (a) w=96 고정 + lgbm·ridge 최대값, (b) min_child_samples 완화, (c) 연 주기 항 제외 민감도. **τ와 함께 G4 사전 등록 문서에 정의를 고정할 것.**
2. **KPX 수요 다년치**: 포털이 최신 버전만 제공 (이력 미제공). 다년 수요는 GEFCom-load로 대체 가능하나, KPX 수요를 grid에 넣으려면 EPSIS(epsis.kpx.or.kr) 수동 확보 필요 — 사용자 확인 필요.
3. **제주 수요 기온 공변량**: NWP TMP band1 수집 완료 상태 (제주 ASOS 지점 4곳) — kpx_demand_jeju(2026년 6개월)와는 기간 불일치. 제주 수요 다년치 확보 시 재산출.

## G4 실행 요점

1. **사전 등록**: `paper/predictions.md`에 데이터셋별 (LPS, RevIN−CondNorm 부호 예측) 기록 후 **grid 실행 전 커밋**. novelty 재스윕 1회 포함 (§8).
2. Grid: 정규화 5 (RAW/RevIN/SAN/FAN/CondNorm) × 백본 4 × 데이터셋 × h {24,96,336} × 시드 5.
   - jeju_wind는 리드 매칭 제약상 **h ∈ {24, 48}** (band1/band2). h=336은 NWP 커버 불가 — 결정 필요: 제외 vs 지속성 공변량. 계획서 §5.2 "데이터 해상도에 맞게 조정" 조항 적용.
   - CondNorm 경로: `first_stage_level(covariates)` → `CondNormTransform` → 백본, train.py에 데이터셋 라우팅 추가 필요 (G2 runner의 방식 재사용).
   - LGBM×SAN/FAN 조합은 구조상 불가 (torch 통계 예측기) — grid에서 N/A 처리, 논문에 명기.
3. DM 검정·MCS 구현 (`src/eval/`) — G4에서 Tab 1과 함께.
4. MLflow `sqlite:///mlflow.db`, run명 규약 준수. NWP 원자료: `curated/raw/kma/nwp_*.parquet` (31개월), 수집기 재실행으로 재현 가능.

## 데이터 특이사항 (논문 Data 절에 기재)

- KMA 아카이브 홀: 2023-06-25~07-04 제주 영역 −99 (재시도로 영구 결측 확인) → 세그먼트 인지 윈도잉, 보간 없음
- jeju_wind 발전량↔ws_da 상관 0.742, ws_d2 0.704 (리드 감쇠 정합 — 공간 매핑 검증)
- 단기예보 아카이브는 2021-06 개편 이후 체계만 사용 (기간 2021-07~2023-12)
