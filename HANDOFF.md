# HANDOFF — 현재 상태와 다음 단계

- **최종 갱신**: 2026-07-13
- **완료 Phase**: G0 ✅ → G1 ✅ → G2 ✅ **[GATE 1: GO]**
- **다음 Phase**: **G3 — 정규화·백본 구현부** (RESEARCH_PLAN.md §10, 부록 A의 [G3] 프롬프트)

## G2 AC 자체 평가 — GATE 1 판정: **GO ✅**

| 기준 (§4.3) | 판정 | 근거 |
|---|---|---|
| IN vs CN 교차 발생 + 교차점 이론 ±0.1 일치 | ✅ 6/6 | λ*_emp vs λ*_OLS이론 최대 편차 0.015 (`results/gate1.md`) |
| h 커질수록 격차 확대 | ✅ 2/2 | 시드쌍 95% CI에서 Δgap 전부 양수·유의 |
| CN-est 저하가 1단계 오차로 설명 | ✅ | 실험/이론 저하 비율 중앙값 1.12 |
| DGP 자체 검증 (λ ≈ R²_level) | ✅ | `tests/test_dgp.py` — λ∈{0,0.3,0.7,1.0}에서 \|R²−λ\|<0.08 |
| 전체 pytest | ✅ | 32 passed |

## G2의 핵심 학술 발견 (논문 반영 필수)

1. **명제 2′ (이론 정련)**: 실험 λ*(0.01–0.03)는 양식화 M1의 λ*(≈0.27)보다 훨씬 작다.
   원인 = 선형 백본의 **암묵적 수준 추적** (잔차 윈도우에서 남은 드리프트를 OLS적으로 흡수).
   → 이론값은 각 정규화 클래스의 **제약 OLS 닫힌형**(`src/theory/linear_class.py`)으로 계산하며
   실험과 ±0.015 수준으로 일치. M1은 해석층(상한)으로 유지. `docs/theory_g1.md` §5.1.
2. **실무 함의 강화**: 공변량 직교 드리프트가 없으면 CN은 극소 λ부터 우세 — IN의 실질 영역은
   드리프트 강건성 필요 구간. 실데이터(G4)에서 LPS-부호 예측의 성공 조건이 더 명확해짐.
3. RAW ≈ IN (in-distribution): 선형 백본이 윈도우 평균을 흡수하므로 — RevIN의 이득은
   순수한 수준 적응이 아니라 분포 이동 강건성이라는 재해석 (Discussion 소재).

## 산출물

- `results/synth_grid.csv` (2,640 runs) / `results/synth_ols.csv` (닫힌형 이론) / `results/gate1.md`
- `paper/figures/fig2_synth.{pdf,png}` — 이론 곡선 위 실험 점 (Fig 2 완성)
- `curated/synth/` — 시리즈+1단계 캐시 (110 npz, gitignore)
- MLflow: `synth{λ}_{norm}_rlinear_{h}_{seed}_L{L}` 규약으로 전 run 기록

## G3 착수 정보

1. **SAN·FAN 공식 구현 이식** (커밋 해시 고정) + 원 논문 대표 수치 재현 테스트
   - SAN: github.com/icantnamemyself/SAN (NeurIPS 2023) / FAN: NeurIPS 2024 공식 레포 확인
2. CondNorm 실데이터 버전 (`src/norms/condnorm.py`): 1단계 LightGBM + 가역 변환 + 가역성·누수 pytest
3. 백본 추가: PatchTST, SegRNN, LightGBM-DMS — ETTh1에서 문헌 범위 성능 확인
4. **데이터셋 7종 로더 + LPS 계산기** (`src/theory/lps.py`, 시간순 CV)
   - ⚠️ **사용자 확인 필요**: Jeju wind (KPX 집계 발전량 + 보관 NWP + 설비용량) — 사내 데이터 위치/포맷
   - KPX 계통 수요 (공개), GEFCom2014 (다운로드 경로 확인 필요), ETTh1/h2·Electricity·Weather (공개)
5. AC: 전 구성요소 pytest 통과 / `results/lps.csv` 산출
6. G4 사전 등록 전에 novelty 재스윕 1회 (docs/novelty_sweep.md 갱신)

## 환경 리마인드

- uv 전용, GPU1 고정, MLflow `sqlite:///mlflow.db`, experiment `ini/norm-boundary`
- 이론–실험 정합 규약: the environment contract + `docs/theory_g1.md` 대응표가 계약
