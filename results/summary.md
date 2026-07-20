# Results Summary

## G3 완료 (2026-07-13)

**NWP 수집 완주**: 65,808/65,808 호출, 실패 0, 20.9GiB, 키 5개 로테이션 — 2021-07~2023-12 제주 13개 지점, 리드 매칭 밴드 2종 (26–49h → h≤24 / 50–73h → h≤48)

**jeju_wind 큐레이션**: 21,720시간 (2021-07~2023-12), 발전량↔예보풍속 상관 **0.742(da)/0.704(d2)** — 리드 증가 시 감소하는 물리적 정합. KMA 아카이브 홀 1건(2023-06-25~07-04, 제주 영역 −99) → 세그먼트 인지 윈도잉으로 처리 (보간 없음)

**jeju_wind LPS = 0.745** (w=96 LightGBM; ridge 0.739) — 계획서 예상("높음") 적중


**ETTh1 h=96 sanity (문헌 범위 확인)**: PatchTST+RevIN **0.383** / SegRNN+RevIN **0.364** / SAN+RLinear **0.404** / FAN+RLinear **0.413** / LGBM-DMS+win-znorm **0.382** — 전부 문헌 범위 내

**LPS (results/lps.csv, w=96 LightGBM 기준)**:

| 그룹 | 데이터셋 | LPS | 예상 부합 |
|---|---|---|---|
| 외생 구동 | gefcom_wind (NWP 풍속) | **0.82** (ridge 0.92) | ✅ 높음 |
| 외생 구동 | gefcom_load (기온) | **0.89** | ✅ 높음 |
| 외생 구동 | gefcom_solar (NWP 12종) | **0.88** | ✅ 높음 |
| 표준 LTSF | etth1 (달력만) | **−0.23** | ✅ 낮음 |
| 표준 LTSF | weather (달력만) | **−0.54** | ✅ 낮음 |
| 표준 LTSF | electricity (달력만) | 0.58 | 중간 |
| 표준 LTSF | etth2 (달력만) | 0.89 ⚠️ | 예상 밖 높음 — w·모델 간 불안정(0.03~0.93), τ 결정 시 정의 민감도 이슈로 다룰 것 |
| 단기 | kpx_demand_national (달력만, 1년) | −0.20 | 기온 공변량 추가 여지 |

- 주목: gefcom_load에서 ridge 0.12 vs LightGBM 0.89 — 기온-부하 U자 비선형이 1단계 비선형 회귀 필요성을 실증
- jeju_wind: KMA NWP 수집 완료 후 산출 예정

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

## G4 실데이터 grid — GATE 2: **GO ✅ 8/8 적중** (2026-07-16 공식 판정)

- 블록 A 완주 1,794/1,794 (사전 등록 4백본). 판정: `results/gate2.md`, Tab 1: `paper/tables/tab1_draft.md`
- 사전 등록(commit cab17c1, 실행 전 시각 증빙) 부호 예측 8/8 — GATE 2 기준(≥6/8) 초과 달성
- MCS(α=0.10): 외생 11셀 전부 {condnorm} 단독 생존 / 표준 12셀은 revin·san·fan 계열 생존, condnorm 탈락
- SAN·FAN도 외생 그룹에서 CondNorm에 전패 → "적응형 정규화가 이미 해결" 공격 봉쇄
- 블록 B(정보 대등): 기본 3백본 완료 — CN 우세 DM 유의 10/11, RAW+cov>RevIN+cov 전 셀
- 블록 C(SOTA): TimeXer-MS 중간 격차 +0.655 (진행 중)

## G4 확장 블록 완주 + G5 분석 (2026-07-16)

- **전 블록 완결**: A 1,794 + B 1,133 + C/D 1,275 = **4,202 runs**
- **블록 B (정보 대등, 5백본)**: CondNorm이 linmix·mlpmix·patchtstcov 3개 백본에서 최적, lgbmcov·segrnncov는 RAW+cov가 근소 우위(각 −0.009/−0.004)이며 CN은 최적과 0.01 이내. DM 유의(CN>RevIN): patchtstcov 11/11, linmix·mlpmix 10/11, segrnncov 2/11, lgbmcov 3/11. RAW−CN 격차는 mixer 사다리에서 단조 감소: linmix +0.127 → mlpmix +0.052 → lgbmcov −0.009 (용량 서사; patchtstcov +0.142은 예외로 명시). RevIN은 전 백본에서 RAW+cov보다 열등
- **블록 C (SOTA-MS)**: TimeXer-MS 격차 +0.412 (55쌍), iTransformer-MS 동일 패턴 — SOTA 공변량 모델 위에서도 CN 압승
- **Fig 3**: Spearman ρ=0.762 (p=0.028), (dataset,h) ρ=0.783 (p<1e-5) — G5 AC 유의성 충족
- **Fig 5**: 외생 4종 전부 격차 h-단조 증가 (명제 3 실데이터 실증)
- **Tab 2**: 분산 귀속 — norm-관련 48.2% vs backbone-관련 0.6% (norm×dataset 상호작용 47% = 적용 경계 그 자체)
- **Tab 3**: τ ∈ [0.30, 0.70]에서 8/8 유지 (사전 등록 τ=0.3 포함)
- `make figures` / `make tables` 재현성 확인

## G5 마감 — ΔLPS (2026-07-20, results/lps_delta.csv)

| 데이터셋 | ΔLPS | R²_pers | 읽기 |
|---|---|---|---|
| jeju_wind / gefcom_wind | 0.742 / 0.773 | ≈0 | 수준 신호가 순수 외생 (지속성 무력) |
| gefcom_load / solar | 0.335 / 0.556 | 0.56 / 0.32 | 강한 지속성 위에서도 공변량 증분 큼 |
| etth1 / etth2 | −0.142 / −0.028 | 0.47 / 0.45 | 달력 공변량은 지속성 대비 무가치 — etth2 절대 LPS 불안정성 해소 |
| electricity | 0.031 | 0.64 | 사전 등록 최저신뢰 셀: 최강 지속성 + 무증분 = 실측 gap ≈0 설명 |
| weather | 0.558 | −0.12 | 지속성 부재 → ΔLPS 적용 대상 아님 (절대 LPS 규칙 유지, 논문에 명시) |

## G6 완료 — IJF 초고 (2026-07-20)

- `paper/main.tex` + `paper/sections/` 10개 파일 + `references.bib`(26건) — elsarticle(authoryear), tectonic 46쪽 무결 컴파일, `make paper`
- 구성: §1 서론 / §2 관련연구 / §3 이론(M1·명제1–3·명제2′·Fig1) / §4 LPS(τ=0.3·사전등록·ΔLPS) / §5 합성(GATE1) / §6 실증(Tab1–3·Fig3–5·블록B/C/D) / §7 토론+결론 / 부록 A 증명·B 설계감사·C 재현성·D 확장표+질적그림
- 검증: 6-체커 적대 검증으로 25건 결함 수정 (수치 전수 대조, 수학 재유도, 인용 26/26 대조, PDF 45쪽 육안)
- TODO 잔여: 저자 메타데이터 1건(주석). IJF 체크리스트: paper/IJF_CHECKLIST.md
