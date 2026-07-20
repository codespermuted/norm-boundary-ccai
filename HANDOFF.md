# HANDOFF — 현재 상태와 다음 단계

## G5 필수 ablation (2026-07-15 사용자 논의에서 확정)

1. **공변량-입력 baseline**: 외생 그룹 4개 × 대표 백본에서 "RevIN + 공변량을 입력 채널로 추가" arm — "정보량 vs 주입 메커니즘" 리뷰 공격 봉쇄용.
2. **CN 실패 크기의 이론 연결**: 표준 벤치마크에서 CondNorm 대패(etth1 2.4 vs revin 0.29)를 1단계 OOS 오차 전파 항으로 정량 분해 (Discussion).
3. (선택) 1단계 val-R² ≤ 0 시 m̂→전역평균 수축 안전장치 — "확장"으로 제안, 사후수정 아님을 명시.
4. 부록 그림 후보: paper/figures/sample_forecasts.png (질적 예측 비교 — 램프 추적/흐린 날 사례).

- **최종 갱신**: 2026-07-13
- **완료 Phase**: G0 ✅ → G1 ✅ → G2 ✅ **[GATE 1: GO]** → G3 ✅
- **다음 Phase**: **G4 — 사전 등록 → 실데이터 grid (GATE 2)** (RESEARCH_PLAN.md §10, 부록 A [G4])

## G3 AC 자체 평가

| AC | 판정 | 근거 |
|---|---|---|
| SAN·FAN 공식 구현 이식 (커밋 고정) + 대표 수치 재현 | ✅ | SAN `7e1ca66`, FAN `838e1b0` 이식. ETTh1 h=96: SAN+RLinear 0.404 / FAN+RLinear 0.413 — 문헌 범위 내. 2단계 사전학습·보조손실 train.py 통합 |
| CondNorm (1단계 LightGBM + 가역 변환) + 가역성·누수 테스트 | ✅ | `src/norms/condnorm.py` — inverse(transform)==y 정확 가역, train-only fit 누수 테스트 |
| PatchTST/SegRNN/LightGBM-DMS + ETTh1 문헌 범위 | ✅ | PatchTST 0.383 / SegRNN 0.364 / LGBM-DMS 0.382 (h=96, 전부 범위 내) |
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

## Discussion 추가 논점 (2026-07-16 사용자 관찰)

- covfair에서 백본 유연성↑·공변량 접근↑일수록 CN의 한계효용 감소: linmix(RAW−CN +0.127) → mlpmix(+0.053) → lgbmcov(−0.009).
  (2026-07-20 정정: 최종 tabB 기준 재산출. 이전 +0.135/+0.064/≈0은 gefcom_solar 재실행 전 중간값.
  주의: patchtstcov +0.142, segrnncov −0.004 — 단조 서사는 mixer 사다리(linmix→mlpmix→lgbmcov)에 한정할 것.)
  이는 비제약 클래스에서 정규화가 재매개화라는 Toner&Darlow 논리 및 명제 2′와 정합.
  RevIN과의 격차는 함수 클래스 제약이라 용량과 무관하게 잔존 — "CN의 가치 = 제약된 딥 파이프라인의 수선 + 모듈적 주입"으로 정밀화하여 Discussion에 수록할 것.

## G5 진단 정밀화 (2026-07-16 SMP 논의)

- **증분 LPS (ΔLPS)**: R²(ȳ_fut ~ cov + ȳ_past) − R²(ȳ_fut ~ ȳ_past). 지속성 강한 시리즈(SMP류)에서
  공변량-이력 중복으로 절대 LPS가 과대평가되는 사각지대 보정. 8개 데이터셋에 병산, τ 민감도와 함께 보고.
  사전 등록 규칙(절대 LPS)은 유지, ΔLPS는 정밀화 제안으로 명시.
- (선택) SMP 스트레스 테스트: EPSIS 시간별 SMP + 연료가·수요 공변량 — "높은 LPS·강한 지속성" 사분면의
  탐색 데이터셋 (사전 등록 외, 부록). 이론 예측: 평시 IN 우세, 레짐 전환 구간 CN 우세.

## G4 종료 — GATE 2 판정 (2026-07-16)

**판정: GO ✅ (8/8 적중, 기준 6/8)** — 상세는 results/gate2.md, results/summary.md.
잔여: 블록 B-deep(슈퍼바이저 진행 중) → 블록 C SOTA → 완료 후 G5 진입.
G5 착수 시: Fig 3(LPS vs 격차 + 사전등록 적중, Spearman ρ), Fig 4(분산 귀속), Fig 5(h별 격차),
Tab 2(MCS·귀속), Tab 3(τ 민감도 + ΔLPS 병산), covfair·SOTA 최종 표, make figures 재현성.

## G5 종료 — AC 자체 평가 (2026-07-20)

| AC | 판정 | 근거 |
|---|---|---|
| 분산 귀속 (정규화/백본/데이터셋/상호작용) | ✅ | Tab 2: norm-관련 48.2% vs backbone-관련 0.6%, norm×dataset 47.1% |
| Fig 3–5 생성 | ✅ | Fig 3 ρ=0.762 (p=0.028), (dataset,h) ρ=0.783 (p<1e-5); Fig 5 외생 4종 h-단조 |
| τ 결정 + 민감도 | ✅ | τ=0.3 (사전 등록), 8/8 유지 구간 [0.30, 0.70] (Tab 3a) |
| make figures 재현 | ✅ | figures/tables/paper 타깃 전부 재실행 확인 |
| ΔLPS 병산 (2026-07-16 SMP 논의 후속) | ✅ | results/lps_delta.csv — 외생: jeju .742/gefwind .773/load .335/solar .556, 표준: etth1 −.142/etth2 −.028/elec .031(R²pers .64)/weather .558(R²pers −.12). etth2 불안정성 해소 + electricity 경계 셀 설명. Tab 3(b)에 R²_pers와 병기 |

- ΔLPS 러너: `experiments/compute_lps_delta.py`. **주의: LightGBM sklearn 래퍼는 소형 데이터에서
  전-코어 기본값이 스레드 경합으로 fit당 ~90초까지 퇴화** — n_jobs=1 강제 몽키패치로 해결 (48분 2개 → 8분 8개).

## G6 종료 — AC 자체 평가 (2026-07-20)

| AC | 판정 | 근거 |
|---|---|---|
| 컴파일되는 초고 | ✅ | paper/main.tex, elsarticle(authoryear)+tectonic, 46쪽, 에러·미해결참조 0 (`make paper`) |
| 그림·표 전부 삽입 | ✅ | Fig 1–5 + 부록 질적 그림, Tab 1–3 + 부록 Tab audit/blocks/covfair/sota |
| TODO ≤ 20 | ✅ | 렌더링 TODO 0개 (main.tex 주석 1개: 저자 메타데이터) |
| 리뷰 방어 4종 | ✅ | §1(RevIN 만능 주장 안함)·§6.8(SAN/FAN 전패)·§6.8(정보대등=피처엔지니어링 반박)·§6.8(벤치마크 재현=에너지 편중 반박) |
| IJF 체크리스트 | ✅ | paper/IJF_CHECKLIST.md |
| 증명 부록 | ✅ | app_proofs: M1 유도, 명제 1–3 완전 증명, 명제 2′ 제약-OLS, 파라미터 매핑 |

- 품질 관리: 초안을 6개 검증 에이전트(수치/수학/구조/인용/과대주장/PDF육안)로 공격 → 25건 수정.
  대표: 블록 B "전 백본 최적" 과장 정정, 낡은 covfair 격차(+0.135/+0.064)를 +0.127/+0.053/−0.009로
  전면 정정(HANDOFF가 오염원이었음 — 본 문서 Discussion 논점도 수정 완료), GATE1 편차 0.015 vs 교차위치 0.029 혼동 정정.
- 잔여(투고 전): 저자·소속 기입, cover letter, 공개 저장소 전환 결정, (선택) revin_all ablation·SMP 스트레스 테스트.
