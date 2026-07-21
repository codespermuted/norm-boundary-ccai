# G7 감사·강화 패스 v2 — AUDIT REPORT

- **최종판**: 2026-07-22 01:20 — 전 항목 종결 (9/9). 1차: 2026-07-21 00:40
- 대원칙 준수: **Block A–D 산출물 무수정·무재실행** (사전 등록 보존). 신규는 전부 `results/g7_*` / Block E·F 네임스페이스.

## 항목별 판정

### 1) 증거 체인 — ✅ 통과
`evidence/prereg_evidence.md`. 등록 커밋 cab17c1(07-13 13:20:23) → 최초 grid run(13:24:36), **+4분 13초**,
git·MLflow 태그(재작성 전 해시 279bb80)·npy mtime 3계열 일치. filter-branch(07-15)의 날짜 보존을 전 커밋에서
확인, 앵커 커밋 0ce99db 교차 검증. GitHub push는 원격 개설(07-15)이 grid 이후라 **보조 증거로 한계 명시**
(확장 블록·분석·원고에 대한 선행성만 입증). 18개 PushEvent 추출 완료.

### 2) 누수 카나리아 — ✅ 통과 (pytest 3종 6 tests, 전체 스위트 78 passed)
`tests/test_canaries.py` (marker `canary`):
- (a) 백색잡음 공변량 → 1단계 val R² = −0.121 ≤ 0, CondNorm이 Raw +11.8% 이내로 수렴 (참 공변량 시 0.08×Raw)
- (b) circular shift → 합성 외생 LPS 0.614 → 0.003 붕괴
- (c) 셔플 분할 누수 검출기(0.325× < 0.6 기준) + 실제 로더 train/test 인덱스 무중첩 검증 (etth1)

### 3) Block E — ✅ 완주 (552 rows, `results/g7_blocke.csv`, `paper/tables/tabE_baselines.md`)
- **first_stage_only ≈ full CondNorm** (외생: jeju .294 vs .307, wind .156 vs .180, solar .057 vs .064;
  표준: 동반 붕괴 etth2 3.37) — CN 가치의 본체 = 공변량→수준 사상
- **dynreg**(고전 동적회귀 대역): 풍력 경쟁력(jeju .285) / 비선형 부하 대패(.371 vs CN .108) — 고전 기법 방어선
- **revin_all**(published 기본값): 7/8 데이터셋에서 target-only revin과 ±noise. **gefcom_solar에서 전 시드
  체계적 발산**(h336 MSE 848–1059). 원인 실증: 강수 채널 VAR228 최소 윈도우 std 4.2e-6 (전역의 1/800) —
  간헐 공변량의 윈도우 정규화 폭발. design_audit §3의 target-only 선택을 정당화하는 **재현 가능한 반례**
- seasonal_naive·climatology 하한 앵커 확보. jeju nMAE 병기 (naive .225 → CN .101)

### 4) Block F — ✅ 완주 (rlinear_q 575/575 + lgbm_q 69/69 = 644 rows)
- **헤드라인**: 경계가 확률 지표에서 유지 — 외생 그룹 CN<RevIN pinball **11/11 셀 (rlinear_q)**,
  **11/11 셀 (lgbm_q)** — 두 백본 계열 교차 확인, 신경망 프로토콜 아티팩트 아님.
  외생 평균 pinball: CN 0.106 vs RevIN 0.197 (rlinear, −46%) / CN 0.085 vs winz 0.155 (lgbm).
  표준 그룹은 Block A 패턴 그대로 역전 (CN 0.521 vs RevIN 0.121)
- **상대 격차는 감쇠** (Block A MSE 73% → pinball 46%; 제곱→1차 동차 손실 효과) — 부호·전 셀 일관성 유지
- **뉘앙스 (정직 보고)**: CN cov80 0.60 vs RevIN 0.82 (명목 0.8) — 과소커버는 **하방 꼬리 집중**
  (P(y≤q10)=0.19 vs 명목 0.10; 상방 0.79≈명목): 1단계 불확실성 미전파 = σ_est 항의 확률 버전.
  향후 과제 명시 (`docs/blockf_summary.md` 최종 문단)
- 산출: `paper/tables/tabF_probabilistic.md`, `docs/blockf_design.md`(백본 2종 제한 사유·CRPS 근사·
  quantile 축소), `docs/blockf_summary.md`(수치 확정판)

### 5) Prop 1′ 비선형 데모 — ✅ 통과 (모순 없음)
`experiments/g7_prop1_demo.py` + `configs/g7_prop1.yaml`(하한 상수 `bound_scale` 교체 가능).
MLP[128,128]×10시드: RevIN 초과위험이 잠정 하한 κ²·Var(ȳ|g)·Var(g) **위** 안착 —
λ=0.4: 0.0851 vs 하한 0.0663 (마진 +0.019 ≫ 2SE 0.0009) / λ=0.8: 0.0672 vs 0.0525 (+0.015 ≫ 0.0018).
raw/CN은 하한의 ~1/20. 그림 `paper/figures/figG_prop1_mlp.{pdf,png}`. 모순 프로토콜 구현·미발동
(발동 시 CONTRADICTION.md + exit 1). 주의: 이 하한은 비선형 클래스용 조건부-분산 형태로, 선형 클래스
gap(κ²[VarVar+Cov²])과 의도적으로 다름 — 이론 확정 시 상수만 교체.

### 6) LPS 추론 모듈 — ✅ 완주 8/8
`src/theory/lps_inference.py` (+ pytest 8, 공식 LPS와 통계량 동일성 고정):

| dataset | LPS | p_perm | p_aligned | 90% CI | λ̂* |
|---|---|---|---|---|---|
| jeju_wind | .745 | **.006** | .039 | [.76, .88] | −0.20 |
| gefcom_wind | .744 | **.008** | .059 | [.70, .85] | −0.51 |
| gefcom_load | .894 | **.002** | .011 | [.90, .95] | 0.68 |
| gefcom_solar | .875 | **.005** | .035 | [.88, .95] | 0.09 |
| etth1 | −.717 | .39 | .38 | [−.02, .40]† | 2.26 |
| etth2 | −.205 | .19 | .19 | [−.29, .42]† | 2.90 |
| weather | .110 | .36 | .41 | [−.22, .66]† | 1.02 |
| electricity | .283 | .058 | .075 | [.38, .57]† | 1.35 |

외생 4종: 순열 유의(p≤.008) + CI 하한 > τ=0.3. 표준 4종: 비유의(electricity만 경계 .058).
**λ̂* 순서가 실측 승패와 8/8 정합** (음수=CN 전역 지배 ↔ 풍력 압승; >1=IN 지배 ↔ etth·weather·elec).
**electricity = 전 진단의 경계 셀** (LPS .283·p .058·λ̂* 1.35·ΔLPS .031·실측 gap −.027) —
사전 등록의 최저 신뢰 표기와 완벽 정합.
†주의: 음수/경계-LPS 시리즈의 MBB CI는 블록 재표집이 fold 간 드리프트를 희석해 상향 편의 —
표준 그룹에선 순열 p가 더 유의미한 통계량 (모듈 문서·results/lps_inference.md Caveats에 명시).
post-hoc 라벨: τ 규칙 불변.

### 7) 통계 강화 — ✅ 완료
`docs/stats_hardening.md` + `experiments/g7_fisher_check.py` → `results/g7_fisher_robustness.csv`:
- 8/8 이항 정확검정 P(X=8)=2⁻⁸≈0.0039 (관대 기준 P(X≥6)=0.145 대비 엄격 기준 통과) — §6 삽입 문장 제공
- Fisher 종속성: 32개 별표 중 **31개가 HMP·Simes 양쪽 생존**, 유일 반전 = etth2 RevIN(Fisher .026 →
  HMP .077) — 논문의 MCS {revin,san} 결과와 정합해 서사 무손상. 권고: 현 Fisher 유지+각주, wave 2는 HMP 사전 등록

### 8) 원고 수정 초안 — ✅ 4종 작성 (`docs/manuscript_revisions_g7.md`, main.tex 미수정)
(a) three-thresholds 박스(0.923 baseline / 0.27–0.28 synthetic-M1 = τ 앵커 / 0.009–0.029 OLS·경험)
(b) epistemic-status 문단(exact/bound/empirical 3층 + figG 데모) (c) τ–LPS 순서 공개(τ=0.25에서도 7/8 ≥ 기준 6/8)
(d) development(Jeju)/confirmation(GEFCom·표준) 라벨 각주

### 9) OSF 사전 등록 초안 — ✅ 갱신 (`docs/osf_prereg_draft.md`)
M5·SMP quadrant + 신규: permutation-CI 규칙 primary 승격, capacity-ladder 단조성(mixer 한정,
PatchTST-어댑터 사전 제외), 확률 지표 secondary(Block F 백본쌍 확정 반영). 미확정 항목 [확정 필요] 표기.

## 종합 — 최종 판정

**9/9 항목 전부 종결·통과.** 사전 등록 산출물(Block A–D·predictions.md·τ 규칙) 무수정 보존 하에:
- 어떤 항목에서도 본 논문의 주장을 **약화시키는 증거가 나오지 않았다**
- 주장을 **강화**하는 신규 증거: 확률 지표 경계 22/22 셀(두 백본), revin_all 재현 반례,
  first-stage 지배, LPS 순열 외생 4/4 유의, λ̂* 실측 8/8 정합, Fisher 별표 31/32 강건,
  비선형 MLP 하한 무모순, 이항 정확검정 2⁻⁸
- 정직 공개 항목: CN 확률 예측 하방 과소커버(σ_est 미전파), MBB CI 상향 편의, Fisher 종속성
  한계(HMP 각주 권고), GitHub push 증거의 적용 범위 한계 — 전부 문서화 완료

신규 실험 규모: Block E 552 + Block F 644 + LPS 추론(순열 ~2×10⁵ fit) = 사전 등록 외 보강 1,196 runs.
G7 총 실행 기간 2026-07-20 21:30 ~ 07-22 01:10, OOM 0회, 레인 자동 재시도 0회.
