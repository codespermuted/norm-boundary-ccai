# When Instance Normalization Hurts
### Exogenously-Driven Level Predictability and Conditional Normalization for Multi-Step Forecasting — 연구 마스터 플랜

- **Status**: Draft v1.0 (2026-07)
- **Target venue**: 1순위 International Journal of Forecasting (Q1) / 2순위 Applied Energy·Energy / 3순위 Energy and AI
- **실행 방식**: 본 문서의 Phase Goal(G0–G6)을 Claude Code에서 순차 실행. 각 Goal은 자기완결적이며 수용 기준(AC)을 만족해야 다음 Phase로 진행.
- **Non-goals**: 새 SOTA 아키텍처 제안 ✗ / 확률 예측(pinball, CRPS) ✗ / Foundation model 비교 ✗ — 논문을 단순·명료하게 유지하기 위해 결정적(deterministic) 다중 단계 예측으로 범위 고정.

---

## 0. Thesis (한 문장 주장)

> **인스턴스 정규화(RevIN 계열)의 이득은 시계열 수준(level)의 외생적 예측가능성이 커질수록 단조 감소하며, 임계점 이후에는 공변량 기반 조건부 정규화(Conditional Normalization)가 지배한다. 이 임계점은 학습 전에 계산 가능한 진단량(LPS)으로 예측된다.**

세 가지 기여로 분해된다:

- **C1 (이론)**: 실패 조건의 닫힌형 특성화 — 세 추정량(무정규화 / 인스턴스 정규화 / 조건부 정규화)의 MSE 우열이 뒤집히는 임계 부등식 유도.
- **C2 (검증)**: 합성 DGP에서 이론 곡선과 실험 점의 일치 확인 + 실데이터에서 사전 등록(pre-registered)된 부호 예측 적중.
- **C3 (도구)**: 학습 전 계산 가능한 진단 지표 **LPS(Level Predictability Score)**와 정규화 선택 결정 규칙.

---

## 1. 연구 배경

### 1.1 문헌 흐름: "아키텍처가 아니라 성분이 성능을 만든다"

- **DLinear (Zeng et al., AAAI 2023)**: 단층 선형 모델이 9개 LTSF 벤치마크에서 Transformer 계열을 대부분 큰 격차로 능가. 다중 단계 직접예측(DMS)에서 아키텍처 SOTA의 의미에 대한 회의 공식화.
- **RLinear (Li et al., 2023, arXiv:2305.10721)**: (1) 선형 매핑이 LTSF 성능의 핵심, (2) **RevIN과 채널 독립(CI)이 성능 향상의 결정적 요인**, (3) 선형 매핑은 lookback ≥ 주기일 때 주기 성분을 정확히 표현. RevIN을 붙이면 선형 레이어 하나가 PatchTST를 능가하고, 무작위 초기화된 미학습 feature extractor조차 경쟁력 있음.
- **Toner & Darlow (ICML 2024, arXiv:2403.14587)**: 정규화 붙은 인기 선형 변형들이 비제약 선형회귀와 함수적으로 동치임을 증명. MSE 손실 하 닫힌형 해 존재, 닫힌형 OLS가 72% 설정에서 SGD 학습을 능가.
- **PatchTST ablation (Nie et al., ICLR 2023)**: patching + CI 제거 시 MSE 최대 58% 악화. "SOTA"의 실질 기여는 attention이 아니라 토큰화·채널 처리.
- **Position papers (2025–2026)**: "There are no Champions in LTSF"(3,500+ 네트워크, 일관된 승자 없음), "Current Benchmarking Hinders Real Progress"(설계 차원 미통제가 오도된 비교 양산), TFB(VLDB 2024; drop-last 트릭, 정규화·분할 선택만으로 순위 변동), TSGym(설계 선택 단위 분해 평가).

**요약**: "성분 > 아키텍처" 총론은 이미 완결. 남은 각론은 **성분의 적용 경계(failure conditions)**.

### 1.2 정규화 계열의 현재 지형 — 전부 내생(endogenous) 추정

| 방법 | 통계량 추정 방식 | 한계 |
|---|---|---|
| RevIN (ICLR 2022) | lookback 윈도우의 평균·분산, 학습 가능한 affine | 윈도우 통계를 지평 전체에 외삽. 국소 통계 변화 자체가 신호인 데이터에서 정보 파괴 (후속 문헌에서 지적됨). 이상치에 파국적 실패 사례 보고(MSE +683%). |
| Dish-TS (AAAI 2023) | 과거값으로 미래 분포 계수 학습 | 내생 |
| SAN (NeurIPS 2023) | slice 단위 통계를 과거값으로 예측 | 내생 |
| FAN (NeurIPS 2024) | 주파수 성분으로 추세+계절 비정상성 처리 | 내생 — RevIN이 기본 추세만 다루고 계절 패턴을 못 다룬다는 한계를 확장했으나 여전히 과거 관측값만 사용 |

### 1.3 빈 공간 (신규성 주장)

기존 적응형 정규화는 모두 **미래 통계량을 과거 관측값으로부터 내생적으로 추정**한다. 그러나 에너지 도메인의 핵심 시계열은 수준이 **외생 공변량으로 결정**된다: 풍력 = 설비용량 × 기상 레짐(NWP), 냉방부하 = CDD 반응, 태양광 = clear-sky 곱 구조. 이때 인스턴스 통계 제거는 **복원 가능한 신호를 파괴**한다.

- "언제·얼마나 손해인가"의 **조건부 정식화**가 없다.
- **외생 조건부 정규화 vs 내생 정규화**의 이론적 경계가 없다.
- 학습 전 **어느 쪽을 쓸지 결정하는 진단량**이 없다.

프레이밍 원칙: **RevIN 비판이 아니라 적용 경계의 특성화(characterization).** RevIN 원논문의 분포 이동 동기를 존중하며, "그 가정이 성립하는 조건"을 명시한다.

---

## 2. 이론 (기여 C1)

### 2.1 생성 모형

$$y_t = m(x_t) + \sigma(x_t)\, z_t$$

- $x_t \in \mathbb{R}^d$: 외생 공변량 (NWP 변수, 설비용량, 기온/CDD, 달력 변수)
- $m(\cdot)$: 공변량이 결정하는 수준(level) 함수
- $\sigma(\cdot)$: 조건부 스케일
- $z_t$: 준정상(quasi-stationary) 형태(shape) 성분 — 주기 + 약한 AR 구조

수준의 외생 예측가능성을 다음으로 정의:

$$\lambda \equiv R^2_{\text{level}} = 1 - \frac{\mathbb{E}\,[\,\mathrm{Var}(\bar{y}_w \mid x)\,]}{\mathrm{Var}(\bar{y}_w)}, \quad \bar{y}_w = \text{윈도우 평균}$$

### 2.2 명제 1 — 표현력 손실 (Representational Deficiency)

**주장**: MSE 손실 하의 선형 예측기에서, RevIN을 붙인 모델의 함수 클래스는 (Toner & Darlow의 제약 회귀 동치 결과에 의해) *윈도우 평균과 공변량의 상호작용을 사용하는 해*를 배제한다. 최적 예측자가 $\bar{y}_w \times x$ 상호작용에 의존하는 DGP에서 RevIN 모델은 닫힌형으로 계산 가능한 비가역적 근사 오차를 갖는다.

**증명 전략**: Toner & Darlow의 augmented-feature 재매개화를 그대로 사용. RevIN = "입력에서 $\bar{y}_w$ 제거 + 출력에 $\bar{y}_w$ 복원"이라는 아핀 제약으로 표현 → 제약 하 OLS와 비제약 OLS의 MSE 차이를 사영(projection) 논증으로 계산. **난이도: 낮음 (기존 도구 조립).**

### 2.3 명제 2 — 교차 조건 (Crossover Inequality) ★ 논문의 핵심 수식

**설정**: 선형-가우시안 가정. 세 추정량:
1. **RAW**: 전역 z-score만 적용
2. **IN**: 인스턴스 정규화 (윈도우 통계 $\hat\mu_w, \hat\sigma_w$ 사용)
3. **CN**: 조건부 정규화 — 1단계 회귀 $\hat m(x), \hat\sigma(x)$로 정규화 (오라클/추정 두 버전)

**주장 (도출할 형태)**:

$$\text{MSE}_{IN} < \text{MSE}_{RAW} \iff \underbrace{\mathrm{Var}(\Delta\mu_{\text{train}\to\text{test}})}_{\text{수준 드리프트}} > \underbrace{\frac{\sigma_z^2}{w} + \text{(외삽 편향)}}_{\text{윈도우 통계 추정 노이즈}}$$

$$\text{MSE}_{CN} < \text{MSE}_{IN} \iff \lambda > \lambda^*(w, h, \sigma_z, \text{1단계 추정 오차})$$

즉 $\lambda$–$\lambda^*$ 평면에서 세 방식의 지배 영역이 나뉜다. 이 그림이 **Figure 1 (개념도)** 이 된다.

### 2.4 명제 3 — 지평 상호작용 (Horizon Interaction)

**주장**: IN은 lookback 윈도우 통계를 지평 $h$ 전체에 상수로 외삽하므로, 지평 내 수준 변화(램프, 폭염 개시, 용량 증설)의 분산이 $h$에 대해 증가하는 DGP에서 $\text{MSE}_{IN}(h) - \text{MSE}_{CN}(h)$는 $h$에 단조 증가. → **"왜 다중 단계 예측에서 특히 문제인가"** 를 답한다.

### 2.5 증명 도구와 완결 기준

- 도구: OLS 사영 논증, 조건부 분산 분해, 국소 수준(local-level) + 외생 추세 상태공간 모형.
- 완결 기준: 명제 2의 임계 부등식이 **수치적으로 검증 가능한 닫힌형**으로 정리될 것. 유도 전 과정은 부록행, 본문에는 임계 부등식과 지배 영역 그림만.

---

## 3. 진단 지표 LPS (기여 C3)

### 3.1 정의

$$\text{LPS} = R^2_{\text{oos}}\left(\bar{y}_w \sim g(x_w)\right)$$

- $\bar{y}_w$: 학습 구간을 예측 태스크와 동일한 윈도우로 슬라이딩했을 때 각 윈도우의 타깃 평균
- $g$: 공변량 → 윈도우 평균 회귀기 (기본: LightGBM, 민감도 분석용: Ridge/GAM)
- $R^2_{\text{oos}}$: **시간순 교차검증**의 out-of-sample R² (누수 금지)

### 3.2 결정 규칙 (논문이 제공하는 실무 도구)

```
LPS ≥ τ  →  Conditional Normalization (공변량 기반)
LPS <  τ  →  Instance Normalization (RevIN 계열)
```

τ는 합성 실험의 λ* 와 실데이터 grid에서 교차 결정. 모든 데이터셋에 대해 이 규칙의 **정규화 선택 적중률 표**를 제시 (Table: LPS vs 실제 최적 정규화).

### 3.3 사전 등록 (Pre-registration)

**G4(실데이터 grid) 실행 전에** 각 데이터셋의 LPS를 계산하고, `paper/predictions.md`에 "RevIN − CondNorm 성능 격차의 부호"를 데이터셋별로 기록·커밋한다. 실험 후 적중 여부를 논문에 그대로 보고. (반증 가능성 확보 — 리뷰 방어의 핵심 장치)

---

## 4. 합성 실험 설계 (기여 C2-1)

### 4.1 DGP 명세

$$x_t = 0.9\,x_{t-1} + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, 1) \quad \text{(외생 공변량, AR(1) + 일·주 계절항 옵션)}$$

$$m_t = \sqrt{\lambda}\; \beta^\top \tilde{x}_t \;+\; \sqrt{1-\lambda}\; u_t, \qquad u_t = u_{t-1} + \eta_t \;\;(\text{random walk drift, } \eta_t \sim \mathcal{N}(0, \sigma_u^2))$$

$$z_t = A\sin(2\pi t / P) + \rho\, z_{t-1} + \nu_t, \qquad y_t = m_t + \sigma_0\, z_t$$

- $\tilde{x}_t$: 표준화된 공변량 (→ 구성상 $R^2_{\text{level}} \approx \lambda$)
- $\lambda$ **sweep**: $\{0.0, 0.1, \dots, 1.0\}$ (11점)
- 주기 $P = 24$, lookback $L \in \{96, 336\}$, horizon $h \in \{24, 96, 336\}$
- 시리즈 길이 20,000 / train:val:test = 6:2:2 시간순 / 시드 10회

### 4.2 비교 대상 (합성 단계)

- 백본: **RLinear 고정** (이론과의 대응을 위해 선형만; 비선형은 실데이터에서)
- 정규화: RAW / IN(RevIN) / CN-oracle (참 $m_t$ 사용) / CN-est (LightGBM 1단계)

### 4.3 성공 기준 — GATE 1 (go/no-go)

- [ ] $\lambda$ 축 위에서 IN vs CN의 MSE 곡선이 **교차**하고, 교차점이 명제 2의 $\lambda^*$ 예측과 ±0.1 이내로 일치
- [ ] $h$가 커질수록 격차 확대 (명제 3 방향성 확인)
- [ ] CN-est가 CN-oracle 대비 성능 저하폭이 1단계 추정 오차 항으로 설명됨

**GATE 1 실패 시**: 이론 재점검 → 그래도 실패하면 논문 각도를 "어떤 조건에서 교차가 흐려지는가"의 경험적 특성화 + LPS 진단 지표 중심으로 전환 (탈출 경로 A).

---

## 5. 실데이터 실험 설계 (기여 C2-2)

### 5.1 데이터셋 — 외생 구동 그룹 vs 표준 벤치마크 그룹

| 그룹 | 데이터셋 | 타깃 | 공변량 | 예상 LPS | 비고 |
|---|---|---|---|---|---|
| 외생 구동 | **Jeju wind** (KPX 집계 발전량, 시간단위) | 발전률(용량 정규화 전 원계열) | 보관된 NWP(리드 매칭), 설비용량 시계열 | 높음 | 사내 데이터. 용량은 시변 — CN의 대표 사례 |
| 외생 구동 | **KPX 계통 수요** (시간단위, 공개) | 전국/제주 수요 | 기온·CDD/HDD, 달력 | 높음 | 공개 재현성 확보 |
| 외생 구동 | **GEFCom2014** load / wind / solar | 각 트랙 타깃 | 대회 제공 기상 | 높음 | 국제 공개 벤치마크 — 재현성 방어 |
| 표준 LTSF | ETTh1, ETTh2 | OT | (달력만) | 낮음 | 문헌 결과 재현: RevIN 우세 예상 |
| 표준 LTSF | Electricity, Weather | 표준 설정 | (달력만) | 낮음~중간 | 〃 |

핵심 설계 의도: **LPS가 낮은 데이터셋에서는 기존 문헌대로 IN이 이기고, 높은 데이터셋에서는 CN이 이긴다** — LPS가 격차의 부호를 예측함을 보인다.

### 5.2 요인 격자 (Factorial Grid)

```
정규화 (5)   : RAW / RevIN / SAN / FAN / CondNorm(제안)
백본   (4)   : RLinear / PatchTST / SegRNN / LightGBM(직접 다중출력 DMS)
데이터셋 (7) : 위 표
horizon (3)  : {24, 96, 336} (데이터 해상도에 맞게 조정)
시드   (5)   : {0..4} (LightGBM은 시드 불필요 → 1회)
```

- SAN·FAN을 **반드시 포함** — "적응형 정규화가 이미 해결했다"는 리뷰 공격의 봉쇄선. 공식 구현 이식 + 단위 테스트로 원 논문 수치 재현 확인.
- CondNorm 구현: 1단계 회귀 $\hat m(x_t)$ (기본 LightGBM; 풍력은 용량 정규화 + 기상 회귀, 부하는 change-point CDD 회귀와 대응) → $\tilde y_t = (y_t - \hat m(x_t)) / \hat\sigma(x_t)$ → 백본 → 역변환. **가역성 단위 테스트 필수.**

### 5.3 프로토콜 (position paper들의 지적 사항 전면 반영)

- **시간순 out-of-time 분할**, 테스트는 계절 주기 1년 이상 커버
- **lookback을 데이터셋×모델별로 밸리데이션 튜닝** ({96, 192, 336, 720}) — lookback 임의 고정이 순위를 뒤집는다는 문헌 지적 방어
- **drop_last=False** (TFB 지적 사항)
- 전처리 파이프라인 단일화 (전 데이터셋 동일 코드 경로), NWP는 **리드 매칭** (아카이브 예보 사용, 관측 재분석 누수 금지)
- 지표: 표준화 스케일 MSE/MAE + 도메인 지표(풍력: 용량 대비 nMAE)

### 5.4 통계 검정

- 쌍별 **Diebold-Mariano** (Harvey 보정, 데이터셋×horizon 단위)
- **Model Confidence Set** (α = 0.10) — "일관된 승자 없음" 문헌 표준에 부합
- 요인별 분산 귀속: (정규화 / 백본 / 데이터셋 / 상호작용)에 대한 ANOVA식 분해 → **"정규화 축이 백본 축보다 몇 배의 분산을 설명하는가"** 수치 제시
- 모든 격차에 시드 기반 신뢰구간

---

## 6. 산출물 계획 — 그림 5장, 표 3개로 고정

| ID | 내용 | 생성 Phase |
|---|---|---|
| Fig 1 | 지배 영역 개념도: (λ, 드리프트 분산) 평면에서 RAW/IN/CN 영역 | G1 |
| Fig 2 | 이론 MSE 곡선 (명제 2 닫힌형) 위에 합성 실험 점 중첩 — 교차점 일치 | G2 |
| Fig 3 | ★ 데이터셋별 LPS(x축) vs "RevIN − CondNorm 성능 격차"(y축) 산점도 + 사전 등록 부호 예측 적중 표시 | G5 |
| Fig 4 | 요인분석 막대 (정규화×백본, 신뢰구간 포함) — "정규화 축 > 백본 축" 시각화 | G5 |
| Fig 5 | horizon별 격차 곡선 (명제 3 실증) | G5 |
| Tab 1 | 메인 결과 grid (데이터셋 × 정규화, 백본 평균, DM 유의성 마크) | G4 |
| Tab 2 | MCS 포함 여부 + 분산 귀속 (%) | G5 |
| Tab 3 | LPS 결정 규칙 적중률 (τ 민감도 포함) | G5 |

## 7. 논문 골격 (IJF 서식 기준, 본문 ~8,000 words)

1. **Introduction** — 성분>아키텍처 문헌 흐름 요약 → 정규화 계열이 전부 내생 → 외생 구동 도메인의 문제 제기 → 기여 C1–C3
2. **Related Work** — LTSF 회의론 / 정규화 계열 (RevIN, Dish-TS, SAN, FAN) / 벤치마킹 비판 / 에너지 도메인 관행 (용량 정규화, clear-sky index, CDD 회귀 = 조건부 정규화의 암묵적 실무 형태였음을 지적)
3. **Theory** — 생성 모형, 명제 1–3 (증명 부록), 임계 부등식 + Fig 1
4. **LPS Diagnostic** — 정의, 결정 규칙, 사전 등록 절차
5. **Synthetic Validation** — Fig 2
6. **Empirical Study** — 데이터·grid·프로토콜·Tab 1·Fig 3–5
7. **Discussion** — 적용 경계, 한계 (내생·외생 혼합 사례, 1단계 추정 오차 전파), VPP/계통 운영 시사점
8. **Conclusion**

리뷰 방어 매핑: "RevIN이 만능이라 주장한 적 없다" → §2에서 원논문 동기 존중 + 경계 특성화 포지셔닝 / "적응형 정규화가 이미 해결" → SAN·FAN 포함 실증 / "피처 엔지니어링일 뿐" → 명제 2 + 사전 등록 적중 / "에너지 편중" → GEFCom + 표준 LTSF 재현.

## 8. 리스크와 킬 크라이테리아

| 리스크 | 발동 조건 | 대응 |
|---|---|---|
| 신규성 선점 | G0 문헌 스윕에서 동일 정식화 발견 | 각도를 LPS 진단 지표 + 도메인 검증 중심으로 전환 |
| GATE 1 실패 | 합성에서 교차 미발생 / λ* 불일치 | 이론 재점검 → 탈출 경로 A (경험적 특성화 논문) |
| GATE 2 실패 | 실데이터에서 사전 등록 부호 예측 적중률 < 70% | "교차가 흐려지는 조건" 분석 추가, 주장 강도 하향 |
| SAN/FAN 재현 실패 | 원 논문 수치 ±10% 미달 | 공식 레포 고정 커밋 사용, 재현 로그를 부록에 공개 |

## 9. 레포 구조 · 환경

```
norm-boundary/
├── the environment contract                # 행동 불변식 (G0에서 생성: 규약 요약, GPU/MLflow 규칙, 데이터 계약)
├── RESEARCH_PLAN.md         # 본 문서
├── pyproject.toml           # uv 관리
├── configs/                 # yaml: dataset / norm / backbone / grid
├── src/
│   ├── data/                # 로더 + curated/ Parquet 계약 (스키마 고정, 리드 매칭 검증 포함)
│   ├── norms/               # revin.py, san.py, fan.py, condnorm.py (+ 가역성 단위 테스트)
│   ├── models/              # rlinear.py, patchtst.py, segrnn.py, lgbm_dms.py
│   ├── theory/              # closed_form.py (명제 1–3 수치 검증), lps.py
│   ├── synth/               # dgp.py (λ sweep)
│   ├── eval/                # dm_test.py, mcs.py, attribution.py, metrics.py
│   └── train.py             # 단일 진입점: config → run → MLflow 로깅
├── experiments/             # grid 정의 yaml + 실행 스크립트
├── tests/                   # pytest: 가역성, 누수, 재현성
├── curated/                 # Parquet (gitignore)
├── results/                 # csv 요약 (MLflow가 원본)
└── paper/
    ├── predictions.md       # ★ 사전 등록 (G4 실행 전 커밋)
    ├── figures/  tables/
    └── main.tex
```

**환경 규약** (ai-research 서버 기준):
- `uv`로 의존성 관리, Python ≥ 3.11, **PyTorch ≥ 2.7 (Blackwell sm_120)**
- **GPU1 사용 고정** (`CUDA_VISIBLE_DEVICES=1`; GPU0은 Xorg 예약)
- MLflow experiment: `ini/norm-boundary`, run 이름 규약 `{dataset}_{norm}_{backbone}_{h}_{seed}`
- 재현성: 시드 고정, `torch.use_deterministic_algorithms(True)` 가능 범위 내 적용, config·커밋 해시를 MLflow 태그로 기록

---

## 10. Claude Code 실행 가이드 — Phase Goals

> 각 Goal은 새 세션에서 단독 실행 가능하도록 작성됨. 실행 순서: G0 → G1 → G2 → **[GATE 1]** → G3 → (사전 등록) → G4 → **[GATE 2]** → G5 → G6.
> 공통 규칙: 모든 Phase는 종료 시 (1) AC 체크리스트 자체 평가, (2) `results/` 요약 갱신, (3) 다음 Phase 착수에 필요한 정보를 `HANDOFF.md`에 기록.

### G0 — 스캐폴딩 + 신규성 스윕 (킬 크라이테리아 확인)
**Goal**: 레포 구조·환경·데이터 계약을 완성하고 스모크 테스트를 통과시킨다.
- §9 구조대로 스캐폴딩, uv 환경 구성, GPU1 스모크 테스트 (작은 텐서 학습 1 step)
- ETTh1 로더 + RLinear + RevIN로 end-to-end 1회 학습이 도는 최소 파이프라인
- 문헌 스윕 체크리스트 작성: "conditional/covariate-informed normalization forecasting" 계열 검색 결과를 `docs/novelty_sweep.md`에 기록 (동일 정식화 발견 시 §8 대응 발동)
- **AC**: `pytest` 통과 / MLflow에 스모크 run 기록 / the environment contract 생성 완료

### G1 — 이론 수치화
**Goal**: 명제 1–3을 `src/theory/closed_form.py`로 구현하고 수치 검증한다.
- 선형-가우시안 설정에서 RAW/IN/CN의 닫힌형 MSE 구현
- 몬테카를로 시뮬레이션과 닫힌형의 일치 확인 (상대오차 < 1%)
- Fig 1 (지배 영역), 이론 곡선 산출 → `paper/figures/`
- **AC**: 닫힌형 vs MC 일치 / λ*(w, h, σ) 함수가 단위 테스트로 고정됨

### G2 — 합성 실험 (GATE 1)
**Goal**: §4의 DGP·sweep을 실행하고 이론 예측과 대조한다.
- `src/synth/dgp.py` 구현 (λ가 R²_level과 일치하는지 자체 검증 포함)
- λ 11점 × h 3 × L 2 × 시드 10 grid 실행 (RLinear, 정규화 4종)
- Fig 2 생성, GATE 1 체크리스트 (§4.3) 평가 → `HANDOFF.md`에 go/no-go 기록
- **AC**: §4.3 세 항목 판정 완료 (통과/실패 모두 근거 수치와 함께 문서화)

### G3 — 정규화·백본 구현부
**Goal**: 실데이터 grid에 필요한 모든 구성요소를 검증된 상태로 준비한다.
- SAN·FAN 공식 구현 이식 (커밋 해시 고정) + 원 논문 대표 수치 재현 테스트
- CondNorm 구현 (1단계 LightGBM + 가역 변환) + 가역성·누수 단위 테스트
- PatchTST/SegRNN/LightGBM-DMS 구현, ETTh1에서 문헌 범위 내 성능 확인
- 데이터셋 7종 로더 + LPS 계산기 (`src/theory/lps.py`, 시간순 CV)
- **AC**: 전 구성요소 pytest 통과 / 각 데이터셋 LPS 값이 `results/lps.csv`에 산출

### G4 — 사전 등록 → 실데이터 grid (GATE 2)
**Goal**: 예측을 먼저 커밋하고, 요인 격자 전체를 실행한다.
- **grid 실행 전**: `paper/predictions.md`에 데이터셋별 (LPS, 예측 부호) 기록 후 git commit
- §5.2 grid 실행 (정규화 5 × 백본 4 × 데이터셋 7 × h 3 × 시드 5; 우선순위: 외생 구동 그룹 먼저)
- DM 검정·MCS 실행, Tab 1 초안 생성
- **AC**: 전 run MLflow 기록 / 사전 등록 적중률 산출 → GATE 2 판정 (§8)

### G5 — 귀속 분석 + 진단 지표 검증
**Goal**: 결과를 논문 주장 구조로 변환한다.
- 분산 귀속 (정규화/백본/데이터셋/상호작용), Fig 3–5, Tab 2–3 생성
- LPS 결정 규칙의 τ 결정 및 민감도 분석
- **AC**: Fig 3에서 LPS–격차 관계가 유의 (Spearman ρ, p-value 보고) / 그림·표 전체가 스크립트 재생성 가능 (`make figures`)

### G6 — 논문 초고
**Goal**: §7 골격대로 LaTeX 초고를 완성한다.
- `paper/main.tex` — 본문 서술, 증명 부록, 재현성 성명 (코드·시드·환경)
- 리뷰 방어 문단 (§7 매핑) 명시적 포함, 투고 체크리스트 (IJF 서식) 작성
- **AC**: 컴파일되는 초고 + 그림·표 전부 삽입 / TODO 마커 20개 이하

---

## 부록 A. 복붙용 Goal 프롬프트

```text
[G0] RESEARCH_PLAN.md를 읽고 §9 레포 구조를 스캐폴딩하라. uv로 환경을 만들고
(PyTorch>=2.7, CUDA_VISIBLE_DEVICES=1), ETTh1 + RLinear + RevIN 최소 파이프라인이
end-to-end로 학습되는 스모크 테스트를 pytest로 작성·통과시켜라. the environment contract에 환경
규약과 데이터 계약을 요약하고, docs/novelty_sweep.md 체크리스트를 만들어라.
완료 후 G0 AC를 자체 평가하고 HANDOFF.md를 작성하라.
```

```text
[G1] RESEARCH_PLAN.md §2를 읽고 명제 1–3의 닫힌형 MSE를 src/theory/closed_form.py로
구현하라. 몬테카를로 대조(상대오차<1%)를 pytest로 고정하고, λ*(w,h,σ) 임계 함수와
Fig 1 지배 영역 그림을 paper/figures/에 생성하라. G1 AC 자체 평가 후 HANDOFF.md 갱신.
```

```text
[G2] RESEARCH_PLAN.md §4의 DGP를 src/synth/dgp.py로 구현하고 (λ≈R²_level 자체 검증
포함), λ 11점 × h{24,96,336} × L{96,336} × 시드 10의 sweep을 RLinear + {RAW, RevIN,
CN-oracle, CN-est}로 실행하라. 이론 곡선 위에 실험 점을 중첩한 Fig 2를 만들고
§4.3 GATE 1 세 항목을 판정, 근거 수치와 함께 HANDOFF.md에 go/no-go를 기록하라.
```

```text
[G3] SAN·FAN 공식 구현을 고정 커밋으로 이식해 원 논문 수치 재현 테스트를 통과시키고,
CondNorm(1단계 LightGBM + 가역 변환)과 PatchTST/SegRNN/LightGBM-DMS를 구현하라.
데이터셋 7종 로더와 LPS 계산기(시간순 CV)를 완성해 results/lps.csv를 산출하라.
가역성·누수 테스트 포함 전체 pytest 통과가 AC다.
```

```text
[G4] 먼저 results/lps.csv를 바탕으로 paper/predictions.md에 데이터셋별 RevIN−CondNorm
격차의 부호 예측을 기록하고 git commit하라. 그 다음 §5.2 요인 격자를 외생 구동
그룹부터 실행하고 (MLflow: ini/norm-boundary, run 규약 준수), DM 검정과 MCS를 돌려
Tab 1 초안을 생성하라. 사전 등록 적중률을 산출해 GATE 2를 판정하라.
```

```text
[G5] grid 결과에 분산 귀속(정규화/백본/데이터셋/상호작용)을 수행하고 Fig 3–5,
Tab 2–3을 생성하라. LPS 결정 규칙의 τ를 결정하고 민감도 분석을 포함하라.
모든 그림·표가 make figures로 재생성되는지 확인하는 것이 AC다.
```

```text
[G6] §7 골격대로 paper/main.tex 초고를 작성하라. 증명은 부록, 본문에는 임계 부등식과
Fig 1–5, Tab 1–3만. 리뷰 방어 문단 4종을 명시적으로 포함하고 IJF 투고 체크리스트를
작성하라. 컴파일 확인 후 TODO 마커를 20개 이하로 정리하라.
```

## 부록 B. 핵심 참고문헌 (초고 인용 목록의 시드)

- Kim et al., *RevIN*, ICLR 2022 — 인스턴스 정규화 원류, 분포 이동 동기
- Zeng et al., *Are Transformers Effective for TSF?* (DLinear), AAAI 2023
- Li et al., *Revisiting LTSF: An Investigation on Linear Mapping* (RLinear), arXiv:2305.10721
- Toner & Darlow, *An Analysis of Linear TSF Models*, ICML 2024 — 제약 회귀 동치·닫힌형
- Nie et al., *PatchTST*, ICLR 2023 — patching/CI ablation
- Fan et al., *Dish-TS*, AAAI 2023 / Liu et al., *SAN*, NeurIPS 2023 / Ye et al., *FAN*, NeurIPS 2024 — 내생 적응형 정규화 계열
- Qiu et al., *TFB*, VLDB 2024 — 평가 파이프라인 편향 (drop-last 등)
- Brigato et al., *There are no Champions in LTSF*, TMLR 2026 — 일관된 승자 부재
- *Position: Current Benchmarking Hinders Real Progress in DL for TSF*, arXiv:2512.22702 — 설계 차원 통제 요구
- Kim et al. / Hong et al., *GEFCom2014* 개요 논문 — 공개 에너지 예측 벤치마크
