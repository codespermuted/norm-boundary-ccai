# 실험 설계 감사 — 백본 × 정규화 × 공변량 대응표 (2026-07-15)

> 계기: "iTransformer는 원래 공변량/정규화가 내장 아닌가, 지금 비교가 정당한가"라는 사용자 질문.
> 결론: 정당하며, 그 근거가 되는 대응 관계를 아래에 고정한다. 논문 §5(프로토콜)와 부록에 그대로 수록할 것.

## 1. 원칙

1. **정규화 분리 원칙**: 모든 백본은 내장 인스턴스 정규화를 제거(`use_norm=False` 등)하고,
   정규화는 5개 arm(RAW/RevIN/SAN/FAN/CondNorm)으로만 주입한다. "published 형태"는 아래 표의
   대응 arm으로 재현된다. (성분 통제 — TFB·TSGym 방법론과 동일)
2. **공변량 대칭 원칙**: 같은 비교 블록 안의 모든 arm은 **동일한 공변량 정보 집합**을 받는다.
   차이는 오직 "그 정보를 어디에 쓰는가"(입력 vs 정규화 복원 경로)뿐.
3. **비교 블록 자기완결 원칙**: 서로 다른 스크립트(에폭 등 미세 차이)의 행을 한 표에서 직접
   비교하지 않는다. (본 grid / covfair-full / SOTA-MS는 각각 내부 균일)

## 2. 백본별 대응표

| 백본 | published 기본 정규화 | 우리 처리 | published ≈ 우리 arm | 공변량 경로 |
|---|---|---|---|---|
| RLinear | RevIN (이름의 R) | 외부 arm | revin | (본 grid) 없음 / (covfair) linmix 피처 |
| PatchTST | RevIN 내장 기본 | 제거 후 외부 arm | revin | 〃 (covfair: PatchTSTCov 최소 확장) |
| SegRNN | 인스턴스 정규화(윈도우 마지막 값 차감) 내장 | 제거 후 외부 arm | revin 근사 (평균 vs 마지막값 차이는 부록에 명시) | 〃 (covfair: SegRNNCov) |
| LGBM-DMS | 관행상 다양 | winz(=RevIN 등가 목적함수) arm | — | (covfair) 피처 |
| iTransformer | **use_norm=True 내장 (NST=무affine RevIN)** | use_norm=False + 외부 arm | **revin** | `_ms`: 공변량 variate 토큰 (공식 지원 사용법) |
| TimeXer | use_norm=True 내장 | 〃 | **revin** | `_ms`: exogenous cross-attention (설계 목적) |

## 3. 공변량 × RevIN 공정성 쟁점과 판단

**쟁점**: published use_norm은 (MS 설정에서) **공변량 열까지 윈도우 정규화**한다.
우리 MS 어댑터는 정규화 arm을 **타깃 채널에만** 적용하고 공변량은 전역 z-score로 준다.

**판단 — 현행이 RevIN에게 유리한(보수적) 선택**: 공변량을 윈도우 정규화하면 공변량이 담은
수준 정보(NWP 풍속의 절대값 등)가 파괴된다 — 그건 우리가 비판하는 메커니즘을 baseline에
강제로 주입해 RevIN을 약하게 만드는 짓이다. 타깃만 정규화하는 현행은 RevIN arm이 공변량
수준 정보를 온전히 활용하게 하므로, CondNorm 우위가 나온다면 그것은 보수적 조건에서의
우위다. (검증 arm으로 `revin_all`(공변량까지 윈도우 정규화 = published 기본 그대로)을 G5
ablation에 추가해 "실무 기본값이 얼마나 손해인지"를 별도 정량화한다 — 예정)

**CondNorm의 공변량 이중 사용(입력+복원)은 불공정한가?** 아니다 — 정보 집합은 모든 arm에서
동일하고(같은 cov 입력), 복원 경로 활용은 방법 그 자체다. 비교 대상이 "같은 정보, 다른
주입 방식"이라는 연구 질문과 정확히 일치한다.

## 4. 비교 블록 구조 (논문 표 구성)

| 블록 | 백본 | 공변량 | 목적 |
|---|---|---|---|
| A. 본 grid (사전등록, GATE 2) | RLinear/PatchTST/SegRNN/LGBM | 백본 입력 없음, CondNorm 복원만 | 사전등록 부호 검증 |
| B. covfair-full | linmix/mlpmix/PatchTSTCov/SegRNNCov/LGBM+cov | 전 arm 과거+미래 피처 | 정보 대등 하 메커니즘 검증 |
| C. SOTA-MS | iTransformer-MS/TimeXer-MS | 전 arm 공식 경로 입력(과거) | SOTA 공변량 모델 위 정규화 선택 |
| D. SOTA-endo | iTransformer/TimeXer-M | 없음 (다변량 상호) | 블록 A의 백본 강건성 확장 |

주: B의 미래 공변량 vs C의 과거-만 차이는 각 모델의 published 형태를 따른 것 (TimeXer/iTransformer는
미래 공변량 입력 경로가 없음). C에서 CondNorm arm은 복원에 미래 공변량(리드 매칭 예보)을 사용하며
이는 방법 정의다 — C 블록 내 타 arm과의 입력 정보는 동일(과거 cov).

## 5. 알려진 잔여 비대칭 (논문 한계 절에 명시)

1. SegRNN published는 last-value 차감 — 우리 revin arm(평균/표준편차)과 미세하게 다름.
2. covfair B의 PatchTSTCov/SegRNNCov는 표준 레시피 부재로 우리가 설계한 최소 확장.
3. gefcom_load 기온은 ex-post 관행 (전 arm 동일).
4. 블록별 에폭 상한 차이 (A: 12, B: 15) — 블록 간 직접 비교 금지로 통제.
