# G0 신규성 스윕 — 킬 크라이테리아 판정 (§8)

- **수행일**: 2026-07-13
- **판정: 킬 크라이테리아 비발동 (연구 계속 진행)** — 아래 인접 문헌 중 어느 것도 본 연구의 세 기여(C1 임계 부등식 / C2 외생 조건부 정규화 vs 내생 정규화 경계 / C3 학습 전 진단량 LPS)와 동일한 정식화를 갖지 않음.

## 검색 쿼리 (수행한 것)

- [x] covariate-informed conditional normalization time series forecasting exogenous variables RevIN
- [x] instance normalization time series forecasting failure / when hurts / exogenous level shift (2025–2026)
- [x] normalization selection diagnostic — "which normalization" decide before training
- [x] exogenous covariate normalization wind power load forecasting weather

## 인접 문헌과 차별점

| 논문 | 내용 | 본 연구와의 차이 |
|---|---|---|
| **On the Role of Reversible Instance Normalization** (arXiv:2603.11869, 2026-03) | RevIN 구성요소의 ablation — 일부 요소가 중복이거나 역효과라는 실증 비판. temporal/spatial/conditional shift 3분류 제시 | 실증 ablation 중심. **임계 조건의 닫힌형 유도 ✗, 외생 공변량 기반 정규화 ✗, 학습 전 진단량 ✗**. 오히려 §1(문헌 흐름)에 인용할 우군 — 관련연구에 반드시 인용 |
| **Noise or Signal? Deconstructing Contradictions… Reversible Normalization** (arXiv:2510.04667) | 가역 정규화의 이론적 모순 4가지 지적 + 적응형 처방(A-IN)이 오히려 시스템적으로 실패함을 보고. "diagnostics-driven analysis" 필요성 주장 | 진단 필요성을 *주장*만 하고 **구체적 진단량·결정 규칙 없음**. 내생 통계 범위 내 논의. LPS의 동기 문단에 인용 가치 높음 |
| **Inner-Instance Normalization for TSF** (arXiv:2510.08657) | 인스턴스 내부(point-level) 분포 이동 처리 (LD/LCD) | 여전히 **내생** (과거 관측값 기반). 공변량 미사용 |
| **IN-Flow** (arXiv:2401.16777) | 정규화를 flow로 학습해 비정상성 처리 | 내생 + 확률적 변환. 임계 조건·진단 없음 |
| **CITRAS** (arXiv:2503.24007) | 공변량 활용 Transformer (아키텍처 통합) | 공변량을 **모델 입력**으로 쓰는 계열. 정규화 통계량으로 쓰지 않으며 이론 없음 |
| **Adaptive wind data normalization** (Patil et al., Wind Engineering 2022) | 풍속 데이터 적응 정규화 실무 연구 | 내생 적응. 도메인 관행 근거로 인용 가능 |
| **Conditional Normalizing Flow for wind power** (arXiv:2206.02433) | 조건부 "normalizing flow" (확률 예측) | 이름만 유사 — 확률밀도 모델링이며 본 연구의 Non-goal(확률 예측) 영역 |

## 결론 및 프레이밍 조정

1. **빈 공간 유지**: "외생 공변량으로 정규화 통계량을 추정하는 조건부 정규화 vs 내생 인스턴스 정규화의 이론적 경계 + 학습 전 결정 규칙"은 여전히 비어 있음.
2. **경계 강화 필요**: 2603.11869(RevIN 회의론)와 2510.04667(적응형 처방의 실패)의 등장은 "RevIN 재검토" 흐름이 뜨거워지고 있음을 의미 — **선점 리스크가 시간에 민감**. G1–G2를 신속히 진행할 것.
3. **Related Work 추가 인용**: 위 두 논문을 §7 골격 2절(Related Work)의 정규화 계열 문단에 추가. "적응형 정규화가 이미 해결했다" 방어선에 2510.04667의 A-IN 실패 사례가 오히려 유리한 증거.
4. **재스윕 시점**: G4 착수 전 1회 재검색 (동일 쿼리 + "level predictability", "normalization crossover" 추가).

## 근거 링크

- https://arxiv.org/abs/2603.11869 — On the Role of Reversible Instance Normalization
- https://arxiv.org/abs/2510.04667 — Noise or Signal? Deconstructing Contradictions in Reversible Normalization
- https://arxiv.org/abs/2510.08657 — Inner-Instance Normalization for Time Series Forecasting
- https://arxiv.org/abs/2401.16777 — IN-Flow
- https://arxiv.org/abs/2503.24007 — CITRAS
- https://journals.sagepub.com/doi/10.1177/0309524X221093908 — Adaptive wind data normalization
- https://arxiv.org/abs/2206.02433 — Conditional normalizing flow wind power
