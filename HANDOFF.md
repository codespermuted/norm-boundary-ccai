# HANDOFF — 현재 상태와 다음 단계

- **최종 갱신**: 2026-07-13
- **완료 Phase**: G0 ✅ → G1 ✅
- **다음 Phase**: **G2 — 합성 실험 (GATE 1)** (RESEARCH_PLAN.md §4, 부록 A의 [G2] 프롬프트)

## G1 AC 자체 평가

| AC | 판정 | 근거 |
|---|---|---|
| 닫힌형 vs MC 일치 (상대오차 <1%) | ✅ | `tests/test_theory.py` — 6개 파라미터 조합 × 4개 추정량, n=2M MC, 전부 <1% |
| λ*(w,h,σ) 함수가 단위 테스트로 고정 | ✅ | 교차 항등식·h 단조성·드리프트 단조성·수치해 일치 테스트 4종 |
| Fig 1 (지배 영역) 산출 | ✅ | `paper/figures/fig1_dominance.{pdf,png}` — h∈{24,96,336} 3패널, IN 영역이 h에 따라 후퇴하는 명제 3 패턴 시각 확인 |
| 전체 pytest | ✅ | 25 passed |

## G1 산출물

- `docs/theory_g1.md` — **양식화 모델 M1 명세·유도·가정(A1–A4)·이론↔G2 DGP↔실데이터 대응표** (실험 스크립트가 따라야 할 계약 문서)
- `src/theory/closed_form.py` — MSE 닫힌형, λ* (닫힌형 + 수치해), 지배 영역, 명제 1 gap, `effective_window_noise`(자기상관 형태 보정)
- `src/theory/simulate.py` — 모델 충실 MC + 명제 1 OLS MC
- `src/theory/figstyle.py` — **논문 전체 그림의 엔티티→색 고정 매핑** (IN=blue, CN=aqua, RAW=yellow; Fig 2–5도 이것을 사용할 것)
- 기준 λ* 값: h=24 → 0.923 / h=96 → 0.664 / h=336 → λ*<0 (σ_Δ=0 기준)

## 이론–실험 정합 규약 (사용자 지시, the environment contract에도 수록)

- G2 백본은 **RLinear 단층 선형만** (이론이 선형-가우시안이므로). 용량은 arm 간 동일, config에 명시.
- G2 DGP → M1 매핑 시 주의 (대응표 참조):
  1. DGP의 u는 √(1−λ)로 스케일 → M1의 hσ_u²에는 **(1−λ)·h·σ_u,dgp²** 대입
  2. AR(1) 공변량의 지평 내 변화 → s_x²(h)를 수치 산출해 대입 (λ 의존 → `lambda_star_numeric` 사용)
  3. 형태 z가 자기상관(AR+계절)이므로 윈도우 노이즈는 `effective_window_noise(σ_z, w, ρ1)` 사용
- 괴리 후보 1순위 = 가정 A3 (정규화가 형태 학습에 주는 2차 효과) — G2에서 경험적으로 점검하고 `docs/theory_g1.md` 괴리 항목에 기록

## G2 착수 정보

1. `src/synth/dgp.py`: 계획서 §4.1 DGP 구현 + **λ ≈ R²_level 자체 검증** 루틴 포함
2. sweep: λ 11점 × h{24,96,336} × L{96,336} × 시드 10 — RLinear + {RAW, RevIN, CN-oracle, CN-est}
3. CN-est 1단계는 LightGBM (의존성 추가 필요: `uv add lightgbm`), CN-oracle은 참 m_t 사용
4. Fig 2: 이론 곡선(λ 함수인 MSE, 위 매핑 적용) 위에 실험 점 중첩 — `figstyle.py` 색 매핑 사용
5. GATE 1 판정 (§4.3): 교차 발생 + 교차점 λ*와 이론 예측 ±0.1 일치 / h 격차 확대 / CN-est 저하가 1단계 오차로 설명 — go/no-go를 근거 수치와 함께 이 파일에 기록
6. MLflow experiment `ini/norm-boundary`는 sqlite(`sqlite:///mlflow.db`)
