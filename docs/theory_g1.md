# G1 이론 노트 — 양식화 모델 M1과 닫힌형 MSE

> 이 문서는 `src/theory/closed_form.py`의 수학적 명세이자, G2 합성 실험·G4 실데이터 실험이 따라야 할
> **이론–실험 대응의 기준 문서**다. 코드와 이 문서가 어긋나면 코드가 틀린 것이다.

## 1. 양식화 모델 M1 (instance-level, 선형-가우시안)

예측 인스턴스 하나 = (lookback 윈도우 w스텝, 지평 h스텝 뒤 타깃). 수준(level) 복원 오차만 비교한다
— 형태(shape) 성분의 예측 가능 부분은 세 방법이 동일하게 처리하므로 공통 잔차 $\sigma_\varepsilon^2$로 흡수 (§4 가정 A3).

인스턴스별 확률 변수 (모두 독립 가우시안):

| 기호 | 분포 | 의미 |
|---|---|---|
| $g_T$ | $\mathcal{N}(0, \lambda V)$ | 타깃 시점 수준의 **공변량 설명 성분** (CN은 NWP로 이것을 관측) |
| $v_T$ | $\mathcal{N}(0, (1-\lambda)V)$ | 타깃 시점 수준의 비설명 성분 |
| $\Delta$ | $\mathcal{N}(0, \sigma_\Delta^2)$ | **공변량 직교** train→test 수준 이동 (윈도우·타깃에 공통) |
| $\delta_x$ | $\mathcal{N}(0, s_x^2(h))$ | 지평 내 공변량 구동 수준 변화 (램프): 윈도우 공변량 수준 $g_w = g_T - \delta_x$ |
| $\delta_u$ | $\mathcal{N}(0, h\sigma_u^2)$ | 지평 내 비설명 드리프트 (random walk): $v_w = v_T - \delta_u$ |
| $\bar\varepsilon$ | $\mathcal{N}(0, \sigma_z^2/w)$ | 윈도우 평균의 형태 노이즈 |
| $e$ | $\mathcal{N}(0, \sigma_{est}^2)$ | CN 1단계 추정 오차: $\hat m = g_T + e$ |
| $\zeta$ | $\mathcal{N}(0, \sigma_\varepsilon^2)$ | 공통 형태 잔차 |

파생량: 타깃 수준 $L = \Delta + g_T + v_T$, 윈도우 수준 $M = \Delta + g_w + v_w$, 윈도우 평균
$\bar y = M + \bar\varepsilon$, 타깃 $y = L + \zeta$.

$V = \mathrm{Var}(g_T + v_T)$는 수준의 총 분산, $\lambda$는 계획서 §2.1의 $R^2_{\text{level}}$과 일치.

## 2. 세 추정량과 닫힌형 MSE

수준 복원 규칙 (train에서 적합된 것은 train 분포 기준, WLOG train 전역 평균 = 0):

| 방법 | 복원 규칙 $\hat y$ | 오차 $y - \hat y$ | **MSE (닫힌형)** |
|---|---|---|---|
| RAW (전역 z-score만) | $0$ | $L + \zeta$ | $V + \sigma_\Delta^2 + \sigma_\varepsilon^2$ |
| IN (RevIN) | $\bar y$ | $\delta_x + \delta_u - \bar\varepsilon + \zeta$ | $s_x^2(h) + h\sigma_u^2 + \sigma_z^2/w + \sigma_\varepsilon^2$ |
| CN-oracle | $g_T$ | $v_T + \Delta + \zeta$ | $(1-\lambda)V + \sigma_\Delta^2 + \sigma_\varepsilon^2$ |
| CN-est | $\hat m = g_T + e$ | $v_T + \Delta - e + \zeta$ | $(1-\lambda)V + \sigma_\Delta^2 + \sigma_{est}^2 + \sigma_\varepsilon^2$ |

핵심 구조: **IN의 오차에는 $V$도 $\sigma_\Delta^2$도 없다** (수준을 통째로 제거·복원하므로 분포 이동에 면역)
— 대신 지평 내 변화($s_x^2 + h\sigma_u^2$)와 윈도우 노이즈($\sigma_z^2/w$)를 지불한다.
**CN의 오차에는 지평 내 공변량 램프가 없다** (미래 공변량으로 타깃 시점 수준을 직접 추정) — 대신
비설명 성분 $(1-\lambda)V$와 공변량 직교 이동 $\sigma_\Delta^2$를 지불한다.

## 3. 명제별 결과

### 명제 2 — 교차 조건 (계획서 §2.3)

**IN vs RAW**:
$$\text{MSE}_{IN} < \text{MSE}_{RAW} \iff \underbrace{s_x^2(h) + h\sigma_u^2 + \sigma_z^2/w}_{\text{지평 외삽 오차 + 윈도우 노이즈}} < \underbrace{V + \sigma_\Delta^2}_{\text{RAW가 추적 못하는 수준 분산}}$$

계획서 §2.3 첫 부등식의 "$\mathrm{Var}(\Delta\mu_{\text{train}\to\text{test}})$"는 M1에서 $V + \sigma_\Delta^2$
(RAW가 적응하지 못하는 수준 변동 전체), "외삽 편향"은 $s_x^2 + h\sigma_u^2$에 대응.

**CN-est vs IN** — 임계 $\lambda^*$:
$$\lambda^* = 1 - \frac{s_x^2(h) + h\sigma_u^2 + \sigma_z^2/w - \sigma_\Delta^2 - \sigma_{est}^2}{V}$$

- $\lambda > \lambda^*$이면 CN 우세. $\lambda^* < 0$: 모든 λ에서 CN 지배 / $\lambda^* > 1$: 모든 λ에서 IN 지배.
- $\sigma_\Delta^2 \uparrow \Rightarrow \lambda^* \uparrow$ (드리프트가 클수록 IN 영역 확대 — RevIN 원논문 동기의 정량화)
- $s_x^2(h)$가 λ에 의존하는 매핑(아래 대응표)에서는 닫힌형 대신 수치 해를 사용 (`lambda_star_numeric`).

### 명제 3 — 지평 상호작용 (계획서 §2.4)

$$\frac{\partial}{\partial h}\left[\text{MSE}_{IN}(h) - \text{MSE}_{CN}(h)\right] = \sigma_u^2 + \frac{\partial s_x^2}{\partial h} \;>\; 0$$

격차는 h에 단조 증가, $\lambda^*(h)$는 h에 단조 감소 → "왜 다중 단계에서 특히 문제인가"의 답.

### 명제 1 — 표현력 손실 (계획서 §2.2)

최적 예측자가 상호작용 $\kappa\,\bar y\, g$를 요구하는 DGP($y = a\bar y + b g + \kappa \bar y g + \text{noise}$)에서,
상호작용을 표현할 수 없는 제약 클래스(선형 $\{1, \bar y, g\}$ — RevIN형 아핀 복원이 여기 속함)의
비가역적 초과 MSE는 (가우시안 4차 모멘트, Isserlis):
$$\Delta\text{MSE}_{\text{Prop1}} = \kappa^2\left[\mathrm{Var}(\bar y)\mathrm{Var}(g) + \mathrm{Cov}(\bar y, g)^2\right],
\qquad \mathrm{Cov}(\bar y, g_T) = \lambda V$$

(상호작용항이 선형 span과 직교함을 이용한 사영 논증. Toner & Darlow의 제약 회귀 관점의 최소 사례.)

## 4. 명시적 가정 (논문 본문에 그대로 기재)

- **A1 (국소 수준)**: lookback 내 수준은 상수 근사, 수준 변화는 윈도우→타깃 간 $\delta$로 집약.
- **A2 (직교 분해)**: $\Delta \perp x$ — 공변량으로 설명되는 이동은 λ축이, 직교 이동은 드리프트축이 담당 (Fig 1의 두 축이 독립인 이유).
- **A3 (공통 형태 잔차)**: 형태 예측력은 세 방법 동일 ($\sigma_\varepsilon^2$ 공통). 정규화가 형태 학습 난이도에 주는 2차 효과는 무시 — G2에서 경험적으로 점검할 괴리 후보 1순위.
- **A4 (RAW = 전역 평균)**: RAW는 윈도우 정보로 수준을 추적하지 않음 (계획서 §2.3 "전역 z-score만"). 비제약 OLS-RAW는 부록 확장 (TODO, G6).

## 5. 이론 ↔ G2 DGP ↔ 실데이터 대응표 (실험 스크립트 계약)

| M1 기호 | G2 DGP (계획서 §4.1) | 실데이터 대응 | 괴리 기록 |
|---|---|---|---|
| $\lambda$ | $m_t = \sqrt{\lambda}\beta^\top\tilde x_t + \sqrt{1-\lambda}u_t$의 λ | LPS | DGP의 λ는 구성상 $R^2_{level}$ ≈ λ — G2에서 자체 검증 |
| $V$ | $\mathrm{Var}(m_t)$ (윈도우 평균 스케일) | 윈도우 평균의 분산 | |
| $h\sigma_u^2$ | random walk $u$의 h스텝 증분 분산 $\times (1-\lambda)$ | — | **주의: DGP에서 u는 $\sqrt{1-\lambda}$로 스케일** → 매핑 시 $(1-\lambda)h\sigma_{u,\text{dgp}}^2$ |
| $s_x^2(h)$ | AR(1) 공변량의 h스텝 변화가 유도하는 $m$ 변화 분산 $\times \lambda$ | 기상 램프 | AR(1) a=0.9: $s_x^2(h) = \lambda V \cdot 2(1 - a^h/\bar\rho)$ 형태 — G2에서 수치 산출 |
| $\sigma_z^2/w$ | 형태 성분 $z_t$의 윈도우 평균 분산 (AR·계절 자기상관 반영해 유효값 산출) | — | iid 가정과 다름: 자기상관 있으면 $\sigma_z^2/w \to \sigma_z^2 \tau/w$ (τ = 적분 자기상관 시간). `effective_window_noise` 함수로 처리 |
| $\sigma_\Delta^2$ | train/test 구간 간 $u$의 누적 이동 | 연도 간 수요 성장 등 | |
| $\sigma_{est}^2$ | CN-est(LightGBM)의 1단계 OOS 오차 | 〃 | G2에서 실측하여 이론 곡선에 대입 |
| 백본 | **RLinear (단층 선형) 고정** — M1이 선형-가우시안이므로 | G4에서 비선형 확장 | 이론–실험 정합 규약 (the environment contract) |

## 6. Fig 1 사양

- 평면: x = λ ∈ [0,1], y = $\sigma_\Delta^2/V$ ∈ [0, 1.5]. 패널: h ∈ {24, 96, 336}.
- 각 셀에서 {RAW, IN, CN-est}의 닫힌형 MSE argmin → 3색 영역 + $\lambda^*(\sigma_\Delta^2)$ 경계선.
- 기본 파라미터: V=1, w=96, σ_z=1, σ_u²=0.0036 (h=336에서 h·σ_u²≈1.2V가 되도록), σ_est²=0.02, s_x²=0, σ_ε=0.
- 기대 패턴: h가 커질수록 IN 영역이 위로 후퇴(드리프트가 아주 클 때만 IN), CN 영역 확대 — 명제 3의 시각화.
