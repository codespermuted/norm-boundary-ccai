# 통계 강화 (G7 item 7) — 부호 기록의 정확 검정 + Fisher 결합 의존성 분석

작성: 2026-07-20 (G7 audit-and-hardening v2). 본 문서의 수치는 전부 동결 원천에서 재검증:
`results/gate2.md`(8/8), `paper/tables/tab3_draft.md`(τ 민감도), `results/g7_fisher_robustness.csv`
(`uv run python -m experiments.g7_fisher_check`로 재현 — 동결 입력 `results/g4_grid.csv`,
`results/g4_errors/`만 읽음).

---

## (a) 사전 등록 부호 기록의 정확 이항 검정

**설정.** 사전 등록(commit `cab17c1`, grid 실행 전)은 8개 데이터셋 각각에 RevIN−CondNorm
격차의 부호를 예측했고, 실측은 **8/8 적중** (`results/gate2.md`). 귀무가설: 규칙이 무정보,
즉 각 부호 예측이 독립 동전던지기 (p = 0.5).

**정확 확률 (이항, n = 8, p = 0.5).**

| 사건 | 확률 | 값 |
|---|---|---|
| 실현된 결과: X = 8 (= 단측 P(X ≥ 8)) | 2⁻⁸ = 1/256 | **≈ 0.0039** |
| 관대한 GATE 기준: P(X ≥ 6) = (C(8,6)+C(8,7)+C(8,8))/2⁸ = (28+8+1)/256 | 37/256 | **≈ 0.145** |

**읽기.** 사전 등록된 통과 기준(≥ 6/8)은 그 자체로는 α = 0.05를 만족하지 못하는 관대한
기준이었다 (P(X ≥ 6 | 동전) ≈ 0.145). 그러나 **실현된 결과는 엄격한 기준을 통과**했다:
8/8은 동전던지기 하에서 p ≈ 0.0039. 즉 GATE 설계는 관대했지만 결과는 관대함에 기대지
않았다 — 이 구분을 본문에 명시하는 것이 정직하고 또 유리하다.

**공개할 한계.** 8개 예측이 완전히 독립이라는 가정은 근사다: 데이터셋은 서로 다른 시계열
이지만 같은 4개 백본·같은 프로토콜을 공유하므로, 규칙이 우연히 프로토콜 특이적 편향과
정렬됐을 가능성은 8개 시행을 완전히 독립으로 세는 것보다 크다. 다만 예측의 원천(LPS)은
grid와 무관하게 데이터·공변량만으로 계산되었고 부호는 데이터셋별로 상이(4+/4−)하므로,
단일 전역 편향("CN이 항상 이긴다"류)으로는 8/8을 만들 수 없다. 문장에는 "under a
binomial reference"로 조건을 명시한다.

**§6 삽입용 문장 (ready-to-paste, `sec6_empirical.tex` "Gate 2: sign predictions" 문단 끝).**

```latex
Under a binomial reference in which each pre-registered sign is an
uninformative coin flip ($n=8$, $p=0.5$), the realized record of eight hits
in eight has exact probability $2^{-8} \approx 0.004$; we note that the
pre-registered pass criterion of at least $6/8$ was itself lenient
($P(X \ge 6) = 37/256 \approx 0.145$ under the same reference), so the
evidence rests on the strict outcome actually observed, not on the
generosity of the gate.
```

---

## (b) Fisher 결합의 의존성 문제 — 진단·대안 비교·권고

### 문제의 구조

Tab 1의 별표는 데이터셋 단위: 각 (backbone, h) 셀에서 DM p값(Harvey 보정, 시드 평균
손실 차분)을 구한 뒤 **Fisher의 방법**(χ²(2K))으로 셀 K개를 결합한다
(`experiments/g4_table1.py`). Fisher는 p값들의 **독립**을 가정하지만, 한 데이터셋의
K개 셀은:

1. **같은 테스트 구간**의 손실 차분에서 계산된다 — 어려운 시기(레짐 전환·이상 구간)는
   모든 백본·모든 h의 차분을 동시에 밀어낸다.
2. h가 달라도 예측 origin 윈도우가 대부분 겹친다.
3. 같은 백본의 다른 h 셀은 같은 학습 표본·같은 튜닝 lookback을 공유한다.

이런 **양(+)의 의존성 하에서 Fisher는 반보수적(anti-conservative)** 이다: χ² 합의 분산이
독립 가정보다 커져 결합 p값이 과소평가된다. 즉 별표가 실제보다 쉽게 붙는다.

### 실측 진단 (results/g7_fisher_robustness.csv)

동결된 `g4_grid.csv` + `g4_errors/`에서 32개 별표 전부를 재계산하고, 의존성-강건 결합
2종과 셀 카운트를 병산한 결과:

- **32개 별표 중 31개는 결합 방식과 무관** — Fisher p가 천문학적으로 작고(최대 5.9e-12,
  대부분 < 1e-30), Simes·조화평균(HMP)로 바꿔도 전부 p < 1e-4로 생존.
- **유일한 취약 별표 = etth2, RevIN vs best(SAN)**: K = 9셀 중 **3셀만 개별 유의**,
  Fisher p = **0.0260** (별표 부여) vs Simes p = **0.130**, HMP p = **0.0769** —
  의존성-강건 결합에서는 α = 0.05 별표가 사라진다 (HMP 기준 α = 0.10에서는 유지).
  이것이 감사에서 지목된 최악 사례다.

### 대안 3종 비교

| 옵션 | 내용 | 장점 | 단점 |
|---|---|---|---|
| **1. Fisher 유지 + 한계 공개** | 동결된 Tab 1 그대로, 본문/각주에 의존성 한계와 위 재분석을 공개 | 표 동결 유지 (재조판·재검증 불필요); Fisher는 예측 문헌에서 표준; 31/32가 결합 방식 불변임을 근거로 결론 강건성 명시 가능 | 형식적으로는 무효한 p값을 별표 기준으로 유지; etth2 별표 1개가 취약한 채 남음 |
| **2. 별표를 셀 카운트(k/K)로 교체** | "K셀 중 k셀 유의" 표기 (예: etth2 RevIN = 3/9). 기존 DM 결과에서 즉시 계산 가능 (`g7_fisher_robustness.csv`의 `k_sig`, `K` 열) | 결합 가정 자체가 불필요 — 의존성 논란 원천 차단; 정보량이 더 많음 (강도 표시) | 단일 유의성 진술이 사라짐 ("데이터셋 수준에서 유의"라 말할 수 없음); 셀 단위 다중성 미보정; 표·캡션 전면 재조판 = 동결 위반 |
| **3. 의존성-강건 결합 (Simes / HMP)** | Simes: 양의 회귀 의존(PRDS) 하 유효. HMP (Wilson 2019, PNAS): **임의 의존성** 하 유효 | p값 하나로 결합하는 형식 유지 — 표 구조 불변, 별표 기준만 교체; 이론적으로 방어 가능 | 검정력 손실 (실측: 별표 32→31); Simes의 PRDS 조건이 이 DM 통계에서 증명된 것은 아님 (HMP는 무조건 유효); 표 재생성 필요 |

### 권고

**옵션 1 + 3의 하이브리드 (현 논문), 옵션 3을 차기 웨이브 사전 등록 (OSF).**

- **현 논문 (동결 존중)**: Tab 1의 Fisher 별표는 그대로 두고, §6.3 통계 문단에 각주
  하나를 추가해 (i) 의존성 한계를 명시하고 (ii) 강건성 재분석을 보고한다: *"32개 별표 중
  31개는 Simes·조화평균 결합에서도 유지되며, 유일한 예외는 ETTh2의 RevIN(9셀 중 3셀
  개별 유의, Fisher 0.026 / HMP 0.077 / Simes 0.130)으로, 이 별표는 지시적(indicative)
  으로 읽어야 한다"*. ETTh2의 정성 결론(RevIN이 CN을 압도, gap −6.31)은 이 별표와
  무관하므로 논문의 어떤 주장도 흔들리지 않는다. 각주 LaTeX 초안은
  `docs/manuscript_revisions_g7.md` 스니펫 (e)에 수록.
- **차기 웨이브 (OSF 사전 등록)**: 데이터셋 수준 결합을 **HMP로 사전 등록** — 임의
  의존성 하에서 유효하고, 이번 재분석에서 검정력 손실이 실질적으로 없음(31/32)이
  확인됐다. `docs/osf_prereg_draft.md` 분석 계획에 반영.

### 재현

```bash
uv run python -m experiments.g7_fisher_check   # -> results/g7_fisher_robustness.csv
```

참고문헌: Wilson, D.J. (2019). The harmonic mean p-value for combining dependent tests.
*PNAS* 116(4), 1195–1200. Simes, R.J. (1986). An improved Bonferroni procedure for
multiple tests of significance. *Biometrika* 73(3), 751–754.
