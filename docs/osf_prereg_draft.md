# OSF Pre-registration DRAFT — Confirmatory Wave 2 (norm-boundary, G7 item 9)

상태: **초안** (2026-07-20, G7 audit-and-hardening v2). OSF 폼 제출 전 확정 필요 항목은
`[확정 필요]`로 표시. 규약: 가설·결과변수·분석 계획은 영어(OSF 폼 스타일), 내부 노트는
한국어. 등록 시점 = 아래 "Registration freeze" 산출물이 커밋된 시각이며, 그 커밋 해시가
증빙이다 (Wave 1의 `cab17c1` 방식과 동일).

---

## 1. Study Information

**Title.** When instance normalization hurts, wave 2: confirmatory replication and
stress test of the LPS decision rule for time-series forecasting normalization.

**Description.** Wave 1 (pre-registered at commit `cab17c1`) fixed a pre-training
diagnostic (the Level Predictability Score, LPS; official spec: window w=96, LightGBM
first stage, expanding chronological CV with 5 folds and min-train-frac 0.4,
channel-mean, covariate set identical to what CondNorm receives) and a threshold
tau = 0.3, and predicted the sign of the RevIN-minus-CondNorm test-MSE gap on 8
datasets; the realized record was 8/8. Wave 2 tests the rule on new data it has never
seen, upgrades the decision procedure to an inference-based rule, and adds two
pre-registered extensions (capacity-ladder monotonicity; probabilistic metrics).

### Hypotheses (confirmatory)

**H1 — M5 confirmatory replication.** On the M5 (Walmart) retail dataset (aggregated
series; see Design), the sign of the dataset-mean RevIN-minus-CondNorm test-MSE gap
will match the prediction issued by the permutation-CI decision procedure (Section 4)
computed from the training data and covariates alone, before any grid run.

**H2 — SMP stress test (high-LPS, high-persistence quadrant).** On hourly Korean
system marginal price (EPSIS SMP) with fuel-price and demand covariates — a series we
predict a priori to fall in the quadrant {absolute LPS above tau, strong persistence
(high R2_pers), low incremental Delta-LPS} — the theory predicts that absolute LPS
overstates the value of conditional normalization:

- **H2a (regime-conditional gap).** The RevIN-minus-CondNorm gap, evaluated on
  pre-declared regime-shift test windows (top-quintile trailing fuel-price change;
  definition frozen at registration), will be algebraically larger (more favorable to
  CondNorm) than the gap on the remaining calm windows.
- **H2b (calm-regime adequacy of IN).** On calm windows the gap will be at or below
  zero (instance normalization adequate), consistent with the Delta-LPS refinement:
  when covariates are largely redundant with the recent history, IN already exploits
  the level signal.

**H3 — Capacity-ladder monotonicity.** Under the covariate-fair protocol (all arms
receive identical past and future covariates), the RAW-minus-CondNorm gap is weakly
decreasing along the declared mixer capacity ladder linmix -> mlpmix -> lgbmcov, on
each wave-2 dataset (evaluated on the dataset-mean gap). PatchTST-class and
RNN-class covariate adapters (patchtstcov, segrnncov) are **excluded from this
hypothesis a priori**: their covariate adapters bottleneck how much level information
reaches the backbone, confounding function-class flexibility with information access
(wave-1 Block B measured patchtstcov at +0.142, off the ladder ordering
linmix +0.127 -> mlpmix +0.052 -> lgbmcov -0.009; they are reported descriptively).

**H4 (secondary) — Probabilistic metrics.** On the Block F backbone pair (Section 5),
the CondNorm-versus-RevIN ordering observed under MSE transfers to probabilistic
evaluation on the exogenously-driven wave-2 data: CondNorm attains lower mean pinball
loss and lower quantile-CRPS, and central-interval coverage closer to nominal 80%,
whenever H1/H2's point-forecast verdict favors CondNorm on that dataset.

## 2. Design Plan

**Study type.** Pre-registered computational experiment; all randomness seeded and
deterministic algorithms enforced (`torch.use_deterministic_algorithms`,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`).

**Blinding / ordering guarantee.** Registration freeze = one commit containing:
(i) this document finalized, (ii) official LPS values, permutation p-values, and CIs
for every wave-2 dataset, (iii) the auto-generated sign predictions, (iv) the frozen
grid configs, (v) the regime-shift window definition for H2 (indices computed from
covariates only). No grid run is launched before that commit; commit timestamps are
the evidence (as audited for wave 1: prediction commit 13:20:23 vs first run
13:24:36 on 2026-07-13).

**Datasets.**

- **M5 (H1):** Walmart daily unit sales, aggregated to `[확정 필요: store-level vs
  department-level aggregates, 채널 수]`; covariates: sell prices, SNAP indicators,
  calendar events + calendar harmonics. Horizons h in {7, 28} (daily resolution;
  the plan's resolution-adjustment clause, RESEARCH_PLAN.md section 5.2, applies —
  the competition horizon is 28). Third-party frozen data (confirmation-style).
- **SMP (H2):** EPSIS hourly system marginal price (mainland), multi-year
  `[확정 필요: 확보 가능 기간; EPSIS 수동 확보 이슈는 HANDOFF.md G4 착수 전 이슈 2 참조]`.
  Covariates: published fuel-price series (LNG, coal indices; monthly/weekly, forward
  filled at issue date), day-ahead demand forecast or lagged demand (leakage
  discipline: no same-period actuals), calendar harmonics. Curated by us during the
  study (development-style; disclosed as in wave 1's Jeju Wind footnote).
- Wave-2 grid on both datasets: normalization arms {RAW, RevIN, SAN, FAN, CondNorm},
  backbones {RLinear, PatchTST, SegRNN, LightGBM-DMS}, seeds {0..4}, Block A protocol
  (epochs 12; lookback tuned per (dataset, backbone, h) on RevIN seed-0 validation
  MSE from {96, 192, 336, 720}, then frozen for all arms — capacity identical across
  arms). H3 uses the covariate-fair protocol of wave 1 (experiments/g4_covfair_full.py:
  mixer backbones with tuned L, NN learning rate, epochs 15, seeds 0-4, and the
  LightGBM covariate run with bounded n_jobs).

**Namespace.** All wave-2 artifacts live under `results/g7_*` (CSV) and
`results/g7_errors/` (per-window loss arrays); wave-1 result files are frozen and
never rewritten.

## 3. Sampling / Data Collection

- Curated Parquet under the repository data contract (monotone DatetimeIndex, no
  missing timestamps within segments, no NaN; segment-aware windowing for archive
  gaps; global z-score fit on train split only; out-of-time splits, DataLoader
  drop_last=False).
- Any dataset failing the curation contract is dropped **before** LPS computation and
  the failure documented; no dataset may be added or removed after registration
  freeze.

## 4. Primary decision procedure (NEW in wave 2): permutation-CI rule

Wave 1's rule was a point comparison (LPS >= tau). Wave 2 **promotes the inference
rule of `src/theory/lps_inference.py` to the primary decision procedure**:

1. Compute official LPS (spec unchanged from wave 1).
2. **Significance:** LPS must be significant against the circular-shift null — the
   covariate series is circularly shifted relative to the target (preserving both
   marginals and autocorrelation), the LPS recomputed per shift; require
   p_perm < 0.05, with B = 999 shift draws (exact enumeration whenever fewer than
   999 distinct stride-w circular shifts exist, as implemented in
   `src/theory/lps_inference.py::permutation_test`); shifts operate on the
   window-mean sequence (stride w, minimum displacement one full window), and the
   season-aligned variant (shifts restricted to whole weeks) is reported as a
   sensitivity companion, not the decision statistic (its admissible-shift count is
   small — 16–90 on wave-1 datasets — so its p-value granularity is coarse).
3. **Confidence interval:** the 90% moving-block bootstrap interval reported by
   the same module (`mbb_ci`, B = 499 replicates, circular blocks of length
   ceil(n_windows^(1/3)), percentile method) must exclude tau = 0.3.
   Known bias, declared up front: for series whose LPS is negative or near zero,
   block resampling dilutes across-fold drift and biases the CI upward (wave-1
   post-hoc evidence: results/lps_inference.md Caveats). The abstention rule in
   step 4 therefore treats "CI straddles tau" as no-prediction rather than a
   RevIN verdict, and the permutation test of step 2 — which has no such bias —
   is the binding significance gate.
4. **Verdict:** CI entirely above tau AND significant -> predict CondNorm (+);
   CI entirely below tau, or not significant -> predict RevIN (−);
   CI straddles tau -> **"boundary — no confirmatory prediction"**, declared before
   the grid and reported descriptively (this replaces wave 1's informal
   "least-confident cell" flag for Electricity with a formal abstention rule).

Delta-LPS and R2_pers are computed alongside (same expanding protocol) and drive the
H2 regime predictions, but the primary sign verdict remains the absolute-LPS
permutation-CI rule.

## 5. Outcome measures

**Primary outcomes.**

- H1: sign of the dataset-mean RevIN-minus-CondNorm test MSE gap (global z-score
  scale; mean over backbones, horizons, seeds) vs the pre-registered verdict.
- H2a/H2b: the same gap computed separately on the pre-declared regime-shift vs calm
  test windows; H2a is the difference of the two gaps, H2b the calm-window gap.
- H3: the ordered triple of dataset-mean RAW-minus-CondNorm gaps along
  (linmix, mlpmix, lgbmcov).

**Secondary outcomes (pre-registered, H4).** Mean pinball loss over quantile levels
{0.1, ..., 0.9}, quantile-approximated CRPS (average pinball across the nine levels,
times 2), and empirical coverage of the central 80% interval (q0.1-q0.9), for the
**Block F backbone pair** (확정, docs/blockf_design.md): RLinear-Q (9-quantile
pinball head, Block-A capacity/lookback) and LightGBM-Q (objective='quantile',
levels {0.1, 0.5, 0.9}; CRPS approximation therefore coarser and compared
within-backbone only). One linear + one GBM quantile-native backbone spans the
capacity axis while keeping the normalization contrast isolated.

**Exploratory (declared, not confirmatory).** Per-(backbone, h) cell hit rates;
SAN/FAN placements; MCS survivor sets; Delta-LPS tau-sensitivity.

## 6. Analysis Plan

- **Significance of per-cell comparisons:** DM tests with Harvey small-sample
  correction on per-(backbone, h) seed-mean loss differentials (Bartlett HAC,
  bandwidth h-1 capped at n/4), unchanged from wave 1.
- **Dataset-level combination: harmonic-mean p-value (HMP)** replaces Fisher.
  Rationale (G7 item 7, `docs/stats_hardening.md`): cells within a dataset share the
  same test period, so they are positively dependent and Fisher is anti-conservative;
  HMP is valid under arbitrary dependence, and the wave-1 re-analysis showed the
  power cost is negligible (31 of 32 stars unchanged). Per-cell counts (k of K
  significant) are reported alongside.
- **Sign record:** exact binomial reference is reported descriptively; with only
  `[확정 필요: 최종 데이터셋 수]` wave-2 datasets the confirmatory claims are the
  per-dataset hypotheses H1-H2, not an aggregate hit rate.
- **H2a:** DM-type test on the regime-split loss differentials (shift-window losses
  vs calm-window losses), HAC variance; declared one-sided in the direction of H2a.
- **H3:** declared as a weak-ordering check (gap_linmix >= gap_mlpmix >= gap_lgbmcov,
  each within seed-level Monte Carlo error); no distributional test is imposed on the
  ordering itself (n=3 rungs), so H3 is confirmed iff the ordering holds on the
  dataset-mean gaps.
- **H4:** paired per-origin pinball/CRPS differentials, DM with Harvey correction;
  coverage compared by absolute deviation from nominal 0.80.
- **Multiple outcomes:** primary hypotheses are few and enumerated; no further
  correction beyond the HMP within-dataset combination. Secondary/exploratory results
  are labeled as such.

## 7. Stopping rules and deviations

- No interim looks at any test-split metric before the grid completes; validation
  metrics are used only where the frozen protocol says so (lookback tuning on RevIN
  seed 0).
- Failed runs are retried under the supervised runner (`scripts/supervised_run.sh`,
  OOM-safe backoff); a cell with fewer than 5 completed seeds after retries is
  reported as incomplete, never silently averaged.
- If SMP data acquisition fails (EPSIS availability), H2 is dropped **before**
  registration freeze — never after; the freeze commit fixes the final dataset list.
- Any deviation from this plan after the freeze is reported in the paper's deviations
  table with a timestamped commit trail.

---

## 내부 노트 (OSF 폼에는 미포함)

- H2의 이론적 근거: HANDOFF.md "G5 진단 정밀화" — SMP류(강한 지속성 + 높은 절대 LPS)는
  공변량-이력 중복으로 절대 LPS가 과대평가되는 사각지대. ΔLPS가 보정 진단. 이론 예측:
  평시 IN 우세, 레짐 전환 구간 CN 우세. 본 문서는 그 예측을 H2a/H2b로 형식화한 것.
- H3 사다리·제외 근거 수치는 wave-1 Block B 최종값 (HANDOFF 2026-07-20 정정본):
  linmix +0.127 / mlpmix +0.052 / lgbmcov −0.009, patchtstcov +0.142, segrnncov −0.004.
- `src/theory/lps_inference.py`는 G7 병렬 트랙 산출물 — freeze 전에 n_perm·CI 방법·수준을
  구현값으로 치환하고, 해당 모듈 pytest 통과를 freeze 커밋에 포함할 것.
- M5 집계 수준·SMP 기간·Block F 백본 쌍은 freeze 전 확정. M5는 제3자 동결(확증형),
  SMP는 자체 큐레이션(개발형) — 논문에 wave-1 Jeju 각주와 같은 공개 문구 적용.
