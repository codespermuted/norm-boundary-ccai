# Pre-registration — shrinkage safety-valve & classical baseline (post-hoc block)

**Date:** 2026-07-23. **Status:** committed **before any run** of either
experiment below.

**This is a POST-HOC block.** It is *not* part of the original pre-registration
(commit `cab17c1`, the LPS spec / τ=0.3 / eight sign predictions). It changes
**no Block A number**; the original Block A results stay frozen at tag
`pre-val-diagnosis`. What is frozen here is the **interpretation rules** below —
fixed before results are seen — not merely the point predictions (which follow
near-mechanically from Table 1; see §A.5).

---

## A. Shrinkage safety valve

**Question.** Does a level-shrinkage safety valve on CondNorm remove the need
for the LPS diagnostic? (The objection: "shrink the level when the first stage
is weak and the ETTh2 6.60 catastrophe disappears, so why diagnose?")

### A.1 Shrink form — primary

Per channel `c`, shrunk level
`ℓ̃_c(x) = μ_c^train + α̂_c · ( ĝ_c(x) − μ_c^train )`, i.e. a fixed intercept at
the **train mean** (matching the §7.3 proposal), with

`α̂_c = clip( Ĉov(ȳ_c^val, ĝ_c^val) / V̂ar(ĝ_c^val), 0, 1 )`.

- **(a) Per channel** — α̂ is estimated per channel (Electricity 321, Weather
  21), the natural correspondence to the per-channel first stage and the
  shrinkage-favourable choice.
- **(b) Clipped to [0,1]** — a first stage anti-correlated on validation would
  give α̂<0 (level amplified the wrong way); clipping forbids it.
- **(c) Fixed intercept at the train mean** — this *is* the §7.3 valve. A
  free-intercept recalibration `a + b·ĝ` would also absorb train→val level
  drift, a different method; it is a **sensitivity** only, reported separately.
- **(d) Validation split = the early-stopping split.** Re-used, not test
  leakage. Stated so a referee need not check.

### A.2 Shrink form — secondary (diagnostic-linked)

`α = clip(LPS, 0, 1)`, reported alongside. Explicitly noted to **under-shrink**:
in M1 the optimal is `α* = λ/(λ+s)`, `s=σ²_est/V`, whereas `LPS = λ − s` (this
identity assumes the covariate-orthogonal shift `σ²_Δ = 0`); e.g. (λ,s)=
(0.80,0.05)→LPS 0.75 vs α* 0.94, (0.30,0.02)→0.28 vs 0.94. LPS mixes the
unexplainable `(1−λ)V` into the deficit; only `σ²_est` should be shrunk. Hence
the empirical α̂ (A.1) is primary.

### A.3 Application — re-train (method A)

The level enters `r_t=(y_t−ℓ̃_t−μ_r)/σ_r`, so α changes the series the backbone
learns. We **re-train** on the shrunk-level residual series (method A).
Restoration-only (method B) trains α=1 residuals and would read as
"half-applied"; not used.

### A.4 Scope (and why this scope)

- Datasets: exogenous **{GEFCom-Wind, Jeju Wind}** + standard **{ETTh1, ETTh2,
  Weather, Electricity}**. Two exogenous, not one, so Tier 3 (A.6) is testable;
  GEFCom-Wind is third-party (max gap 0.841) so the exogenous check is not
  "confirmed on the dataset we curated."
- Backbones: **{RLinear, LightGBM-DMS}**. LightGBM (deterministic, fast,
  flexible function class) pre-empts "shrinkage only helps flexible backbones."
  PatchTST excluded (≈38 min/config).
- Horizons: **{24, 96, 336}** (Jeju **{24, 48}**). Seeds: **Block A's 5**
  (LightGBM deterministic → 1). Two shrink coefficients (A.1 primary, A.2
  secondary). No full grid — this suffices to test whether shrinkage replaces
  the diagnostic (standard) and whether it costs on exogenous (exogenous).

### A.5 Point predictions (frozen)

| dataset | LPS | α̂≈ | CondNorm as-run | shrunk-CN predicted | best instance-norm |
|---|---|---|---|---|---|
| GEFCom-Wind, Jeju (+GLoad/GSolar analog) | 0.74–0.89 | ~0.9 | wins | ≈ CondNorm, still wins | — |
| ETTh1 | −0.72 | 0 | 2.2016 | → Raw ≈ 0.396 | 0.3771 |
| ETTh2 | −0.21 | 0 | 6.6044 | → Raw ≈ 0.387 | 0.2869 |
| Weather | 0.11 | ~0 | 0.7274 | → Raw ≈ 0.182 | 0.1666 |
| Electricity | 0.28 | ~0 | 0.1726 | → Raw ≈ 0.150 | 0.1375 |

These follow near-mechanically from Table 1; the pre-registration's value is the
**interpretation rules (A.6), fixed before results**, not the point values.

### A.6 Interpretation rules — three tiers (fixed in advance)

- **Tier 1 — pre-registered concession (predicted to TRIGGER).** If shrunk-CN is
  within **10%** of the best instance-norm arm on **≥3 of 4** standard datasets,
  the "catastrophic failure" framing is **retracted**. Surviving claims: the
  pre-training cost argument (A.7) and the asymmetry (A.7). Relative gaps of the
  prediction are ETTh1 +5.0%, ETTh2 +34.8%, Weather +9.0%, Electricity +8.7% →
  3/4 within 10%, so this is expected to trigger; we report it as a concession,
  not a pass.
- **Tier 2 — falsification (predicted NOT to trigger).** If shrunk-CN is **≤**
  the best instance-norm arm on **≥2 of 4** standard datasets, the boundary
  itself is wrong on the standard side and the standard-group decision rule
  loses its basis.
- **Tier 3 — exogenous safety-valve cost (predicted NOT to trigger).** If α̂
  shrinkage degrades CondNorm by **>5%** on **either** exogenous dataset, the
  valve has a real cost, reported as-is. (Tier 3 is the only check that the
  empirical α̂ actually avoids the self-harm that `clip(LPS)` would cause.)

### A.7 Surviving narrative (whatever Tier 1 does)

Shrinkage removes the catastrophe but leaves a systematic ~5–9% penalty, and
finding that penalty costs one training run of an already-wrong model; **LPS
gives the same signal for free, before training.** And the boundary is
**asymmetric**: CondNorm's failure off-region is repairable (shrink to +5–9%),
but instance normalization's failure on exogenous series is **not** — Prop 1
(affine restore cannot represent ȳ×g) is the reason, Block B (equal
information, RevIN+cov < Raw+cov on every backbone) the evidence. The stack's
**default (RevIN) sits on the un-repairable side**, which is precisely why a
*pre-training* diagnostic is needed.

---

## B. Classical baseline — regression with ARIMA errors

**Purpose.** A properly-specified classical baseline in main-text Table 1, as
the thesis's origin ("classical practice was right"), and to know before a
referee whether classical methods beat CondNorm on exogenous series. (`dynreg`
already does on Jeju 0.285 and GEFCom-Wind 0.171.) Under the instrument pivot,
a classical method beating CondNorm on exogenous series **strengthens** the
boundary claim (the layer, not any one method).

### B.1 Method

**Regression with ARIMA errors** = regression mean of exogenous (lead-matched)
covariates **+ Fourier seasonal terms** for the long hourly periods (daily
K≤3 on s=24; weekly terms as needed), with **ARIMA errors** — *not* SARIMAX with
a seasonal AR/MA at s=24 (slow, unstable on hourly data, and less orthodox).
Direct multi-step. This is exactly the "regression with ARIMA errors" an IJF
referee expects.

### B.2 Order selection (train split only — no test peeking)

Fourier order K and ARIMA (p,d,q) chosen on the **train split only**, by
auto-search over `p,q ∈ {0,1,2,3}`, `d ∈ {0,1}`, K over a fixed small grid,
selected by **AICc**. Covariates and splits identical to CondNorm's;
lead-matched archived forecasts only (no reanalysis).

### B.3 Scope / priority

**Exogenous 4** (GEFCom wind/load/solar + Jeju) **first** — univariate targets,
where `dynreg` beat CondNorm and the thesis stakes lie. **Standard group only if
budget remains** — there covariates are calendar-only, so the method degenerates
to SARIMA (low information). No prediction is committed on whether ARIMAX beats
CondNorm; it is a baseline reported as-is.

**Compute note.** Per-channel SARIMAX(s=24) on Electricity (321ch × 3h) would be
days and often fails to converge; the Fourier+ARIMA-errors form (B.1) with the
exogenous-first ordering (B.3) keeps this to CPU-hours on the datasets that
matter.

---

## Governance

Frozen at the commit that adds this file, **before running** A or B. Results
reported as-is against A.5/A.6 and B. No Block A number changes; originals
frozen at tag `pre-val-diagnosis`. Re-run entry points and the exact configs are
committed alongside.
