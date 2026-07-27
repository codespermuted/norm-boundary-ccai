# norm-boundary — complete experiment ledger

**What this file is.** The single authoritative record of every experiment in this
project: purpose → setup → result → verdict, with the number, the source CSV, and
an honest note on what it does and does not support. Every headline number here was
re-derived from the frozen CSVs during the 2026-07-24 consolidation pass; where a
figure differs from an older document, this file is correct and the older one is
superseded.

Scope of the claim, in one line: **instance normalization (RevIN and its endogenous
successors) has an applicability boundary set by whether the series level is
predictable from exogenous covariates; on the far side of that boundary the
endogenous default destroys recoverable signal, and a pre-training score (LPS)
tells you which side you are on.** CondNorm is an *instrument* used to expose the
boundary, not a method we advocate.

Reading guide for the other files:
- `results/summary.md` — phase-by-phase completion record (G0–G6). Historical;
  stops before Block E/F and G8. Kept for the phase-gate history.
- `results/experiment_log.md` — running journal for the Phase-2 (shrinkage/reframe)
  cycle. Kept for the turn-by-turn narrative and the G8 failure as-it-happened.
- `evidence/` — frozen pre-registration and governance artifacts.
- `HANDOFF.md` — state and next steps.
- **This file supersedes all of them for the actual numbers.**

---

## 0. Status at a glance — what is solid, what is not

| Result | Status | Where |
|---|---|---|
| Closed-form M1 risks + crossover (Prop 2, 3) | **Solid** — MC-verified <1%, pytest-pinned | §1 |
| Prop 1 representational deficiency (restated) | **Solid** — corrected this pass, MC-verified | §2 |
| Synthetic crossover matches theory (GATE 1) | **Solid** — 2,640 runs, max dev 0.015 | §3 |
| Pre-registered sign rule 8/8 (MSE, MAE, MASE) | **Solid** — reproduces from frozen CSV | §4 |
| Strategy dissociation 11/12 vs 0/12 | **Solid** — Table 1 with benchmarks | §5 |
| Block B: RevIN+cov worse than Raw+cov | **Solid on 4/5 backbones**, null on PatchTST-Cov | §6 |
| Variance attribution: norm×dataset = 47% | **Solid, but carried by the exogenous arm** (0.4% among endogenous only) | §7 |
| SOTA replications (Block C/D) | **Solid** — pattern holds on TimeXer/iTransformer | §8 |
| LPS as a *regime classifier* | **Solid** — 8/8 aggregated + 10/10 graded wind zones | §4, §4a |
| LPS as a calibrated *dial* (orders magnitude) | **Not established** — aggregated ρ = −0.75 was cross-dataset heterogeneity; within a homogeneous family ρ = +0.47 (n=10, p=0.17, underpowered) | §4a |
| Aggregate 2⁻⁸ significance of 8/8 | **Overstated** — effective n ≈ 2, honest p ≈ 0.25 | §4 |
| Shrinkage removes need for the diagnostic | **Did not adjudicate; but sizing α needs LPS** | §9 |
| Probabilistic (Block F): CondNorm intervals | **Under-covers** — 0.50–0.66 vs 0.80 nominal | §10 |
| Lead-matching claim vs implementation | **KNOWN DEFECT** — jeju h=48 and GEFCom-Load | §11 |
| Single evaluation origin, MSE-centric | **KNOWN LIMITATION** — no rolling origin | §11 |

The four rows in the lower block are the ones a referee ends the paper on, and they
are recorded here as plainly as the wins.

---

## 1. Theory — closed-form boundary (G1)

**Purpose.** Characterize when instance normalization (IN) helps vs hurts, in a
function class matched to the theory (single-layer linear ↔ linear-Gaussian M1).

**Setup.** Stylized instance model M1 (`docs/theory_g1.md`, `paper/sections/app_proofs.tex`):
three level-restoration strategies — Raw (global z), IN (window mean/var), CN
(covariate-conditional). Closed-form MSEs derived; verified against Monte Carlo
(n=2M) and pinned in `tests/test_theory.py`.

**Result (verified <1% rel. error, all parameter grids).**
- Risk table: `MSE_IN = s_x²(h) + hσ_u² + σ_z²/w + σ_ε²` (no V, no shift term);
  `MSE_CN-est = (1−λ)V + σ_Δ² + σ_est² + σ_ε²` (no within-horizon ramp).
- Crossover (Prop 2): CN beats IN iff `λ > λ* = 1 − (s_x² + hσ_u² + σ_z²/w − σ_Δ² − σ_est²)/V`.
- Horizon (Prop 3): `∂/∂h [MSE_IN − MSE_CN] = σ_u² + ∂s_x²/∂h > 0` — the gap grows
  with horizon; λ* falls as h grows.
- Baseline λ* (drift-free): 0.923 (h=24), 0.664 (h=96), <0 (h=336). **These are the
  most IN-favorable stylization and are NOT the anchor for τ** (see §4).

**Verdict.** Solid. The algebra is elementary (variance accounting) but correct and
matched to the experimental function class. It is *motivating apparatus*, not a
standalone theory contribution.

---

## 2. Prop 1 — representational deficiency (G1, restated 2026-07-24)

**Purpose.** What does attaching IN to a backbone *remove* from its function class,
and does that removal separate IN from Raw (not just from the Bayes optimum)?

**Correction history (important).** The originally-published Prop 1 bounded the
interaction-free affine class F = {c₀ + c₁ȳ + c₂g}, which *contains* Raw+covariates,
so it could not separate IN from Raw — yet §7 cited it for exactly that. This pass
restated it after adversarial verification (code / algebra / independent MC / use)
rejected both the original and a first repair. The current form is verified by
`experiments/g7_prop1_verify.py` (all three closed forms reproduce to 3+ sig figs).

**Structural fact (no distributional assumption).** The composed forecast is
`ŷ = ȳ + s·f((x − ȳ·1)/s, g)` for *any* backbone f — additively separable in the
window mean with unit slope. Confirmed on the real stack in float64:
`|f(x+c) − f(x) − c| ≤ 2.7e-12` at c = −5000. So IN cannot represent a window-level
response with slope ≠ 1, nor any window-level × covariate interaction; Raw can.

**Result (two regimes).**
- Affine backbone: F pays `κ²[Var(ȳ)Var(g) + Cov²]` (interaction, paid by Raw too);
  IN pays additionally `(a−1)²Var(ȳ)(1−ϱ²)` (pinned slope, IN only).
- Unconstrained backbone: Raw → Bayes (excess 0); IN floors at
  `[(a−1)² + κ²Var(g)]·Var(ȳ)(1−ϱ²) > 0`. **The IN−Raw gap GROWS with capacity**,
  because the interaction migrates from a shared cost to an IN-only one.

**Hypotheses that bite (documented, asserted in the verify script).** Gaussianity is
load-bearing only for the interaction term (Isserlis; understated ~50% under t₈).
The pinned-slope and floor terms need g centred (else `(a−1+κE[g])²`), not ȳ
centred. Conditioning on the instance scale s is required or F and R are non-nested.

**Verdict.** Solid after correction. The repo already had the correct floor formula
in `experiments/g7_prop1_demo.py:129` (`revin_irreducible`); it simply had not been
put in the proposition. The MLP demo (a=1) confirms the floor: RevIN-MLP excess
0.085 (λ=0.4) / 0.067 (λ=0.8) vs Raw-MLP 0.003 — `results/g7_prop1.csv`.

---

## 3. Synthetic validation — GATE 1 (G2)

**Setup.** λ (11) × h {24,96,336} × L {96,336} × seed (10) × norm (4), RLinear fixed
= **2,640 runs**. `results/synth_grid.csv`.

**Result (GATE 1 = GO).**
- Crossover ±0.1: 6/6. λ*_emp vs λ*_OLS(Prop 2′) max abs dev **0.015**.
- Horizon widening (λ=0.8): 2/2 (seed-pair 95% CI non-decreasing).
- CN-est degradation explained by theory: median ratio 1.12.
- Key finding: λ*_M1 ≈ 0.27–0.28 (restoration-rule upper bound) ≫ λ*_OLS ≈
  0.007–0.029 (where trained linear predictors actually cross). The gap measures
  implicit level tracking by the linear backbone. `results/gate1.md`.

**Verdict.** Solid. Caveat: the ±0.1 tolerance is loose relative to the true
crossing (0.007–0.029), so the trivial null λ*=0 also passes — the synthetic gate
confirms the *ordering and horizon effect*, not the precise location.

---

## 4. Pre-registered sign rule — GATE 2 (G4) + what LPS does NOT do

**Pre-registration.** `paper/predictions.md` + `results/lps_official.csv` +
`experiments/compute_lps_official.py`, commit `cab17c1` (2026-07-13 13:20:23 KST),
**before the first grid run** (13:24:36, +4m13s). Each artifact has exactly one
commit in the history. LPS = OOS R² of window-mean levels on covariates (w=96,
LightGBM, expanding CV); rule "LPS ≥ τ, τ=0.3".

**LPS values (official, `results/lps_official.csv`).**

| exogenous | LPS | | standard | LPS |
|---|---|---|---|---|
| gefcom_load | 0.894 | | electricity | 0.283 |
| gefcom_solar | 0.875 | | weather | 0.110 |
| jeju_wind | 0.745 | | etth2 | −0.205 |
| gefcom_wind | 0.744 | | etth1 | −0.717 |

**Result — 8/8 sign hits, robust to metric.** RevIN−CondNorm dataset-mean gap sign,
predicted before running, all eight correct. Re-derived from `results/g4_grid.csv`
this pass under three metrics — **8/8 under MSE, MAE, and MASE**:

| dataset | gap MSE | gap MAE | gap MASE | pred | hit |
|---|---|---|---|---|---|
| jeju_wind | +0.404 | +0.216 | +0.263 | + | ✓ |
| gefcom_wind | +0.841 | +0.498 | +0.541 | + | ✓ |
| gefcom_load | +0.237 | +0.207 | +0.559 | + | ✓ |
| gefcom_solar | +0.126 | +0.140 | +0.652 | + | ✓ |
| etth1 | −1.824 | −0.673 | −1.599 | − | ✓ |
| etth2 | −6.314 | −1.458 | −4.601 | − | ✓ |
| electricity | −0.027 | −0.019 | −0.069 | − | ✓ |
| weather | −0.555 | −0.308 | −0.768 | − | ✓ |

**What LPS does NOT do (both verified this pass, both go in as limitations).**
1. **The aggregate 2⁻⁸ = 0.004 is misleading.** The 8 datasets are 2 correlated
   regimes; granting only that the two *regimes* were called correctly gives p =
   0.25. Effective n ≈ 2. The informative content is per-dataset, not the hit count.
2. **Aggregated data did not establish LPS as a magnitude dial.** Pooled Spearman
   ρ = +0.783 (23 cells) is *cluster separation*; **within the 11 aggregated
   exogenous cells ρ = −0.750 (p = 0.008)** — the two highest-LPS datasets (load
   0.894, solar 0.875) had the two *smallest* gaps. The pre-registered auxiliary
   "magnitude increases with LPS" is **not supported by the aggregated panel** and
   is withdrawn there. The nested-cell p = 9.9e-6 is invalid; cluster-permutation
   p = 0.023. **§4a (graded-LPS) shows the aggregated −0.75 was a cross-dataset
   heterogeneity artifact: within a homogeneous wind family the trend is positive
   (+0.47), consistent with the theory but underpowered.**

**Verdict.** The sign rule (classifier) is solid and metric-robust. The dial claim
is *not established*: negative across heterogeneous datasets, positive but
underpowered within a homogeneous family (§4a). LPS is a demonstrated regime
classifier, not (yet) a calibrated dial. The threshold plateau τ∈[0.30,0.70] sits
in the empty gap between the two clusters (0.283 → 0.744), so it shows insensitivity
to misplacement on this panel, not fine calibration; the aggregated boundary is
stress-tested at effectively n=1 (electricity), extended to 10 graded values in §4a.

---

## 4a. graded-LPS — does LPS order magnitude within a homogeneous family? (2026-07-27)

**Pre-reg** `evidence/prereg_graded_lps.md` (commit `5f888f3`, before the grid).
**Runner** `experiments/graded_lps.py`. **Data** `results/graded_lps.csv`,
`results/graded_lps_lps.csv`, `results/graded_lps_zonegaps.csv`.

**Question.** The aggregated within-exogenous ρ = −0.75 (§4) was the project's most
damaging honest finding, but it rests on 4 *heterogeneous* datasets (wind/load/solar
differ in covariate structure). Does LPS order the IN-penalty magnitude within a
*homogeneous* family?

**Setup.** GEFCom-Wind is a mean over 10 wind zones with genuinely different NWP
skill; disaggregate into 10 subseries (same physical process, same 2-covariate
structure), each with its own pre-training LPS. Arms RLinear {Raw, RevIN, CondNorm}
× h{24,96,336} × 5 seeds. Modeling identical to the main grid (frozen LPS protocol,
train-only first stage). GEFCom covariates are target-time forecasts → no
lead-matching defect. LightGBM-DMS deferred (h=336 ≈ 40 min/zone; RLinear is the
theory-matched linear backbone and suffices for the M1 test).

**Result (RLinear, 10 zones; `results/graded_lps_zonegaps.csv`).**

| quantity | value |
|---|---|
| per-zone LPS range | 0.575–0.758 (narrow, all > τ) |
| **PRIMARY** Spearman(LPS, RevIN−CondNorm gap) | **+0.467** (p = 0.174, n = 10) |
| Spearman(LPS, Raw−CondNorm, covariate value) | +0.503 (p = 0.138) |
| Spearman(LPS, RevIN−Raw, endogenous) | −0.479 (p = 0.162) |
| sign rule (RevIN−CondNorm > 0) | **10/10 zones** |
| horizon trend (gap by h) | 0.53 → 0.79 → 0.90, monotone ↑ (corrected 2026-07-27: an earlier draft printed 0.49/0.76/0.86, which reproduces from no aggregation of the frozen CSV; canonical = RLinear-only zone-mean via `graded_lps_analyze.py`, which also had a pooling bug — partial LGBM rows leaked into the primary, +0.503 vs the correct +0.467 — fixed the same day) |

**Verdict (as-is, honest).** The magnitude ordering is **directionally positive**
within a homogeneous family (+0.47), consistent with M1, and it **reverses the
aggregated −0.75** — so the aggregated negative was cross-dataset heterogeneity, not
a failure of the coordinate. But n = 10 over a narrow LPS range (0.575–0.758) is
**underpowered** (p = 0.17): this does not establish a calibrated dial. The
classifier is **confirmed at graded values (10/10)** and the horizon prediction
holds. Net for the paper: removes the most damaging finding ("LPS orders magnitude
*negatively*") and replaces it with "classifies robustly; trends correctly within a
family; a calibrated dial needs a wider-range panel." **Caveat:** RevIN−Raw here is
*endogenous* (neither arm sees covariates) — this is NOT the Block B information-
parity result (§6); it is small and noisy as expected. LightGBM robustness and solar
zones remain optional next steps (solar reintroduces heterogeneity).

---

## 5. Main grid (Block A) — the strategy dissociation

**Setup.** 5 norms {Raw, RevIN, SAN, FAN, CondNorm} × 4 backbones {RLinear,
PatchTST, SegRNN, LightGBM-DMS} × h × 5 seeds = **1,794 runs** (LGBM×SAN/FAN N/A).
Lookback tuned on RevIN-arm val loss, frozen across arms. FROZEN at tag
`pre-val-diagnosis`; `results/g4_grid.csv`. All 40 Table-1 cells reproduce exactly.

**Result — Table 1 (test MSE, backbone×h×seed mean), with benchmarks + MASE.**

Covariate-conditioned level models (first-stage-only, ridge dynreg, CondNorm) vs the
best instance-norm arm, per row:

| group | beats best-IN (MSE) | beats best-IN (MASE) |
|---|---|---|
| exogenous (4 datasets × 3 arms) | **11/12** | **10/12** |
| standard (4 datasets × 3 arms) | **0/12** | **0/12** |

Both exceptions are ridge dynreg (on gefcom_load, MSE; +gefcom_solar, MASE) — the
nonlinear temperature/level response, the same failure that costs ridge two hits in
the LPS sensitivity check. No covariate-conditioned model wins on any standard
dataset under either metric. This is the paper's central result and it is about the
*layer*, not about CondNorm-the-method.

MCS (α=0.10): 11 exogenous (dataset,h) cells all retain {CondNorm} alone; standard
cells retain RevIN/SAN/FAN. **Caveat:** MCS returns a *singleton* in 20 of 23 cells —
on a single test path with backbone-averaged pseudo-models this reports a ranking,
not an uncertainty set. Read as a compact ranking, consistent with the DM marks.

---

## 6. Information parity (Block B) — the mechanism

**Setup.** Every arm gets identical past+future covariates, 5 covariate-capable
backbones, exogenous group only. 1,133 runs. `results/g4_covfair_full.csv`.

**Result — isolated cost of the layer, `MSE(RevIN+cov) − MSE(Raw+cov)`** (cluster
bootstrap over 11 seed-averaged cells, this pass):

| backbone | mean | 95% CI | cells>0 |
|---|---|---|---|
| LGBM-Cov | +0.0191 | [+0.0116, +0.0269] | 11/11 |
| Linear mixer | +0.0205 | [+0.0056, +0.0368] | 8/11 |
| MLP mixer | +0.0238 | [+0.0119, +0.0390] | 10/11 |
| SegRNN-Cov | +0.0112 | [+0.0058, +0.0172] | 10/11 |
| **PatchTST-Cov** | **+0.0038** | **[−0.0091, +0.0167]** | 7/11 |

Four of five exclude zero; **PatchTST-Cov is null** and is reported as such (not
absorbed into "every backbone"). Magnitudes are hundredths of an MSE unit — one to
two orders below the +0.841 headline gap; across all 55 cells the range is −0.0261
to +0.0849, 46 positive. Small because it is the layer cost alone, not covariate
access.

Capacity: Raw−CN gap shrinks along the mixer ladder (+0.127 linmix → +0.052 mlpmix →
−0.009 lgbmcov; exploratory, mixer-ladder only, PatchTST-Cov +0.142 and SegRNN-Cov
−0.004 sit outside). The IN−Raw difference does not trend to zero at any capacity —
consistent with Prop 1's floor. **The five backbone means show no monotone relation
to flexibility, so the theory's "grows with capacity" prediction is not resolved
by this panel.**

**Verdict.** This is the load-bearing mechanism evidence and the only claim fully
supported without pre-registration. It shows the layer, not covariate access, is the
operative variable — but it also shows CondNorm is *not* the best method here
(Raw+cov beats it on the two most flexible backbones), which is consistent with the
instrument framing.

---

## 7. Variance attribution (G5)

**Result (`results/g4_grid.csv`, log-MSE, this pass).**
- All 5 arms: norm×dataset **47.12%**, norm main 0.73%, all backbone terms 0.6%.
- **Endogenous 4 arms only: norm×dataset 0.43%, all norm-related 0.73%, backbone
  0.55%.**

**Verdict.** The 47% is carried by the exogenous (CondNorm) arm. This *locates* the
boundary rather than deflating it: within the endogenous family normalization is
near-immaterial (0.4%), so the boundary runs between endogenous and exogenous level
handling, not among RevIN/SAN/FAN. Reported both ways in the paper; the honest
reading is the stronger one.

---

## 8. SOTA replications (Blocks C/D)

**Setup.** Arms injected into TimeXer and iTransformer (built-in norm removed),
with (C) and without (D) official covariate paths. 1,275 runs; 4,202 study total.
`paper/tables/tabC_sota.md`.

**Result.** RevIN−CondNorm mean gap +0.4122 (TimeXer-MS, 55 pairs), +0.4282
(iTransformer-MS); CondNorm best on all 4 exogenous datasets for both. Endogenous
mode (D) reproduces Block A qualitatively (CN wins exo, loses standard, e.g.
iTransformer etth2 CN 3.47 vs RevIN 0.30).

**Verdict.** Solid robustness: the pattern is not an artifact of the minimal
covariate adapters. **Cross-block numeric comparison is prohibited** (different
budgets); each block is internally uniform.

---

## 9. Shrinkage safety-valve (G8) — the failed-then-informative block

**Question.** "Shrink the level toward the train mean when the first stage is weak;
the ETTh2 6.60 catastrophe disappears — so why need a pre-training diagnostic?"

**Setup.** Pre-registered (`evidence/prereg_shrinkage_arimax.md`). ℓ̃ = μ + α(ĝ−μ),
re-trained (method A). Two coefficients: α̂ = clip(Cov/Var, 0,1) per channel
(val-split); clip(LPS). 204 runs. `results/g8_shrinkage.csv`.

**Result — four-point α response (matched cells, this pass).**

| dataset | LPS | α=0 (Raw) | α=clip(LPS) | α=α̂ | α=1 (CN) |
|---|---|---|---|---|---|
| etth1 | −0.717 | **0.386** | 0.387 | 0.511 | 2.751 |
| etth2 | −0.205 | **0.445** | 0.451 | 0.778 | 8.781 |
| weather | 0.110 | 0.185 | **0.177** | 0.190 | 0.842 |
| electricity | 0.283 | 0.144 | **0.139** | 0.155 | 0.173 |
| gefcom_wind | 0.744 | 1.072 | 0.208 | **0.169** | **0.169** |
| jeju_wind | 0.745 | 0.688 | 0.316 | **0.295** | **0.295** |

**Verdict (Option C, reframed as positive).** The loss-minimizing α tracks LPS:
LPS<0 → best at α=0; LPS 0.11/0.28 → best at α=clip(LPS) (beats Raw by 4.6%/3.7%,
electricity also beats best-IN); LPS≈0.74 → best at α=1. So **sizing the safeguard
needs exactly what LPS measures** — the objection does not remove the diagnostic, it
relocates it. The diagnostic-free alternative (textbook val recalibration α̂) is
worse than Raw on all four standard datasets (+0.2% to +75%): it minimizes *level*
MSE, not forecast MSE, and the backbone already recovers part of the level.

**Correction to the internal diagnosis.** The experiment_log's "variance" story is
wrong. Verifier simulation: under a calibrated null α̂ is low-variance (P(α̂>0.45)
= 0.001), and the oracle test-split α (0.451/0.401/0.538/0.860) matches the val
estimate to within 0.03–0.12. α̂ is a low-variance estimator of the *wrong estimand*.
So no variance-targeted estimator (James-Stein, CV-fold, gate) rescues it —
running one would be estimator-shopping.

**Pre-registration honesty note.** The manuscript previously said shrinkage was
"deliberately not applied"; that was false (this block existed) and is corrected.
The A.6a matched-cell rule was fixed after seeing one smoke point but before any
matched baseline, and is disclosed as such.

---

## 10. Probabilistic (Block F) — the honest failure

**Setup.** Quantile arms (rlinear_q 9 quantiles, lgbm_q 3), pinball + empirical
cov80. 644 runs. `results/g7_blockf.csv`.

**Result.** CondNorm has the **best pinball on all 4 exogenous datasets** (0.135–
0.165 vs 0.16–0.32 for IN) — the point-forecast boundary transfers to pinball. **But
its intervals under-cover badly on exactly that group:** empirical cov80 (nominal
0.80) = jeju 0.576, gefcom_wind 0.502, load 0.663, solar 0.653, against RevIN
0.79–0.89. On the standard group CondNorm collapses (etth2 cov 0.032).

**Verdict.** Kept as an honest caveat, not a result. CondNorm sharpens the point
forecast but does not propagate first-stage uncertainty, so its intervals are
overconfident precisely where the method wins on points. This **contradicts any
"tighter reserve margins" operational-impact story** (reserve sizing is a quantile
decision) and must be stated wherever that story appears (incl. the workshop paper).

---

## 11. Known defects and limitations (read before trusting any exogenous number)

1. **Lead-matching claim vs implementation (DEFECT).** The paper states all NWP
   inputs are "archived operational forecasts matched by issue lead." The code does
   not implement horizon-dependent band selection: `experiments/g4_grid.py:461`
   fits the first stage once per dataset, outside the horizon loop.
   - **jeju h=48** consumes the `ws_da` band that `src/data/curation.py:53` itself
     labels "h≤24-valid" → the h=48 cell (the only real-data Prop 3 test) is
     compromised. Fix: horizon-specific covariate set (`ws_d2` only for h=48),
     re-run ~100 cells.
   - **gefcom_load** covariate is *realized* station temperature
     (`curation.py:105-108`), not a forecast — perfect weather foresight, disclosed
     only in limitations. GEFCom-Load carries the highest LPS (0.894).
   - GEFCom wind/solar h=96/336 are **NOT** compromised: GEFCom2014 ships the whole
     target-month ExpVars block, so those horizons are within the availability
     envelope. (This corrects an over-broad worry raised mid-consolidation.)
2. **Single evaluation origin (LIMITATION).** One chronological train/val/test split
   per dataset; seeds vary initialization, not origin. No rolling-origin / multiple-
   origin evaluation. Every DM mark, MCS set and margin inherits one realized path.
   Interim free fix: promote the validation segment as a second origin (r-space
   corrected, all four exogenous signs reproduce). Proper fix: 4–6 expanding origins.
3. **Metric monoculture (LIMITATION).** Main results are MSE on the train-fit global-
   z scale. MASE added this pass (§4–5). No sMAPE/RMSSE; probabilistic in §10 only.
4. **LPS is bimodal across the aggregated panel (LIMITATION, partly addressed).**
   0.74–0.89 vs −0.72–0.28, nothing between; the aggregated boundary is stress-tested
   at n=1 (electricity). §4a (graded-LPS) added 10 graded values (0.575–0.758) within
   a homogeneous wind family: the classifier holds 10/10 and the magnitude trend is
   positive (+0.47) but underpowered over that narrow range. Still open: intermediate
   values in the empty gap (0.28–0.575) and a wider spread for a calibrated dial.
5. **No classical baseline actually run.** Only a ridge `dynreg` stand-in, which
   *beats* CondNorm on Jeju and GEFCom-Wind. A properly specified regression-with-
   ARIMA-errors is scoped (`evidence/prereg_shrinkage_arimax.md` §B) but not run.
6. **Jeju Wind is self-curated** and developed interleaved with method development
   (disclosed as a development dataset; GEFCom + standard benchmarks are third-party
   and confirmation-style).
7. **val/test scale artifact** (`evidence/condnorm_val_scale_diagnosis.md`):
   CondNorm's logged val_mse is 9–66× its test MSE (log-scale r-space vs global-z
   difference, not leakage). Documented, not code-patched; proven argmin-invariant
   for univariate, ≤1.61% for the one multivariate cell checked.

---

## 12. Reproducibility

- `make figures` / `make tables` regenerate from the released CSVs (no raw data).
- `uv run pytest` — theory identities pinned (`tests/test_theory.py`, 14/14).
- `experiments/g7_prop1_verify.py` — Prop 1 closed forms vs MC, incl. the documented
  hypothesis-failure cells (heavy-tail, non-centred), asserted.
- `experiments/graded_lps.py` — graded-LPS runner (`--lps`, `--grid`,
  `--rlinear-only`); reuses the frozen torch_run/lgbm_run + LPS protocol per zone.
- Frozen artifacts: `results/g4_grid.csv` (tag `pre-val-diagnosis`), pre-registration
  commits `cab17c1` (main grid) and `5f888f3` (graded-LPS),
  `evidence/mlflow_snapshot_20260722.db.gz`.
- **Open governance item:** several MLflow git-commit tags in the evidence snapshot
  resolve to no object after a history rewrite; the "independently verifiable" claim
  needs either published grafts or an honest "objects are gone" note before release.

---

*Consolidated 2026-07-24; graded-LPS (§4a) added 2026-07-27. Numbers re-derived from frozen CSVs during this pass;
where this file and an older document disagree, this file is authoritative.*
