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
- `docs/archive/experiment_log.md` — running journal for the Phase-2 (shrinkage/reframe)
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
| Endogenous successors under parity | **Solid, and larger** — SAN +0.048 [+0.032,+0.064], FAN +0.154 [+0.106,+0.206] vs RevIN +0.016 | §6 |
| Block B: RevIN+cov worse than Raw+cov | **Solid: all 5 means positive on every panel**; CIs exclude zero on 4/5 (cell bootstrap), 3/5 (dataset-clustered), 5/5 (no Load) | §6 |
| CN's ETTh blow-up is calendar overfitting, not extrapolation | **Solid** — 0/20,160 test levels leave the train range; level R² 0.92→−0.27 | §14.1 |
| "0.5 CPU-s screen" / "180 GPU-hours" | **Corrected** — measured 0.036 s; 179 h all rows / 154 h deduped, 42 h of it CPU | §14.2 |
| Operational anchor (0.1053→0.1001, 1.6 MW) | **Solid but re-sourced** — Block E, 5.0% not 4.9%; belongs to level-conditioning, not to the audited toggle (0.2%) | §14.3, §14.5 |
| tCO₂/yr conversion of the reserve delta | **Withdrawn as a claim** — capacity × 8760 h × energy factor is a category error; kept only as an explicit upper bound | §14.5 |
| CondNorm as the best level-conditioner | **False on the operational cell** — ridge dynreg 0.0899 vs CN 0.1001 (but dynreg fails on load/solar) | §14.5 |
| Mechanism: is the cost the *window mean*? | **Partly** — mean channel +0.0104 [+0.0043,+0.0170], 6/6 cells, carries the horizon signature; but only 41%/21% of the total | §15 |
| Dropout backbones are thread-count reproducible only | **KNOWN DEFECT (new)** — ~0.008 MSE spread, same order as the effect | §15.1 |
| Covariate-scaling confound | **Open** — control run (G10, 120 runs) reverses the contrast on wind and is numerically unusable on solar | §15.2 |
| Recalibrated interval coverage | **Central only** — one-sided tails are asymmetric (GEFCom-Wind 0.147/0.958); reserve is an upper-tail decision | §14.5 |
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
| per-zone LPS range | 0.5753–0.7585 (3dp rounding is ambiguous at the top end; print 4dp) |
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

**CAVEAT (added 2026-07-28), and it is a governance problem, not an arithmetic
one.** No script in the repo produces the 11/12 count. It requires reading **Block E
rows** (`paper/tables/tabE_baselines.md`, deterministic arms at fixed L=336) against
**Block A rows** (`tab1_draft.md`, per-cell tuned lookback) — exactly the cross-block
comparison that `tabE_baselines.md`'s own header forbids ("Block A/B 표의 수치와
직접 비교하지 않는다"), while `experiments/g7_blocke.py`'s docstring asserts the
opposite ("directly comparable to them in scale"). The count reproduces *exactly*
(11/12, 10/12, 0/12, 0/12) under that mixed reading, and drops to **9/9 (MSE) /
8/9 (MASE) vs 0/12** without the GEFCom-Load rows (sole MASE exception: ridge
dynreg on GEFCom-Solar, 1.2530 vs 1.2117). Until the two source files are
reconciled, this count must be reported as *indicative*, not as a test — which is
what the workshop paper now does (the number itself was moved out of the body).

MCS (α=0.10): 11 exogenous (dataset,h) cells all retain {CondNorm} alone; standard
cells retain RevIN/SAN/FAN. **Caveat:** MCS returns a *singleton* in 20 of 23 cells —
on a single test path with backbone-averaged pseudo-models this reports a ranking,
not an uncertainty set. Read as a compact ranking, consistent with the DM marks.

---

## 6. Information parity (Block B) — the mechanism

**Setup.** Every arm gets identical past+future covariates, 5 covariate-capable
backbones, exogenous group only. 1,133 runs. `results/g4_covfair_full.csv`.

**Result — isolated cost of the layer, `MSE(RevIN+cov) − MSE(Raw+cov)`**
(**i.i.d. percentile bootstrap over the 11 seed-averaged (dataset,h) cells**,
B=10⁵, `default_rng(0)`; re-verified 2026-07-28):

| backbone | mean | 95% CI (cell bootstrap) | cells>0 | 95% CI (dataset-clustered) |
|---|---|---|---|---|
| LGBM-Cov | +0.0191 | [+0.0117, +0.0268] | 11/11 | [+0.0106, +0.0290] |
| Linear mixer | +0.0205 | [+0.0055, +0.0370] | 8/11 | **[−0.0010, +0.0461]** |
| MLP mixer | +0.0238 | [+0.0119, +0.0392] | 10/11 | [+0.0114, +0.0356] |
| SegRNN-Cov | +0.0112 | [+0.0057, +0.0172] | 10/11 | [+0.0027, +0.0213] |
| **PatchTST-Cov** | **+0.0038** | **[−0.0092, +0.0167]** | 7/11 | [−0.0152, +0.0182] |

Four of five exclude zero; **PatchTST-Cov is null** and is reported as such (not
absorbed into "every backbone"). Magnitudes are hundredths of an MSE unit — one to
two orders below the +0.841 headline gap; across all 55 cells the range is −0.0261
to +0.0849, 46 positive. Small because it is the layer cost alone, not covariate
access.

**The family-scope result (added 2026-07-29).** The parity block contains SAN and
FAN arms as well, and nothing had ever reported them against RAW+cov. Pooled over
backbones and cells (SAN/FAN are inapplicable to LightGBM-Cov, hence n=44):

| arm | cost vs RAW+cov | 95% CI | cells>0 |
|---|---|---|---|
| RevIN | +0.0157 | [+0.0102, +0.0214] | 46/55 |
| **SAN** | **+0.0476** | [+0.0317, +0.0643] | 40/44 |
| **FAN** | **+0.1535** | [+0.1055, +0.2063] | 38/44 |

Per backbone, SAN's interval excludes zero on all four torch backbones
(+0.043/+0.044/+0.051/+0.053) and so does FAN's (+0.157/+0.209/+0.160/+0.088).

**This matters for scope.** The paper's title and abstract say "instance
normalization" while Table 1 tests RevIN alone — an overclaim nobody had caught,
because it is an ML-reviewer catch and no ML reviewer was in the random panel. The
successors that refine *which* endogenous statistic to use lose **more**, not less:
SAN 3x and FAN 10x RevIN's cost. Reporting them turns the family-scope claim from
an unearned generalization into a measured one, and it is the strongest single
number in the block. Now in the body.

**CORRECTION (2026-07-28), wording of the interval.** Earlier drafts of this file
and of the workshop paper called this a *cluster* bootstrap. It resamples the 11
**cells** i.i.d., not the 4 **datasets**. The published CIs reproduce only under
the cell version (lgbmcov / segrnncov / patchtstcov exact; linmix and mlpmix
within bootstrap noise). Under a genuine dataset-clustered bootstrap (4 clusters,
right column above) the linear mixer straddles zero and **"four of five exclude
zero" becomes three of five**. Means are identical under both. Both are now
reported; the paper states the cell version as primary and the dataset-clustered
version as the conservative bound.

**Panel sensitivity (2026-07-28), and where the null lives.**

| panel | cells | cell-boot CIs excluding 0 | note |
|---|---|---|---|
| full | 11 | 4/5 | PatchTST-Cov null (+0.0038) |
| no GEFCom-Load | 8 | **5/5** | PatchTST-Cov +0.0141 [+0.0022, +0.0252], 7/8 |
| no Load, jeju h=24 only | 7 | 3–4/5 | PatchTST-Cov back to null (+0.0117, [−0.0005, +0.0237]); linmix on the boundary ([+0.0000, +0.0453]) |

**CORRECTION (2026-07-28, second pass):** an earlier version of this line said
PatchTST-Cov's *only* negative cells were the three Load cells. That is wrong and
self-contradicting (it would make the 8-cell panel 8/8, not the 7/8 recorded
above). PatchTST-Cov has **four** negative cells: gefcom_load h=24/96/336
(−0.0261 / −0.0215 / −0.0236) **and gefcom_wind h=336 (−0.0184)**. The three Load
cells are its three *largest* negatives, so removing them still leaves one
negative — hence 7/8 — and demoting Load **strengthens** Block B rather than
weakening it. What survives every panel is
the sign: all five backbone means are positive (+0.004…+0.024 on 11 cells,
+0.012…+0.022 on the strictest 7-cell base). 34/40 pairs positive on the no-Load
panel.

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

**Result.** CondNorm has the **best pinball on all 4 exogenous datasets** — the
point-forecast boundary transfers to pinball. (**CORRECTION 2026-07-28:** the
parenthetical "0.135–0.165 vs 0.16–0.32" printed here in earlier passes does not
reproduce. From `g7_blockf.csv` (rlinear_q) CN's per-dataset pinball is 0.0577
solar / 0.0852 load / 0.1349 wind / 0.1649 jeju → **0.058–0.165**; the quoted
0.135–0.165 was the two wind datasets only. The instance-norm comparators span
**0.092** (solar RevIN) to **0.431** (wind FAN), not 0.16–0.32. The qualitative
claim is unchanged and slightly stronger: CN best in 4/4 datasets and 11/11
cells.) **But
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

## 13. Quantile recalibration (G8, 2026-07-28)

Closes Block F's open item (CN interval under-coverage, first-stage uncertainty
unpropagated). Runner `experiments/g8_recal.py` (Block F training mirrored
byte-for-byte; adds val/test prediction export + conformal), ledger
`results/g8_recal.csv`, table `paper/tables/tabG8_recal.md`, unit tests
`tests/test_g8_recal.py`.

- **Method**: per-quantile additive split conformal — δ_k = q_k-quantile of
  validation residuals, added to each predicted quantile, re-sorted
  (rearrangement). Fitted on the val segment (dual use with early stopping,
  disclosed; test untouched), applied identically to every arm.
- **Reproduction control**: the `none` variant reproduces Block F per-dataset
  CN cov80 exactly (0.502/0.576/0.653/0.663 — jeju/wind/load/solar order by
  value: wind 0.502, jeju 0.576, solar 0.653, load 0.663).
- **Primary (rlinear_q, exogenous, 275 runs)**: CN cov80 per dataset
  0.50–0.66 → **0.79–0.83** (nominal 0.80); val cov80
  0.800 by construction. CN mean pinball **improves** 0.1058 → 0.1034. With
  every arm recalibrated, CN keeps best pinball **11/11** (dataset,h) cells.
  RevIN (already calibrated 0.822) essentially unchanged (0.811). Per-step
  offsets add nothing over pooled (mean cov80 0.810 vs 0.811).
- **lgbm_q cross-check (h≤48, 5 cells × 3 arms, disclosed scope cap — Block F
  logs put h=96/336 quantile DMS at 1.3–4.5 h/run)**: same pattern. CN cov80
  0.59–0.71 → **0.80–0.83** (jeju24 .639→.824, jeju48 .633→.826,
  wind24 .586→.803, load24 .673→.795, solar24 .705→.810); CN pinball improves
  in all five cells and stays best-of-arm in all five (e.g. load24
  0.0633 vs winz 0.0942 / raw 0.0976).
- **Weighting conventions (added 2026-07-28) — the primary bullet mixed two.**
  The coverage aggregate `0.598 → 0.811` is the mean of the four **per-dataset**
  means; the cell-weighted form (what `tabG8_recal.md` prints) is
  `0.6005 → 0.8096`. The pinball aggregate `0.1058 → 0.1034` is the mean over the
  **11 cells**; dataset-weighted it is `0.1107 → 0.1082`. Neither is wrong, but any
  document quoting them must say which. Restricted to the 8 non-Load cells the
  pattern holds (CN cov80 mean 0.577 → 0.819, best pinball 8/8).

---

## 15. G9 — the mean-only ablation: is the layer's cost the window mean? (2026-07-29)

**Pre-registration** `evidence/prereg_meanonly.md`, written before the first run
(and honest about the one way it is weaker than G4/G4a: not git-timestamped ahead
of the run). **Runner** `experiments/g9_meanonly.py`, **arm**
`src/norms/revin_mean.py` (invariants in `tests/test_revin_mean.py`), **analysis**
`experiments/g9_meanonly_analyze.py`, **data** `results/g9_meanonly.csv` (180 runs)
and `results/g9_meanonly_cells.csv`.

**Why.** §2/Fig. 1 of the workshop paper blame the *window mean*; Table 1 toggles
RevIN, which removes mean, scale and affine together. Every hostile reviewer
simulation named this as the largest hole. The arm removes and restores the window
mean only, so against RAW it differs in exactly one channel:

    RevIN − RAW        total layer cost   (Table 1)
    RevINMean − RAW    mean channel
    RevIN − RevINMean  scale + affine

**Setup.** GEFCom-Wind + GEFCom-Solar (third-party, forecast covariates — neither
the Load reference cell nor Jeju's caveats), linmix + mlpmix, h{24,96,336}, 5
seeds. Same `nn_run`, same frozen lookback per cell, same splits/optimizer/loss
scale as Block B.

**Result (equal weight over 6 cells per backbone, percentile bootstrap).**

| backbone | mean channel | scale+affine | total | mean's share |
|---|---|---|---|---|
| linmix (primary) | **+0.0104 [+0.0043, +0.0170], 6/6** | +0.0152 [−0.0044, +0.0377], 3/6 | +0.0257, 4/6 | **40.7%** |
| mlpmix | +0.0061 [+0.0018, +0.0107], 5/6 | +0.0226 [+0.0057, +0.0463], 5/6 | +0.0287, 6/6 | 21.2% |

By horizon on linmix — the two channels separate cleanly:

| h | mean channel | scale+affine |
|---|---|---|
| 24 | +0.0075 | +0.0214 |
| 96 | +0.0093 | +0.0187 |
| 336 | **+0.0146** | **+0.0056** |

**Pre-registered endpoints.** PRIMARY 1 (mean channel > 0 on both backbones)
**CONFIRMED**. PRIMARY 2 (mean carries ≥50%) **NOT CONFIRMED** — 41% / 21%.
SECONDARY 3 (|mean| > |scale+affine|) **NOT CONFIRMED**. SECONDARY 4 (mean channel
grows with horizon) **CONFIRMED** on the bit-reproducible backbone.

**Verdict — what the paper may now say.** The stale-mean mechanism is **real and
isolated**: positive in 6 of 6 cells with an interval excluding zero, and it is the
channel that carries the *horizon* signature the theory predicts (grows +0.008 →
+0.015 while the scale channel's cost falls +0.021 → +0.006). But it is **not the
whole cost**, and at short horizons it is the minority: the frozen instance *scale*
is the larger, if sign-unstable, half. On GEFCom-Solar the instance scale actually
**helps** at every horizon (−0.016 to −0.003) while the mean channel still costs —
the two channels are doing different things and should no longer be described as
one. §2's attribution was over-stated in the direction the pre-registration
anticipated (case 2), and the manuscript now says so.

### 15.1 A reproducibility limit this ablation exposed (applies to Table 1)

Re-running the parity block's RAW and RevIN cells inside G9 reproduces **linmix
bit-exactly** (0.247848 / 0.320192 to all printed digits) and **mlpmix not at all**
(e.g. gefcom_wind h=336 RAW: frozen 0.343108 vs rerun 0.294098).

Root cause, verified directly: **PyTorch samples CPU dropout masks per intra-op
thread chunk**, so a backbone with dropout is reproducible only at a fixed thread
count. Same cell, same seed, varying `torch.set_num_threads`:

| threads | 28 | 16 | 8 | 4 | 1 |
|---|---|---|---|---|---|
| mlpmix RAW mse | 0.23983 | 0.24082 | 0.24152 | 0.23320 | 0.23966 |

A spread of ~0.008 MSE — **the same order as the layer effect being measured**.
linmix (no dropout) is invariant (0.24801525 at 4 and 28 threads).

**Consequences.**
1. G9's primary panel is linmix; mlpmix is reported as internally consistent only.
2. **Table 1's three dropout-bearing backbones (mlpmix, segrnncov, patchtstcov)
   have intervals conditional on the thread configuration of the original run.**
   This was not previously stated anywhere and is now disclosed in the paper.
3. Any future re-run of Block B must pin `torch.set_num_threads` to be comparable.
   Recommend adding it to the environment contract.

### 15.2 G10 — the covariate-scaling control, and why it failed to close the confound

**Runner** `experiments/g10_covscale.py`; **data** `results/g10_covscale.csv`
(120 runs). Adds `cov_instance_norm` to `nn_run` (default off, so the frozen parity
path is unchanged — verified: the RAW and RevIN arms reproduce the frozen block to
2e-7).

**Question.** Covariates are globally z-scored in every arm while RevIN rescales
only the target window, so part of the measured layer cost could be a units
mismatch. Control: window-normalize the covariates too, with their own lookback
statistics. Four arms (raw, revin, raw_cn, revin_cn) on linmix, both GEFCom sets,
h{24,96,336}, 5 seeds.

**Result — it does not close it.**

| dataset | h | raw | revin | raw_cn | revin_cn | baseline | matched |
|---|---|---|---|---|---|---|---|
| wind | 24 | 0.2478 | 0.3202 | 0.2871 | 0.2289 | **+0.072** | **−0.058** |
| wind | 96 | 0.2503 | 0.3070 | 0.3513 | 0.2509 | **+0.057** | **−0.100** |
| wind | 336 | 0.3809 | 0.4185 | 0.4411 | 0.3642 | **+0.038** | **−0.077** |
| solar | 24 | 0.1611 | 0.1465 | **1.79** | 0.3163 | −0.015 | −1.47 |
| solar | 96 | 0.1463 | 0.1456 | **23.4** | 6.678 | −0.001 | −16.7 |
| solar | 336 | 0.1923 | 0.1950 | **788** | 141.3 | +0.003 | −646 |

Two findings, both against us:
1. **On the wind sets the contrast reverses.** Put both on the same footing and the
   layer stops costing and starts helping (+0.072 → −0.058 at h=24). The units
   mismatch is therefore a live part of the measured effect, not a dismissable one.
2. **On solar the control is unusable as implemented.** Night windows have
   near-zero covariate variance, so dividing by the window std explodes the RAW
   arm (788 MSE at h=336). This is a defect of the control (needs a variance
   floor), not of the data.

**Consequence for the paper.** The confound stays open and is now stated in the
body with the control's outcome, not merely as a possibility. Do not describe it as
closed. Next steps if anyone picks this up: variance-floored control on solar, and
a decomposition that separates the *footing change* (raw → raw_cn, which is itself
large and adverse) from the *layer toggle* — the current design confounds the two.

### 15.3 A confound G9 narrows but does not close

Covariate channels are globally z-scored on train statistics in every arm, while
RevIN rescales only the target window. Part of the measured layer cost could be a
units mismatch rather than level handling. The mean-only arm changes the target's
location without touching its scale and still costs, which bears on it, but does
not settle it. **Superseded by §17: the confound was real, it was measured, and
the headline it threatened has been withdrawn.**

---

## 16. G12 — where the audited cost lands: ramp-conditional decomposition (2026-07-29)

**Pre-reg** `evidence/prereg_ramp_footing.md`. **Runner** `experiments/g12_ramp.py`.
**Data** `results/g12_ramp.csv` (per bin), `results/g12_ramp_cells.csv` (per cell).
**No model was trained**: the per-origin losses have been on disk since Block B.

**Why.** The paper's climate pathway asserted that the stale window mean goes
wrong "exactly at the ramps whose errors fossil-fuelled reserves absorb". That
had never been measured. Under `docs/philosophy.md` it becomes a number or it
leaves the paper.

**Ramp statistic, fixed before any loss array was opened.**
`ramp(t) = max_{k} |y[t+L+k] − y[t+L+k−1]|`, the largest realized one-hour change
inside the horizon. Deliberately *not* the staleness quantity — it never
references the lookback window, the window mean, or a model output, so the test
cannot be circular. Equal-count terciles by rank within each (dataset, backbone,
h) cell (value-cut terciles leave an empty bin on the solar set, where a large
share of origins are all-night with ramp exactly 0); ties are broken by origin
index, which is arbitrary but independent of every arm.

**PRIMARY — confirmed.** Linear mixer, seven clean cells (GEFCom-Wind ×3,
GEFCom-Solar ×3, Jeju h=24), equal cell weight:

| ramp tercile | layer cost | RAW arm's own MSE | cost as a share of it |
|---|---|---|---|
| bottom | +0.0139 | 0.184 | 8.0% |
| middle | +0.0185 | 0.224 | 6.9% |
| **top** | **+0.0335** | 0.319 | 9.4% |

`hi − lo = +0.0196 [+0.0044, +0.0332]` (percentile bootstrap over cells,
B=10⁵) — **2.4× larger in the top ramp tercile**, interval excludes zero.

**And the honest qualifier, which the paper must print next to it.** The
concentration is **absolute, not relative**: as a share of each bin's own error
the cost is flat (`hi − lo = +0.013 [−0.092, +0.099]`, 4/7 positive), because
the high-ramp bin is harder for every arm (RAW 0.184 → 0.319). Reserve is sized
in absolute units, so the absolute reading is the operative one — but a referee
will compute the ratio, and we print it first.

**Per cell — this is a wind phenomenon.**

| cell | cost by tercile (lo/mid/hi) |
|---|---|
| GEFCom-Wind h=24 | +0.065 / +0.041 / **+0.111** |
| GEFCom-Wind h=96 | +0.053 / +0.023 / **+0.094** |
| GEFCom-Wind h=336 | +0.036 / +0.060 / +0.017 (94 origins, 76% tied at the median ramp) |
| Jeju Wind h=24 | −0.018 / +0.015 / +0.002 |
| GEFCom-Solar h=24/96/336 | negative or near-zero throughout |

Solar contributes negatives (RevIN beats RAW there at this footing), so the
pooled result is carried by wind. The abstract's "on wind, solar and load"
phrasing is not supported and is withdrawn.

**SECONDARY 1 (monotone across three terciles) — failed**, 1/7 cells. The middle
bin is not ordered; only the extremes are.

**SECONDARY 2 (≥3 of 5 backbones) — confirmed in sign, 4/5.** linmix +0.0196
[+0.0044,+0.0332], segrnncov +0.0139 [+0.0041,+0.0251], lgbmcov +0.0144
[−0.0021,+0.0333], patchtstcov +0.0071 [−0.0394,+0.0500], **mlpmix −0.0195
[−0.0450,+0.0031] — reverses**, driven entirely by its three wind cells
(−0.073/−0.022/−0.062) while its solar cells are positive. Reported, not
smoothed.

**SECONDARY 3 (do the bins mean what they should?) — confirmed, and it is the
most useful number in the block because it involves no model at all.** Level
error on the same origins, global-z units, seven clean cells:

| ramp tercile | window mean (what RevIN restores) | covariate-implied level | gap |
|---|---|---|---|
| bottom | 0.303 | 0.083 | 0.220 |
| middle | 0.340 | 0.087 | 0.253 |
| top | 0.369 | 0.105 | **0.263** |

Both degrade, the window mean degrades more, and the absolute advantage of the
covariate-implied level grows by 20% from the bottom tercile to the top. The
effect is concentrated at the long wind horizons (GEFCom-Wind h=96: gap
0.177 → 0.422; h=336: 0.059 → 0.157) and is absent on solar, whose ramps are
diurnal and whose multi-day window mean already averages over them.

**Verdict.** The climate pathway is now a measurement rather than a mechanism
story, at wind horizons, in absolute units, with the relative-flatness caveat
printed alongside. This is the block that earns §4.

---

## 17. G11 — covariate footing: the pre-registered test that killed the headline (2026-07-29)

**Pre-reg** `evidence/prereg_ramp_footing.md`, including a same-day amendment
(recorded, not silent) that replaced the PRIMARY endpoint with a strictly harder
one after a smoke run showed the first version was itself misspecified.
**Runner** `experiments/g11_footing.py`, **analysis** `experiments/g11_analyze.py`,
**data** `results/g11_footing.csv`.

**Why.** G10 was reported as "a control that failed to close the confound". That
ruling was wrong. G10 window-normalizes the covariates *with their own lookback
statistics*, which destroys the covariate level — the exact signal this project
says instance normalization discards. It changes the units and the information at
once and can adjudicate neither. A misspecified control is a finding, not an
excuse (`docs/philosophy.md` §2).

**Four covariate footings**, series globally z-scored on train statistics first;
`s` = the per-window target standard deviation, the divisor RevIN uses:

| code | covariate input | covariate level | per-instance units |
|---|---|---|---|
| G global | `cov_z` | kept | global (mismatched to a RevIN target) |
| W window | `(cov − mean_w)/sd_w` | **destroyed** | own (= G10, = CrossLinear) |
| Wf window, floored | as W, `sd_w ← max(sd_w, 0.1)` | destroyed | own (readable on solar) |
| S scale | `cov_z / s` | kept | the target's |

The (raw, G) and (revin, G) cells reproduce the frozen parity block, which is the
implementation check (`--verify`).

**Result — linear mixer, six GEFCom cells** (Jeju and the CondNorm-at-other-footings
rows were still running when this entry was written; the table is regenerated by
`experiments/g11_table.py`, which is what the paper `\input`s).

| cell | RAW (global) | CN | toggle @ global | toggle @ window† | toggle @ scale |
|---|---|---|---|---|---|
| GEFCom-Wind h=24 | 0.248 | 0.177 | **+0.072** | −0.058 | −0.024 |
| GEFCom-Wind h=96 | 0.250 | 0.167 | **+0.057** | −0.100 | −0.033 |
| GEFCom-Wind h=336 | 0.381 | 0.199 | **+0.038** | −0.077 | +0.006 |
| GEFCom-Solar h=24 | 0.161 | 0.064 | −0.015 | −0.017 | −0.022 |
| GEFCom-Solar h=96 | 0.146 | 0.061 | −0.001 | −0.027 | −0.036 |
| GEFCom-Solar h=336 | 0.192 | 0.066 | +0.003 | −0.169 | −0.074 |
| **mean** | 0.230 | 0.122 | **+0.026** [+0.002,+0.052] | −0.075 [−0.118,−0.037] | −0.030 [−0.050,−0.012] |

† variance-floored. Unfloored, the RAW arm reaches 1.79 / 23.4 / 787.9 on the three
solar cells; those numbers are in `paper/tables/tab_footing.md` and not pooled
anywhere.

**PRIMARY (amended) — NOT CONFIRMED.** `min_f RevIN_f − RAW_global = −0.0142
[−0.0195, −0.0076]`, positive in 1/6 cells. The originally-registered endpoint
fails too: `RevIN_scale − RAW_scale = −0.0303 [−0.0504, −0.0119]`, 1/6.

**Consequence, executed in the manuscript without negotiation** (this is what the
pre-registration bought): the title changed, the abstract's unconditional penalty
claim is gone, the §3 heading "It is the layer, not the information" is gone, and
the parity block is demoted from Table 1 to an appendix paragraph explicitly
labelled as the thing being withdrawn. **The measured accuracy penalty of instance
normalization under covariate parity is footing-dependent and this project no
longer reports it as a portable number.**

**SECONDARY 1 — CONFIRMED, 6/6.** Best endogenous configuration over all four
footings versus CondNorm, per cell: gaps +0.074 / +0.073 / +0.112 (solar 24/96/336)
and +0.051 / +0.083 / +0.165 (wind 24/96/336); mean +0.095. **The boundary is
footing-invariant even though the toggle is not.** Note the honest limit: in 3 of 6
cells the largest toggle magnitude exceeds the CN gap, so "wins by more than the
toggle ever moved" is false and was corrected in the manuscript to the actual range.

**SECONDARY 2 — CONFIRMED, and it is the new headline.** Destroying the covariate
level (window-floored minus global), per cell:

| arm | wind 24 / 96 / 336 | solar 24 / 96 / 336 |
|---|---|---|
| RAW | **+0.039 / +0.101 / +0.060** | −0.006 / +0.062 / +0.422 |
| RevIN | **−0.091 / −0.056 / −0.054** | −0.009 / +0.036 / +0.251 |

On the wind cells the asymmetry is exactly as predicted: removing the covariate
level costs the globally scaled arm and *helps* the instance-normalized arm, which
was not using it. On solar both arms degrade, because the floored window
normalization is still close to degenerate there; the manuscript therefore scopes
this claim to the wind cells.

**Implementation note.** `nn_run(..., cov_footing=...)` in
`experiments/g4_covfair_full.py`; `cov_instance_norm=True` is retained as the legacy
spelling of `"window"` so the G10 rows still reproduce, and the default keeps the
frozen parity path byte-identical.

### 17.1 Round-1 review forced two analyses that changed what the block means → REVIEW_LOG.md
### 17.2 Round-2 review: the level manipulation was two-factor → REVIEW_LOG.md

---

## 18. G13 — is the boundary an implementation oversight? (2026-07-29)

**Pre-reg** `evidence/prereg_ramp_footing.md` (§G13, with the smoke seed disclosed).
**Runner** `experiments/g13_stats.py`.

**Why.** The round-1 ML reviewer named the one experiment that could collapse the
whole framing: if the structural claim is that `yhat = ybar + s·f(·)` never exposes
`ybar` or `s` to the backbone, then hand the backbone those two numbers. If that
closes the gap to covariate-conditioned level handling, there is no boundary, only an
omitted input. We had not run it.

**Result (70 runs, linear mixer, 7 clean cells).** Share of the RevIN-to-CondNorm
gap closed by handing the backbone `ybar` and `s` as extra input channels:

| cell | RevIN | RevIN+stats | CN | gap closed |
|---|---|---|---|---|
| GEFCom-Wind h=24 | 0.320 | 0.319 | 0.177 | 1.0% |
| GEFCom-Wind h=96 | 0.307 | 0.291 | 0.167 | 11.4% |
| GEFCom-Wind h=336 | 0.419 | 0.367 | 0.199 | 23.4% |
| GEFCom-Solar h=24/96/336 | | | | 1.2 / 2.0 / 7.1% |
| Jeju Wind h=24 | 0.316 | 0.316 | 0.295 | −0.6% |

**PRIMARY: +6.5% [+1.5%, +13.0%]** against a pre-registered withdrawal threshold of
50%. **Not an omitted input.** The extra channels do help a little
(`revin+stats − revin` = −0.0115), and the control confirms the endpoint is
interpretable (`raw+stats − raw` = −0.0047, so the channels are not simply harmful),
but the boundary is essentially untouched. The largest closure is at the longest
horizon, which is where the frozen statistic is most stale and therefore where
knowing it helps most — consistent with the mechanism rather than against it.

This is the experiment a round-1 reviewer named as the one that could collapse the
framing. It did not.

---

## 19. Round-3 review, and the two corrections it forced (2026-07-29) → REVIEW_LOG.md

---

## 20. G15 — the interval width, which is what a reserve decision consumes (2026-07-29)

**Why.** The round-3 energy practitioner pointed out that the paper kept saying "the
width delta is the number still missing" while owning a quantile backbone,
split-conformal recalibration and per-arm coverages — i.e. it was one metric away
from the only physical number it could honestly report. Added `width80` and
`upper_margin` to `q_metrics` and re-ran `rlinear_q` at h=24 (70 runs, own output
file so the frozen `g8_recal.csv` header is untouched).

**Result** — mean 80% interval width in global-z units after identical
split-conformal recalibration of every arm, coverage in brackets:

| dataset | RevIN | CondNorm | narrower by |
|---|---|---|---|
| GEFCom-Wind | 2.215 [0.761] | **1.116** [0.822] | **50%** |
| GEFCom-Solar | 0.693 [0.801] | **0.435** [0.820] | 37% |
| GEFCom-Load | 1.072 [0.820] | **0.535** [0.788] | 50% |
| Jeju Wind | 1.827 [0.816] | **1.404** [0.821] | 23% |

At equal or better coverage the covariate-conditioned interval is 23–50% narrower.
Coverage is location; width is what gets procured. Not pre-registered: a metric
added to an existing block in response to review, and labelled as such.

---

## 14. Verification pass 2026-07-28 — corrections that touch the paper's claims

Four independent re-derivations from the frozen CSVs, run while restructuring the
CCAI workshop paper. Everything below either corrects this ledger or pins down a
number the ledger asserted without a source. Corrections already folded into §§4a,
5, 6, 10, 13 above are not repeated here.

### 14.1 Why CondNorm blows up on ETTh1/ETTh2 — it is NOT extrapolation

The manuscript previously implied the first stage "fits noise"; the sharper and
verified statement is that it **overfits the calendar and cannot extrapolate**:

- **Not out-of-range.** 0 of 20,160 test level values on either dataset leave the
  per-channel train-y range (max excursion 0.0000 train-y sd) — as LightGBM leaf
  averaging requires. No clipping is needed and none would help.
- **Generalization failure.** Level R² 0.9162 → **−0.2675** (etth1) and 0.9494 →
  **−0.0635** (etth2), in-sample → test, pooled on the global-z scale (the same
  scale as the reported MSE). `corr(level, y)` collapses +0.958 → +0.278 and
  +0.975 → +0.184.
- **Two different routes.** etth1 is variance-dominated (test level MSE 1.4081 =
  bias² 0.394 + var 1.015; `sd(level)/sd(y)` 0.920 → 1.158); etth2 is
  bias-dominated (3.3386 = 2.843 + 0.495, from a 1.295 train-sd level shift a
  fixed calendar function cannot see).
- **Visible without any backbone.** On test rows the covariate level's own error is
  **2.13×** (etth1) and **13.78×** (etth2) the RevIN 96-window-mean level error,
  versus **0.151×** / **0.348×** on gefcom_wind / jeju_wind. The boundary is a
  property of the level layer alone.
- **Convention matters.** Per-channel-averaged R² is much more negative than
  pooled global-z (etth1 −1.2993 vs −0.2675; weather −4.1259 vs −0.0152), inflated
  by channels with near-constant test segments. Use pooled global-z, and say so.
- **A positive LPS is not a promise about the pointwise level.** weather has
  LPS +0.110 but pointwise test level R² −0.0152. Different estimands (LPS scores
  window *means* under expanding-window CV).
- **Matched-cell rule is load-bearing.** §9's α=1 column (2.751 / 8.781) reproduces
  *only* under the pre-registered A.6a rule (seed mean within each (backbone,h)
  cell, equal weight over the 6 cells; `experiments/g8_tier_verdict.py`). Pooling
  all 18 matched rows equally gives **3.2635 / 11.4790** instead, because
  rlinear-CN rows are far worse than lgbm_dms-CN rows.

### 14.2 Compute accounting — the "180 GPU-hours" and "0.5 CPU-s" claims

- `sum(wall_s)` over **all 3,117 rows** of `g4_grid.csv` = 646,026.5 s = **179.45 h**
  (this is what "≈180 hours" means). Over the **3,069 unique-key rows** that make up
  the reported 1,794 + 1,275 runs it is 553,640.9 s = **153.79 h**; the difference is
  48 superseded duplicate-key re-runs. State which one you mean.
- **42.34 h of the deduped total (27.5%, 69 rows) is `lgbm_dms` — CPU LightGBM, not
  GPU.** "on a single consumer GPU" should read "on one workstation (a consumer GPU
  plus CPU LightGBM)"; "GPU-hours" is wrong as a unit for the total.
- **The 0.5 CPU-second LPS cost was never logged** (no timing column in any lps CSV,
  no instrumentation in `compute_lps_official.py` / `src/theory/lps.py`). Re-measured
  on one pinned core (`taskset -c 0`, OMP/MKL=1): **0.0359 s median** (0.0347–0.0431,
  n=7), 0.048 s including load + featurize. The old figure was a conservative guess
  that overstated the cost ~14×; the true contrast with the validating study is
  **~7 orders of magnitude**, not six.
- Run accounting 1,794 + 1,133 + 1,275 = 4,202 and the 450 graded-zone runs
  reproduce exactly.

### 14.3 The operational anchor is a Block E quantity, and its arithmetic

- `0.1053 → 0.1001` is **not** in `g4_covfair_full.csv` (that file has no MAE column
  at all). It is the `nmae` column of **`results/g7_blocke.csv`**, jeju_wind h=24,
  arms `linmix_revin` vs `linmix_condnorm`, 5-seed means **0.10533780** and
  **0.10005140**, all 5 seeds improving. Both arms do receive identical past+future
  covariates (same `MixBackbone.forward(x, cp, cf)`), so the *information-parity
  property* holds — but Block E feeds exog **+ 6 calendar harmonics** where Block B
  feeds exog only, and `g7_blocke.py` forbids comparing the two blocks' numbers
  (jeju h=24 linmix MSE: Block B 0.3157/0.2951 vs Block E 0.3402/0.3040).
- `nmae = mae(global-z) × sd_train / max_train(y) = mae × 58.995317 / 241.442`.
- **The percentage is rounding-path dependent:** the printed 0.1053 → 0.1001 gives
  4.94%, the underlying means give **5.019%**. Quote **5.0%**.
- **Reserve arithmetic:** Δnmae × capacity = 0.0052864 × 300 MW = **1.586 MW**.
  It is *not* 4.94% of nameplate (that would be 14.8 MW). The 300 MW fleet is
  hypothetical — the nMAE denominator is the train-split max, **241.442 MW**, on
  which the same Δ is 1.276 MW.
- **Emissions:** 1.586 MW × 8760 h × 0.5 tCO₂/MWh = **6,947 tCO₂/yr** (6,808–11,531
  across the cited 0.49–0.83 range). "A few thousand" understated it ~2×; say
  **≈7,000 tCO₂/yr**, and say explicitly that the chain runs through reserve
  *capacity* held year-round, not through fleet energy (the fleet-energy reading
  would give ~20,000 tCO₂/yr and is a different claim).

### 14.4 Two internal contradictions in §11 to fix on the next pass

- §11 item 2 calls jeju h=48 "the only real-data Prop 3 test", contradicting item 1
  of the same list (GEFCom wind/solar h=96/336 are declared un-compromised).
  Recomputed, the horizon-widening prediction holds monotonically on all three
  GEFCom datasets (wind +0.63/+0.90/+0.99, load +0.086/+0.251/+0.373, solar
  +0.108/+0.130/+0.142), so dropping jeju h=48 does **not** remove the real-data
  horizon evidence.
- MASE for weather is off in the third decimal (recomputed denominator 0.4016 gives
  seasonal-naive 0.752 / RevIN 0.531 vs the printed 0.754 / 0.532), most likely
  per-channel vs pooled averaging on a 21-channel dataset. Changes no comparison.

### 14.5 Adversarial reviewer pass on the restructured manuscript (2026-07-28) → REVIEW_LOG.md
### 14.6 Blind cold read of the final PDF (2026-07-29) → REVIEW_LOG.md
### 14.9 External fact-check of the novelty claims (2026-07-29) → REVIEW_LOG.md
### 14.8 Random CCAI reviewer panel (2026-07-29) → REVIEW_LOG.md
### 14.7 Effect of demoting GEFCom-Load to a reference cell (all verified) → REVIEW_LOG.md

---

*Consolidated 2026-07-24; graded-LPS (§4a) added 2026-07-27; G8 recalibration
(§13) added 2026-07-28; verification pass (§14, with corrections folded into §§4a,
5, 6, 10, 13) added 2026-07-28. Numbers re-derived from frozen CSVs during each
pass; where this file and an older document disagree, this file is authoritative.*


---

## 21. Round-4 review — four claims found false against our own tables (2026-07-29) → REVIEW_LOG.md
