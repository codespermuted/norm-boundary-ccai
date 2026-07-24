# Pre-registration — graded-LPS (GEFCom-Wind zone disaggregation)

**Date:** 2026-07-24. **Status:** per-zone LPS computed and committed BEFORE the
forecasting grid (the LPS is a pre-training quantity; the grid MSEs are the
outcome). Mirrors the original G4 protocol (LPS committed before the grid).

## Question

Our aggregated 8-dataset panel is bimodal in LPS, so the decision rule is validated
only as a two-regime **classifier**. The open question (paper's §3 / Limitations):
does LPS order the **magnitude** of the instance-normalization penalty, or only the
side? Across the 4 aggregated exogenous datasets the within-regime Spearman is
**negative** (ρ = −0.75), but those are 4 heterogeneous, high-leverage points
(wind/load/solar have different covariate structures). GEFCom-Wind is a mean over
10 wind zones with genuinely different NWP skill; disaggregating gives a
**homogeneous** family (same physical process, same 2-covariate structure) with a
graded LPS spread, which is the clean test the aggregated data cannot provide.

## Design (frozen)

- **Series:** the 10 GEFCom2014 Wind zones (Task 15), each y = capacity-normalized
  power, covariates = forecast wind speed at 10 m / 100 m (competition-provided,
  valid at target time; no reanalysis, within the availability envelope → no
  lead-matching defect). Splits 60/20/20 chronological.
- **LPS:** frozen protocol (w = 96, LightGBM, expanding CV, min_train_frac 0.4) —
  identical to `experiments/compute_lps_official.py`.
- **Arms:** {Raw, RevIN, CondNorm} on RLinear (5 seeds) and {winz, Raw, CondNorm}
  on LightGBM-DMS (deterministic). Horizons {24, 96, 336}. Lookback fixed at
  L = 336 across arms and zones (arms differ only in normalization). First stage =
  train-only LightGBM on target-time covariates. All identical to the main grid.
- **Runner:** `experiments/graded_lps.py` (committed with this file, before `--grid`).

## Per-zone LPS (committed; `results/graded_lps_lps.csv`)

| zone | LPS | zone | LPS |
|---|---|---|---|
| z02 | 0.575 | z01 | 0.701 |
| z08 | 0.630 | z06 | 0.728 |
| z09 | 0.631 | z05 | 0.747 |
| z10 | 0.643 | z07 | 0.751 |
| z03 | 0.699 | z04 | 0.758 |

Range 0.575–0.758 (aggregated gefcom_wind = 0.744). All above τ = 0.3, so this
tests magnitude-ordering **within** the exogenous regime — exactly where the
aggregated ρ was negative.

## Predictions (committed before the grid)

- **PRIMARY (magnitude / the dial question).** M1 predicts the gap increases with λ:
  `MSE_In − MSE_Cn = [s_x²(h) + hσ_u² + σ_z²/w] − [(1−λ)V + σ_Δ² + σ_est²]`, whose
  `−(1−λ)V` term rises with λ. So we predict **Spearman(LPS, RevIN−CondNorm gap) > 0**
  across the 10 zones (per-zone gap = mean over backbones × horizons × seeds).
  This is a genuine test: the aggregated cross-dataset ρ was −0.75, so a positive
  within-family ρ would show the aggregated negative was a heterogeneity artifact and
  that LPS **is** a dial within a homogeneous family; a flat/negative ρ leaves LPS a
  classifier. **We commit to reporting ρ as-is either way** (no re-selection).
- **SECONDARY (sign rule).** All 10 zones have LPS > τ, so the rule predicts
  CondNorm-wins on all: RevIN − CondNorm gap **> 0 on 10/10 zones**.
- **SECONDARY (the layer, under RLinear parity).** RevIN − Raw gap **> 0** on the
  zones (instance normalization worse than global scaling), consistent with Block B.
- **Horizon (auxiliary).** The gap grows with h within each zone (Prop 3).

## Interpretation rules (fixed before results)

- If PRIMARY ρ > 0 and significant: report "LPS orders the magnitude within a
  homogeneous family; the aggregated negative ρ reflects cross-dataset heterogeneity"
  — promotes LPS from classifier toward dial (paper's largest gap, closed in the
  favorable direction).
- If PRIMARY ρ ≈ 0 or < 0: report "even within a homogeneous graded family LPS does
  not order magnitude" — LPS stays a classifier; the paper's honest scoping is
  confirmed, not weakened. Either outcome is publishable and reported as-is.
- No dataset/zone is dropped; no estimator is swapped after seeing results.

## Governance

Committed at the commit that adds this file + `experiments/graded_lps.py`, with the
per-zone LPS already computed (pre-training quantity) but **before any forecasting
run**. `results/graded_lps.csv` is produced by `--grid` after this commit.
