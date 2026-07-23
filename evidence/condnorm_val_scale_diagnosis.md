# CondNorm validation-MSE scale artifact — diagnosis

**Date:** 2026-07-23  **Discovered at commit:** `e5e17c8`  **Frozen tag:** `pre-val-diagnosis`

A logged-metric anomaly in the G4 results, its root cause, an impact analysis,
and the audited scope. Committed **before** any code change or re-run, per the
project's pre-registration governance.

## 1. Observation

In `results/g4_grid.csv`, the CondNorm arm's `val_mse` is far larger than its
`test_mse` on **all eight** datasets (9x–66x). Raw and RevIN are normal
(val/test ≈ 1). Example (rlinear, h=24, seed 0):

| dataset | CondNorm test | CondNorm val | ratio | RevIN ratio |
|---|---|---|---|---|
| jeju_wind | 0.291 | 3.363 | 11.6x | 1.3x |
| gefcom_wind | 0.169 | 3.530 | 20.9x | 0.8x |
| gefcom_solar | 0.060 | 3.936 | 65.6x | 1.0x |
| etth2 | 22.386 | 252.602 | 11.3x | 0.7x |

## 2. Root cause — different scales for the two eval paths

For CondNorm only, validation and test are computed on different scales:

- **Validation** computes MSE directly on `series`, which for CondNorm is
  `(resid - mu_r)/sd_r` — the **residual-standardized (r-space)** series.
- **Test** denormalizes to the level (`pred*sd_r + mu_r + level`) and
  re-standardizes to the shared **global-z** scale (`(·-mu_g)/sd_g`).

For Raw/RevIN, `series` is already global-z, so val and test share a scale
(ratio ≈ 1). Since a global-z error equals the r-space error times `sd_r/sd_g`
(exactly, per window: `pred_z - true_z = (pred_r - true_r)·sd_r/sd_g`), the
inflation of `val_mse` over its global-z equivalent is `(sd_g/sd_r)^2`.

**Conversion (univariate, exact):**
`val_mse_globalz = val_mse_rspace · (sd_r/sd_g)^2`.
(Multivariate: a per-channel reweighting — see §5.)

## 3. Quantitative confirmation

`sd_g`, `sd_r` are train-split std of `y` and of the first-stage residual.
Predicted `(sd_g/sd_r)^2` vs. observed val/test ratio (rlinear h24 seed0);
univariate energy datasets are the clean test:

| dataset | C | (sd_g/sd_r)^2 | observed val/test | train R^2 | OOS LPS |
|---|---|---|---|---|---|
| jeju_wind | 1 | 9.1 | 11.6 | 0.890 | 0.745 |
| gefcom_wind | 1 | 18.0 | 20.9 | 0.944 | 0.744 |
| gefcom_load | 1 | 24.7 | 16.7 | 0.960 | 0.894 |
| gefcom_solar | 1 | 56.1 | 65.6 | 0.982 | 0.875 |

The scale factor reproduces the ratio across a 6x span. train R^2 > OOS LPS is
consistent with mild first-stage overfit.

**Residual after scale correction is fully accounted for.** Dividing out
`(sd_g/sd_r)^2` leaves a CondNorm val/test ratio that matches the RevIN control
and tracks the first stage's val-vs-test generalization:

| dataset | CN val/test (corrected) | RevIN val/test | first-stage R²_val | R²_test |
|---|---|---|---|---|
| jeju_wind | 1.27 | 1.26 | 0.632 | 0.697 |
| gefcom_wind | 1.16 | 0.84 | 0.858 | 0.871 |
| gefcom_load | 0.68 | 0.96 | 0.909 | 0.903 |
| gefcom_solar | 1.17 | 0.97 | 0.944 | 0.942 |
| electricity | 0.86 | 0.90 | 0.896 | 0.911 |

Where the corrected ratio exceeds 1 (val harder than test), the first stage
also generalizes better to the test window (R²_test > R²_val, e.g. jeju
0.697 > 0.632, gefcom_wind 0.871 > 0.858, electricity 0.911 > 0.896) — the
residual is genuine first-stage generalization, not a hidden test-favoring
artifact. (Reproduce: `scratchpad/val_scale_check.py`, `firststage_r2.py`.)

## 4. Leakage excluded

The first stage fits on **train rows only**: `firststage()` calls
`first_stage_level(..., train_end=frame["t1"])` (`experiments/g4_grid.py`
~L108) → `model.fit(features[:train_end], y[:train_end])`
(`src/norms/condnorm.py:36`). It predicts all t, but the fit indices `[0,t1)`
are strictly disjoint from val `[t1,t2)` and test `[t2,T)`. The worst case
(test rows in the first-stage fit) is excluded. Independent corroboration: the
Block E `f/s-only` column reproduces the CondNorm test numbers through a
separate path.

## 5. Impact — audited scope

`sd_r`, `sd_g` are computed once from train, before backbone training, so the
r-space→global-z factor is a **per-run constant**. Consequence by runner (every
paper-cited MSE comes from one of these three; all share the same idiom):

| Block | Runner | Target | Early-stopping argmin | Verdict |
|---|---|---|---|---|
| **A** (main grid, `tab:main`, 8/8, MCS) | `g4_grid.py::torch_run` | exog: univariate; standard: multivariate | exog: exact; standard: may shift | exog **provably unaffected**; standard = the only exposure |
| **B** (covariate-fair, `tab:covfair`) | `g4_covfair.py::run_one` | univariate (covariates are input-only) | exact | **provably unaffected** |
| **C/D** (SOTA-MS, `tab:sota`, gaps +0.4122/+0.4282) | `g4_covfair_full.py` | univariate | exact | **provably unaffected** |

For a **univariate** target the scale factor is a single positive constant, so
`argmin_epoch(val_rspace) = argmin_epoch(val_globalz)` exactly: early stopping
selects the identical checkpoint, and `test_mse` is computed in correct
global-z. This covers **all of Block B and C/D, and Block A's exogenous group**
— including all four pre-registered exogenous signs.

The **only** place a reported number could move is **Block A's standard group**
(multivariate target: `val_mse` weights channels by `1/sd_r_c^2`, global-z by
`1/sd_g_c^2`, so a corrected val could pick a different epoch). Within it:

- **etth1, etth2, weather:** CondNorm loses by huge margins (e.g. ETTh2 6.60 vs
  0.29); no epoch shift flips these signs. Robust.
- **electricity is the at-risk cell** and must be checked, not asserted: its
  realized RevIN−CondNorm gap is only **−0.0268** (5x closer to zero than any
  other dataset), it has the **most channels (321)** so the reweighting is
  largest, and it was flagged in the pre-registration as the **lowest-confidence
  sign**. A ~15% CondNorm improvement would flip it. Its corrected val/test
  ratio (0.86, vs RevIN 0.90) shows no aggregate test bias, but epoch-selection
  under reweighting is a separate question resolved only by re-running.

**If electricity's sign flipped**, pre-registration integrity is untouched
(commit-before-run) and 7/8 still clears the pre-registered ≥6/8 gate; the only
loss would be the abstract's "8/8" headline. That bounded downside is exactly
why it is checked before submission, not after.

`val_mse` is **not reported in the paper** (headline uses `test_mse`), and
lookback tuning selects on the **RevIN** arm's val only (`tune_lookback`
~L381), never CondNorm's. Blocks E/F: `f/s-only`/`dynreg` and the LightGBM arms
have no backbone early stopping and are unaffected; Block F (pinball) shares the
`build_frame`/first-stage infrastructure and the same idiom.

## 6. Remediation (pending decision — not yet applied)

**Do not change the training run path.** For the multivariate standard group,
recomputing `val_mse` in global-z would change the selected checkpoint and thus
`test_mse` — so editing the run path without re-running would make the public
repository **fail to reproduce the paper's numbers** (a worse defect than the
val-scale logging in a pre-registered, `make figures`-reproducible study).

1. **Document, don't rewrite.** State in the `val_mse` column comment, README,
   and an appendix footnote that CondNorm's `val_mse` is in residual-standardized
   scale, with the conversion `val_mse_globalz = val_mse · (sd_r/sd_g)^2`
   (univariate). Preserving the as-run code is a stronger integrity signal than
   a silent post-hoc fix.
2. **If a corrected metric is wanted,** put it behind a flag whose default is
   the as-run path, and state that the paper's numbers come from the default.
3. **Verification (necessary, targeted): re-run electricity CondNorm** (all
   horizons, all seeds, torch backbones) with early stopping on the global-z
   (channel-reweighted) validation, into a separate results file, and confirm
   `test_mse` — hence the electricity sign — is unchanged. etth2 need not be
   re-run (6.60 vs 0.29 is unflippable); weather optional. **A full grid re-run
   is not warranted:** everything except Block A's standard group is provably
   unaffected, and the frozen originals stand at tag `pre-val-diagnosis`.
