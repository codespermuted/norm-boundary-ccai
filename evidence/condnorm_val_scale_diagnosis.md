# CondNorm validation-MSE scale artifact — diagnosis

**Date:** 2026-07-23  **Discovered at commit:** `e5e17c8`  **Frozen tag:** `pre-val-diagnosis`

This note documents a logged-metric anomaly found in the G4 grid results, its
root cause, and an impact analysis. It is committed **before** any code change
or re-run, per the project's pre-registration governance.

## 1. Observation

In `results/g4_grid.csv`, the CondNorm arm's `val_mse` is systematically far
larger than its `test_mse`, across **all eight** datasets (9x–66x). Raw and
RevIN are normal (val/test ≈ 1). Example (rlinear, h=24, seed 0):

| dataset | CondNorm test | CondNorm val | ratio | RevIN ratio |
|---|---|---|---|---|
| jeju_wind | 0.291 | 3.363 | 11.6x | 1.3x |
| gefcom_wind | 0.169 | 3.530 | 20.9x | 0.8x |
| gefcom_solar | 0.060 | 3.936 | 65.6x | 1.0x |
| etth2 | 22.386 | 252.602 | 11.3x | 0.7x |

The pattern is CondNorm-specific and independent of whether CondNorm wins
(energy group) or loses (standard group).

## 2. Root cause — a scale mismatch between the two eval paths

The G4 runner (`experiments/g4_grid.py::torch_run`) evaluates validation and
test on **different scales for CondNorm only**:

- **Validation** (`eval_mse`, ~L232) computes MSE directly on `series`. For
  CondNorm, `series = (resid - mu_r) / sd_r` (~L146) — the **residual-
  standardized (r-space)** series. So `val_mse` is in r-space.
- **Test** (~L280–285) denormalizes the prediction back to the original level
  (`pred_y = pred*sd_r + mu_r + level`) and then re-standardizes to the shared
  **global-z scale** (`(pred_y - mu_g)/sd_g`). So `test_mse` is in global-z.

For Raw/RevIN, `series = (values - mu_g)/sd_g` (~L149) is already global-z, so
val and test share a scale (ratio ≈ 1). CondNorm is the only arm whose val and
test are logged on different scales.

Because a global-z error equals the r-space error times `sd_r/sd_g` (exactly,
per window: `pred_z - true_z = (pred_r - true_r)·sd_r/sd_g`), the expected
inflation of `val_mse` over its global-z equivalent is **`(sd_g/sd_r)^2`**.

## 3. Quantitative confirmation

`sd_g` and `sd_r` are the train-split standard deviations of `y` and of the
first-stage residual. Predicted `(sd_g/sd_r)^2` vs. the observed val/test ratio
(rlinear, h=24, seed 0); univariate datasets are the clean test:

| dataset | C | (sd_g/sd_r)^2 | observed val/test | train R^2 | OOS LPS |
|---|---|---|---|---|---|
| jeju_wind | 1 | 9.1 | 11.6 | 0.890 | 0.745 |
| gefcom_wind | 1 | 18.0 | 20.9 | 0.944 | 0.744 |
| gefcom_load | 1 | 24.7 | 16.7 | 0.960 | 0.894 |
| gefcom_solar | 1 | 56.1 | 65.6 | 0.982 | 0.875 |

On the univariate energy datasets the scale factor reproduces the observed
ratio up to a residual 0.7–1.3x — the genuine val-vs-test difficulty, the same
order as RevIN's own ~1.3x val/test gap. The implied train R^2 (0.89–0.98)
exceeds each dataset's out-of-sample LPS, consistent with mild first-stage
overfit (so `sd_r` is a slight under-estimate). Multivariate datasets
(etth/weather/electricity) are a channel-mean approximation and are not
expected to match exactly, but run the identical code path.
(Reproduce: `scratchpad/val_scale_check.py`.)

## 4. Leakage excluded

The first stage is fit on **train rows only**: `firststage()` calls
`first_stage_level(..., train_end=frame["t1"])` (`experiments/g4_grid.py`
~L108), and `first_stage_level` fits `model.fit(features[:train_end],
y[:train_end])` (`src/norms/condnorm.py:36`). It then *predicts* all t, but the
fit indices `[0, t1)` are strictly disjoint from val `[t1, t2)` and test
`[t2, T)`. The worst-case "test rows leaked into the first-stage fit" scenario
is therefore excluded. (Independent corroboration: the Block E `f/s-only`
column reproduces the CondNorm test numbers through a separate code path.)

## 5. Impact on reported results

`sd_r` and `sd_g` are computed **once** from the train split, before backbone
training (`experiments/g4_grid.py` ~L141, L145), so the r-space→global-z scale
factor is a **per-run constant** across epochs.

- **Exogenous group (jeju_wind, gefcom_wind/load/solar): univariate (C=1).**
  The scale factor is a single positive constant, so
  `argmin_epoch(val_mse_rspace) = argmin_epoch(val_mse_globalz)` exactly. Early
  stopping selects the identical checkpoint it would under correct scaling, and
  `test_mse` is computed in correct global-z. **The four pre-registered
  exogenous sign predictions are provably unaffected.**
- **Standard group (etth1/2, weather, electricity): multivariate.** `val_mse`
  is a channel-mean; r-space weights channels by `1/sd_r_c^2` vs global-z's
  `1/sd_g_c^2`, so early stopping *could* select a marginally different epoch.
  This region is near-optimal, and the standard-group conclusion (CondNorm
  loses decisively; the four "instance-norm wins" signs hold by large margins,
  e.g. ETTh2 CondNorm 6.60 vs 0.29) is robust to a marginal epoch shift. A
  small verification re-run (a few standard-group configs) would close this
  fully.
- `val_mse` is **not reported in the paper** (headline uses `test_mse`), and
  lookback tuning selects on the **RevIN** arm's val only
  (`tune_lookback`, ~L381), never CondNorm's.

**Conclusion.** This is a benign validation-metric logging-scale artifact. It
does not affect any reported `test_mse`, the pre-registered 8/8 sign record,
the MCS results, or `tab:main`. Pre-registration integrity (commit-before-run)
is intact, and the primary results themselves are unaffected.

## 6. Proposed remediation (pending decision — not yet applied)

1. Fix `eval_mse` to denormalize CondNorm's validation prediction to global-z
   before computing `val_mse` (cosmetic: does not change any `test_mse` or,
   for univariate, any early-stopping choice).
2. Add a one-line appendix footnote and a repo/README note explaining the
   `val_mse` column scale, so a reviewer browsing the public MLflow snapshot
   finds the explanation.
3. Optional: targeted verification re-run of a handful of standard-group
   configs to confirm `test_mse` is unchanged under fixed val logging. **A full
   grid re-run is not warranted** — the diagnosis shows the headline numbers
   are already correct.
