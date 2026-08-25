# Pre-registration — G9, the mean-only ablation

**Written before any G9 run.** Status of this document: it is a genuine
before-the-fact record, but unlike the main grid (`cab17c1`) and the zone study
(`5f888f3`) it is **not git-timestamped ahead of the run** unless it is committed
before `experiments/g9_meanonly.py` is executed. Say so wherever it is cited; do
not describe G9 as pre-registered in the same sense as G4.

## Why

The manuscript's mechanism (§2, Fig. 1) attributes the cost of the normalization
layer to one thing: the lookback **window mean** is a frozen, stale forecast of
the level, and it is stale exactly at a ramp. The measurement (Table 1) toggles
**RevIN**, which removes the window mean, the window scale *and* a learnable
affine together. The appendix already concedes this — "the attribution is
motivated rather than isolated" — and a reviewer who notices it can question the
causal reading of the whole audit.

One arm closes it. RevIN decomposes as

    RevIN − RAW        = total layer cost                (what Table 1 measures)
    RevINMean − RAW    = the mean channel alone          (what §2 claims is the cause)
    RevIN − RevINMean  = scale + affine                  (the remainder)

## Design (fixed here, before running)

- **Arms**: `raw`, `revin_mean`, `revin`. `revin_mean` removes and restores only
  the window mean (`src/norms/revin_mean.py`, invariants pinned in
  `tests/test_revin_mean.py`). The series is already globally z-scored on train
  statistics, so `revin_mean` differs from `raw` in exactly one channel.
- **Datasets**: `gefcom_wind`, `gefcom_solar`. Chosen a priori, not by outcome:
  both are third-party with competition-provided target-time forecast
  covariates, so neither carries the GEFCom-Load reference-cell problem
  (realized temperature) nor Jeju's development-set and h=48 band caveats. Two
  different physical drivers.
- **Backbones**: `linmix` (the linear function class the M1 theory is matched
  to) and `mlpmix` (the backbone with the largest measured layer cost in the
  parity block, +0.024). Both chosen before seeing any G9 number.
- **Horizons** {24, 96, 336} × **5 seeds** = 6 (dataset, h) cells, 90 runs per
  arm, 270 total.
- **Everything else identical to the information-parity block**: same runner
  (`nn_run` from `experiments/g4_covfair_full.py`), same lookback per
  (dataset, backbone, h) read from the frozen `results/g4_covfair_full.csv`,
  same splits, same optimizer, same early stopping, same global-z loss scale.

## Endpoints and predictions

**PRIMARY.** Pooling the 6 cells per backbone, with `Δmean = MSE(revin_mean) −
MSE(raw)` and `Δtotal = MSE(revin) − MSE(raw)`:

1. `Δmean > 0` in mean on **both** backbones.
2. `Δmean / Δtotal ≥ 0.5` on **both** backbones — the mean channel carries the
   majority of the measured layer cost.

**SECONDARY.**

3. `|Δmean| > |Δtotal − Δmean|`: the mean channel is larger in magnitude than
   the scale+affine remainder.
4. `Δmean` grows with horizon (the staleness argument is a horizon argument).

**CONTROL.** The re-run `raw` and `revin` cells must reproduce
`results/g4_covfair_full.csv` to within seed-level float noise. If they do not,
the run path is not the parity block's and nothing here is comparable to
Table 1; fix that before reading any endpoint.

## What each outcome means for the paper — decided now

- **1 and 2 hold** → §2's attribution is isolated, not merely motivated. The
  appendix caveat is replaced by the measurement, and the mechanism sentence in
  the body can be stated causally.
- **1 holds, 2 fails** (mean channel positive but a minority of the cost) → the
  honest statement is that the mean channel is real but *not* the whole story,
  and §2 must say the layer's cost is shared between the frozen mean and the
  frozen scale. Figure 1 stays, its caption gains a qualifier.
- **1 fails** (`Δmean ≈ 0` or negative while `Δtotal > 0`) → the paper's stated
  mechanism is **wrong**, the cost lives in the scale/affine channel, and §2 and
  Fig. 1 must be rewritten around window *scale* rather than window mean. This
  is the outcome that would hurt, and it is the reason to run the arm rather
  than argue the point.

No outcome licenses dropping the arm from the write-up.

## Analysis rule

Seed-mean within each (dataset, backbone, h) cell, then equal weight over the 6
cells per backbone — the same rule as the parity block. Report per-backbone
means with a percentile bootstrap over cells, and report the per-cell table in
full so the pooling cannot hide a sign flip.
