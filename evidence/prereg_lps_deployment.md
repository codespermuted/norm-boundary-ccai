# Pre-registration and execution plan — deployment-variant LPS

**Status: code complete and tested; the recomputation has NOT been run.**
Every deployment-LPS number in `paper/workshop_ccai.tex` is a
`[TODO: RESULT]` placeholder until the run below is executed and the paper is
updated from the emitted CSVs. Nothing frozen is touched by any step here.

## 1. The defect this addresses

The pre-registered LPS (`experiments/compute_lps_official.py`, commit
`cab17c1`, 2026-07-13) places five evaluation blocks over the final 60 % of the
non-overlapping length-96 windows. Those blocks span the **whole** series,
including rows the forecasting grid later uses as its test segment.

Nothing in the screen consumes a model output, a fitted parameter or a split
boundary, and it is genuinely computed before any forecaster is trained — but
"before training" is not "before the forecast origin", and the paper's
`pre-training screen` framing invites the stronger reading. The stronger
quantity is what a practitioner could actually compute at gate closure, and it
is the one a reviewer is entitled to see.

## 2. What was implemented

| file | change |
|---|---|
| `src/theory/lps.py` | new optional `eval_end` argument. `eval_end=None` is the frozen path, left bit-identical (`test_eval_end_none_is_the_legacy_path`). `eval_end=<first test row>` re-runs the **identical** protocol on the pre-origin prefix — same `w`, same five expanding folds over the final 60 % of the windows that exist there. The return dict gains `variant`, `eval_end`, `first_eval_window`. |
| `experiments/compute_lps_deployment.py` | resolves each series' forecast origin from the grid's own split code and emits both variants side by side, for the eight panel datasets and the ten GEFCom-Wind zones. |
| `tests/test_lps_deployment.py` | 13 tests. The load-bearing one is `test_deployment_ignores_everything_at_and_after_the_origin`: corrupting every post-origin row leaves the deployment score bit-identical while the full score moves. |

The deployment variant is **not** a truncation of the frozen fold boundaries.
Truncating would leave ragged, unequal blocks and change the estimator as well
as the data. Re-running the same protocol on the prefix changes only what is
available, which is the comparison the objection calls for.

## 3. Forecast origins (resolved, `--dry-run`, no model fitted)

Origins are read from `experiments/g4_grid.build_frame` / `graded_lps.zone_frame`,
never re-derived. For the curated sets the grid indexes the full builder frame
while the LPS indexes `longest_contiguous(...)` of it, so the origin is carried
across **by timestamp**.

| series | LPS rows | origin row | origin timestamp | windows full → deployment |
|---|---|---|---|---|
| etth1 | 17 420 | 11 520 | 2017-10-24 00:00 | 181 → 120 |
| etth2 | 17 420 | 11 520 | 2017-10-24 00:00 | 181 → 120 |
| electricity | 26 304 | 21 043 | 2018-11-24 21:00 | 274 → 219 |
| weather | 52 696 | 42 156 | 2020-10-19 19:30 | 548 → 439 |
| jeju_wind | 17 376 | — | 2023-07-04 01:00 | **variants coincide** |
| gefcom_wind | 11 456 | — | 2013-07-10 22:00 | **variants coincide** |
| gefcom_load | 60 600 | 48 480 | 2010-07-14 01:00 | 631 → 505 |
| gefcom_solar | 18 984 | 15 187 | 2013-12-24 20:00 | 197 → 158 |
| gwind_z01…z10 | 16 712–16 800 | 13 369–13 440 | 2013-07-11…07-14 | 174/175 → 139/140 |

**Jeju Wind and GEFCom-Wind need no recomputation.** Their LPS is computed on
the longest contiguous segment, which ends *before* the grid's test segment
begins — Jeju at 2023-06-25 against a test start of 2023-07-04 (the
2023-06-25…07-04 KMA archive hole falls in between), GEFCom-Wind at
2013-04-22 against 2013-07-10 (that frame has 42 segments). For these two the
published LPS is **already** a pre-origin quantity and the deployment variant
is identical by construction, not by measurement. `test_wind_sets_are_already_pre_origin`
pins this; `--dry-run` prints it.

Six panel datasets and all ten zones therefore carry a genuine recomputation.
Every one keeps at least 120 pre-origin windows, well above the 6 the five
expanding folds require, so no series is dropped for lack of data.

## 4. Run it

```bash
cd research1
uv run python -m experiments.compute_lps_deployment --dry-run      # origins, fits nothing
uv run python -m experiments.compute_lps_deployment --check-legacy # frozen CSVs must reproduce
uv run python -m experiments.compute_lps_deployment --all          # the recomputation
# or: make lps-deployment
```

Order matters: `--check-legacy` is the gate. If the full variant no longer
reproduces `results/lps_official.csv` and `results/graded_lps_lps.csv` to 4 dp,
the run exits non-zero and nothing downstream should be believed.

Cost: the screen is ~0.04 s per univariate series; the panel is dominated by
Electricity's 321 channels and Weather's 21. Single workstation, CPU only,
minutes not hours. No GPU, no forecaster training, no frozen file rewritten.

Outputs:

* `results/lps_deployment.csv` — eight datasets: `lps_full`, `lps_deployment`,
  `delta`, `side_full`, `side_deployment`, `prereg_side`,
  `deployment_matches_prereg`, plus the origin and window counts.
* `results/lps_deployment_zones.csv` — the same for the ten zones.

## 5. Reading the result, decided in advance

*This section is committed before the recomputation is run, and its commit
timestamp is the receipt. It fixes what each outcome does to the manuscript
while all three outcomes are still live.*

τ stays at 0.3. It was fixed at pre-registration and is not re-tuned against
this recomputation; the script hard-codes it and hard-codes the eight
pre-registered sides read from `paper/predictions.md` at `cab17c1`.

Three outcomes, and what each does to the paper:

1. **All eight sides hold, all ten zones stay above τ.** §2 and Appendix A
   report both variants; the "computed before any forecaster training, but
   evaluated over blocks spanning the full series" hedge introduced in this
   revision is replaced by the plain deployment statement, and the classifier
   claim is restated on the deployment variant.
2. **A side flips.** The flip is named in the body — dataset, both LPS values,
   which side each implies — in the same sentence as the classifier claim, not
   in the appendix. The sign-prediction count is restated on the deployment
   variant and the full-variant count is kept beside it, labelled. `_summarize()`
   prints exactly this block so it cannot be missed.
3. **A value lands in the interior [0.30, 0.70].** That is the region no
   observed dataset or zone occupies (Fig. 4's shaded band), so it is
   informative rather than embarrassing: report it, and say that the threshold
   now has an interior observation where it previously had none. Do not move τ.

Electricity is the cell to watch: its full-variant LPS is 0.283, the closest
point to τ in the panel, and the pre-registration itself flagged it in advance
as the lowest-confidence sign prediction. A deployment value above 0.30 flips
it; a value below leaves the panel's one-sided gap intact.

## 6. What must not happen

* No number in the paper may be edited to match an expectation before the run.
* τ is not re-tuned, and the 2026-07-13 sign predictions are not re-issued.
* `results/lps_official.csv` and `results/graded_lps_lps.csv` stay frozen; the
  deployment values live in their own files.
* A flip is reported, not buried. The pre-registration's value comes from
  reporting what it predicted badly.
