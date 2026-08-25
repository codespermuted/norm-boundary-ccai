# When Instance Normalization Hides the Level

Code, frozen results and pre-registration records for:

> **When Instance Normalization Hides the Level:
> An Audit of Exogenous Covariate Forecasting**
> Anonymous submission, Tackling Climate Change with ML @ NeurIPS 2026.

Deep forecasters ship instance normalization (RevIN and its successors) as a
default: each input window is standardized by its own statistics and the removed
level is restored at the output. The level a forecast returns to is therefore
guessed from the target's own recent past. In energy forecasting that guess is
needless — wind, solar and load levels are set by weather that is forecast in
advance.

This repository is the audit. It toggles the layer under **information parity**
(identical covariates in every arm, only the normalization changed) across eight
datasets, five normalization arms and seven architectures.

**What the audit returns is not a cost.** The penalty's size and even its sign
flip with the *covariate footing* — how covariates are scaled against the
window-normalized target — a convention covariate forecasters fix without
discussion. What survives every footing is a boundary: the level the layer
discards has to be rebuilt from the covariates, and only two of the five
backbones in the parity grid reliably manage it.

---

## Layout

```
src/            normalization arms, backbones, data loaders, theory
experiments/    the grid, the score, and every analysis and figure script
evidence/       pre-registration records, written before the runs they govern
results/        frozen result CSVs and the authoritative ledger (RESULTS.md)
paper/          the submitted PDF, its figures, tables and bibliography
tests/          reversibility, leakage canaries, data contracts
configs/        run configurations
```

`results/RESULTS.md` is the authoritative ledger: every block, what it tested,
and what it returned, including the endpoints that did not confirm.

## Reproducing the numbers in the paper

Every number printed in the paper is re-derived from the frozen CSVs by one
script:

```bash
uv sync
uv run python -m experiments.verify_paper_numbers
```

It re-derives 68 quantities — the toggle values, the intervals, the run counts
(4,202), the cell counts, all 40 cells of the access-versus-layer table and the
footing table's column means — and prints `OK`/`XX` per line. It requires only
`results/`; no GPU, no re-training.

To recompute the score itself from the curated data:

```bash
uv run python -m experiments.compute_lps_official   # the eight-dataset panel
uv run python -m experiments.graded_lps --lps       # the ten GEFCom-Wind zones
```

Re-running the forecasting grid needs a GPU and the curated datasets (see
below); `experiments/g4_grid.py` and `experiments/g4_covfair_full.py` are the
entry points, and `Makefile` collects the analysis targets.

## Data

Raw KMA and GEFCom2014 source data are **not redistributed** here for licensing
reasons. `src/data/collectors/` and `src/data/curation.py` contain the
collectors and the preparation steps; the data contract each curated frame must
satisfy is asserted in the loaders and pinned in `tests/test_curation.py`.

The per-origin squared-error arrays that the ramp split consumes
(`results/g4_errors/`, 6,433 files, ~170 MB) are omitted here for size. They are
written by the grid runner.

## Reproducibility limits, stated up front

* PyTorch samples CPU dropout masks per intra-op thread chunk, so **any backbone
  with dropout reproduces only at a fixed thread count**. Varying threads from 1
  to 28 moves a single MLP-mixer cell by up to 0.008 MSE — the same order as the
  effect being measured. Every block added after that discovery pins the thread
  count, and every primary endpoint is carried by the dropout-free linear mixer,
  which re-runs bit-exactly.
* The information-parity block predates that fix, so its three dropout-bearing
  backbones keep intervals conditional on the thread configuration of the
  original run.
* Splits are strictly out-of-time; global scaling is fit on the train segment
  only; no window ever spans a curated archive gap.

## Pre-registration

`evidence/` holds the written record for each block. They are **not equally
strong**, and the paper grades them rather than treating them as one thing:

| block | record | commit-timestamped ahead of its run? |
|---|---|---|
| main grid | `prereg_predictions_maingrid.md` | yes |
| zone study | `prereg_graded_lps.md` | yes |
| shrinkage valve | `prereg_shrinkage_arimax.md` | yes |
| footing, ramp | `prereg_ramp_footing.md` | no |
| mean-only | `prereg_meanonly.md` | no |
| fed-back statistics | `prereg_ramp_footing.md` (§G13) | no |

`evidence/prereg_evidence.md` documents the timestamp chain for the main grid.
Endpoints that did not confirm are reported in `results/RESULTS.md` and in the
paper's appendix, next to the ones that did.

## License

MIT, see `LICENSE`.
