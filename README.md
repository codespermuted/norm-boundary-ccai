# When Instance Normalization Hurts

Code, theory, and pre-registered experiments for:

> **When Instance Normalization Hurts: Exogenously-Driven Level Predictability
> and Conditional Normalization for Multi-Step Forecasting**
> Jaehong Yu, Jaesung Hong, Sungwon Lee (Independent Researchers)
> arXiv preprint, 2026. *(arXiv ID to be added on announcement)*

**TL;DR.** Instance normalization (RevIN and its successors) assumes that
lookback-window statistics extrapolate to the forecast horizon. When the
series level is instead driven by *exogenous* covariates (wind, solar, load
with weather forecasts), that assumption fails and instance normalization
hurts — predictably. We derive the crossover in closed form, and reduce it to
a pre-training diagnostic, the **Level Predictability Score (LPS)**: the
out-of-sample R² of window-mean levels regressed on covariates. Decision
rule: **use conditional normalization when LPS ≥ τ = 0.3**, otherwise keep
instance normalization. In a pre-registered study (predictions committed to
version control before any experiment run), the rule went **8/8** on sign
predictions across eight datasets.

## Using the diagnostic on your own data

```python
from src.theory.lps import lps, delta_lps, calendar_features

# y: (T,) or (T, C) target array; X: (T, d) covariates known at forecast time
score = lps(y, X, w=96)          # official spec: w=96, LightGBM first stage,
                                 # expanding chronological CV (5 folds)
# score >= 0.3  ->  conditional normalization
# score <  0.3  ->  instance normalization (RevIN-style)
```

`src/theory/lps_inference.py` adds a circular-shift permutation test
(`permutation_test`) and a moving-block-bootstrap CI (`mbb_ci`) around the
same statistic, and `delta_lps` measures the covariate contribution *over* a
persistence baseline (the refinement discussed in §7 of the paper).

## Repository layout

| Path | Contents |
|---|---|
| `src/norms/` | Normalization arms (Raw, RevIN, SAN, FAN, CondNorm) behind a common `forward(x, mode)` interface; every arm has an invertibility test |
| `src/theory/` | Closed-form risks, LPS / ΔLPS, permutation & bootstrap inference |
| `src/data/` | Curation pipelines and loaders (contract-asserted Parquet), incl. KMA NWP / data.go.kr collectors |
| `experiments/` | Grid runners, analysis, and table/figure generators for every block |
| `configs/` | Frozen experiment configs |
| `tests/` | Invertibility, leakage-canary, inference, and smoke tests (`uv run pytest`) |
| `paper/` | Full LaTeX source of the paper |
| `results/` | Logged metrics (CSV) and per-window error arrays used by the tables |
| `evidence/` | Pre-registration evidence trail (see below) |
| `docs/` | Design audits, statistical hardening notes, OSF wave-2 draft |

## Reproducing

Dependencies are managed with [uv](https://docs.astral.sh/uv/) (Python ≥ 3.11,
PyTorch ≥ 2.7 cu128):

```bash
uv sync
uv run pytest              # invertibility / leakage / canary / smoke tests
make figures tables        # regenerate every figure and table from logged results
make paper                 # build paper/main.pdf (tectonic)
```

Training runs are seeded and deterministic
(`torch.use_deterministic_algorithms`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`);
all 4,202 runs log full configs and the git commit hash to MLflow.

**Data.** Raw datasets are not redistributed here. GEFCom2014 and the standard
LTSF benchmarks (ETTh1/2, Electricity, Weather) come from their original
public sources; the Jeju wind set is built by the collectors in
`src/data/collectors/` (KMA API-hub and data.go.kr keys required — see
`docs/env.template`). All curation code, with contract assertions, is included.

## Pre-registration evidence

The LPS specification, the threshold τ = 0.3, and all eight sign predictions
were frozen in commit `cab17c1` (2026-07-13 13:20:23 KST), **four minutes
before the first grid run** (13:24:36). The full evidence chain — commit
trail, MLflow start times, a frozen snapshot of the MLflow tracking database,
and the hash correspondence across a later history rewrite — is documented in
[`evidence/prereg_evidence.md`](evidence/prereg_evidence.md). Post-hoc
additions (baselines, probabilistic metrics, inference module) are kept in
separate namespaces and labeled as such in the paper.

## License

MIT — see [LICENSE](LICENSE). The datasets remain under their original
providers' terms.
