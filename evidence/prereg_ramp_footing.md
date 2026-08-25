# Pre-registration — G11 (covariate footing) and G12 (ramp-conditional cost)

**Written 2026-07-29, before either quantity was computed.** Governed by
`docs/philosophy.md`: each endpoint below names not only what counts as
success, but **which sentence of `paper/workshop_ccai.tex` is withdrawn if it
fails**. An endpoint you can survive failing is not an endpoint.

## Disclosure of what is already known

Honesty about the starting state, since neither block is a clean-slate
pre-registration:

- **G10 is already run and its result is known to the author of this file.**
  120 runs, linear mixer, both GEFCom sets, arms {raw, revin} × {covariates
  global, covariates window-normalized with their own lookback statistics}.
  On GEFCom-Wind the isolated layer cost `revin − raw` is
  `+0.072/+0.057/+0.038` at h=24/96/336 under global covariates and
  `−0.058/−0.100/−0.077` under window-normalized covariates. On GEFCom-Solar
  the window-normalized arms are numerically unusable (night windows have
  near-zero covariate variance; the RAW arm reaches 1.8/23.4/788 MSE).
  The 2026-07-28 manuscript reports this as an open confound.
- **The per-origin loss arrays used by G12 already exist** (`results/g4_errors/`,
  written during G4/Block B). No forecasting model is trained for G12. The
  ramp statistic itself, the binning, and every endpoint below were fixed
  before any loss array was opened.
- Nothing in G11's scale-matched arm has been run. Nothing in G12 has been
  computed.

---

## G11 — Which covariate footing, and is the audited effect footing-robust?

### The defect in the existing control

G10 window-normalizes the covariate channels **with their own lookback
statistics**, which removes the covariate level. The covariate level is the
exact quantity this paper argues instance normalization discards. So G10
changes two things at once — the units *and* the information — and cannot
adjudicate either. Reporting it as "a control that failed to close the
confound" was the wrong ruling; the correct ruling is that it is a
misspecified control.

### The three footings

The series is globally z-scored on train statistics before any arm sees it.
Let `s` be the per-window standard deviation of the target lookback in those
units — the divisor RevIN uses.

| code | covariate input | covariate level | per-instance units |
|---|---|---|---|
| **G** | `cov_z` | preserved | global (mismatched to a RevIN target) |
| **W** | `(cov − mean_w cov)/sd_w cov` | **destroyed** | per-window (own) |
| **S** | `cov_z / s` | preserved up to a common per-instance factor | per-window (target's) |

`G` is the frozen parity block. `W` is G10. **`S` is the units-only control
the confound actually calls for**: it matches the per-instance scaling without
removing the covariate mean. It also cannot blow up on solar, because it never
divides by a covariate variance.

`W` is additionally re-run with a variance floor (`sd_w ← max(sd_w, 0.1)` in
globally z-scored units) so the GEFCom-Solar cell becomes readable; the
unfloored numbers stay in the ledger.

### Grid

Arms {RAW, RevIN} × footings {G, W, W-floored, S} × datasets {GEFCom-Wind,
GEFCom-Solar} × h {24, 96, 336} × 5 seeds, linear mixer (no dropout,
bit-reproducible, matched to the analysis) and MLP mixer at a pinned thread
count. `raw` and `revin` under footing `G` must reproduce the frozen parity
block; that is the implementation check.

### AMENDMENT, 2026-07-29, after a single smoke run and before the grid

**The PRIMARY endpoint as first written was misspecified, in the same way G10
was, and is replaced by a strictly harder one.** Recorded here rather than
silently corrected.

The original PRIMARY was `RevIN_S − RAW_S`. But `RAW_S` is not a control: the
RAW target carries no per-instance scaling, so dividing *its* covariates by `s`
introduces a mismatch instead of removing one. A smoke run made this visible
(GEFCom-Wind, h=24, seed 0, linear mixer: `RAW_G` 0.217 → `RAW_S` 0.249 — the
RAW arm is damaged by the "control"). Comparing two arms that have each been
distorted in a different direction adjudicates nothing.

The confound is an alternative explanation for *one* arm's disadvantage, so the
refutation must give that arm its best case:

> **PRIMARY (amended).** `min over footings f of RevIN_f` versus `RAW_G`, per
> cell. The layer cost survives if RevIN, allowed its most favourable footing
> of the four, is still worse than RAW at the footing RAW is natively in.

This is harder to pass than the original — RevIN is handed the maximum over
four footings — and it is the comparison a hostile reviewer would demand. The
original endpoint is still computed and reported alongside, so the amendment
cannot hide a result. Everything else (grid, analysis rules, withdrawal
commitments) is unchanged; the withdrawal clauses below now attach to the
amended endpoint.

### Endpoints

**PRIMARY (as amended above).** `min_f RevIN_f − RAW_G`, seed-averaged per
cell and equally weighted over the six GEFCom cells, is **positive** on the
linear mixer. Reported alongside: the original `RevIN_S − RAW_S`, and the
within-footing toggle at each footing separately.

- *Confirmed* → the units-mismatch explanation is refuted by measurement. The
  appendix paragraph "The covariate-scaling confound, and the control that
  failed to close it" is deleted and replaced by a result, and the
  Limitations clause "the covariate-scaling confound … [is] open" is removed.
- **Not confirmed → the following are withdrawn, in the manuscript, without
  negotiation:** the abstract's unconditional "instance normalization costs
  accuracy over plain global scaling on all five covariate-capable backbones",
  the §3 heading "It is the layer, not the information", and the one-line rule
  as an unconditional statement. They are replaced by the footing grid stated
  as the finding: the cost of endogenous level handling is footing-dependent,
  here is the table, and here is which footing the field actually ships.

**SECONDARY 1 (footing robustness of the boundary).** The best endogenous arm
over *all* footings, per cell, versus CondNorm on the same cells.

- If CondNorm still wins every cell, §2's boundary claim and the LPS rule are
  footing-robust and will say so.
- If some footing closes the gap, the LPS rule is restated as conditional on
  footing and the closing footing is named in the body.

**SECONDARY 2 (decomposition of the W reversal).** Report `RAW_W − RAW_G` and
`RevIN_W − RevIN_G` separately. The prediction from the paper's own mechanism
is that destroying the covariate level costs the RAW arm materially and costs
the RevIN arm little or nothing, because the RevIN arm was not using it. This
is a prediction that can fail.

**Analysis rules, fixed now.** Seed mean within each (dataset, h) cell; equal
weight over cells; percentile bootstrap over cells, B=10⁵, `default_rng(0)`,
as in Table 1. GEFCom-Load and Jeju are not in this block (realized-temperature
reference cell and the h=48 band defect respectively).

---

## G12 — Does the audited cost concentrate in the ramps?

### Why this is a withdrawal risk and not a bonus

The abstract says the stale level goes wrong "exactly at the ramps whose errors
fossil-fuelled reserves absorb". That is the paper's entire climate pathway and
it has never been measured, only motivated from a mechanism and illustrated on
one hand-selected-by-rule origin. Under `docs/philosophy.md` it either becomes
a number or it leaves the paper.

### The ramp statistic — fixed before any loss array is opened

For a test origin `t` with horizon `h`, on the realized target only:

```
ramp(t) = max_{k=1..h-1} | y[t+k] − y[t+k−1] |
```

the largest one-hour change realized inside the forecast horizon, in
capacity-factor units for the energy sets. Chosen deliberately so that it is
**not** the staleness quantity: it does not reference the lookback window, the
window mean, or any model output. A definition like `|mean(y over horizon) −
mean(y over lookback)|` would be the thing RevIN gets wrong by construction and
the result would be circular.

Bins: terciles of `ramp(t)` computed **within each (dataset, h) cell** over
that cell's test origins, so bin membership never depends on an arm.

### Quantity

Per origin, the audited layer cost is `L_revin(t) − L_raw(t)` where `L` is the
per-origin squared-error loss already stored in `results/g4_errors/` under
`{dataset}_{arm}+cov_{backbone}_{h}_{seed}.npy`, seed-averaged first. Cells:
the seven clean exogenous cells (GEFCom-Wind ×3, GEFCom-Solar ×3, Jeju h=24).
Reported again on all eleven for continuity.

### Endpoints

**PRIMARY.** On the linear mixer, over the seven clean cells, the mean layer
cost in the **top ramp tercile** exceeds that in the **bottom** tercile, and
the bootstrap interval on the difference (percentile, over cells, B=10⁵)
excludes zero.

- *Confirmed* → the climate pathway becomes a measurement. It is promoted into
  the body and §4 is rewritten around it.
- **Not confirmed → the abstract clause "so on wind, solar and load it goes
  stale exactly at the ramps whose errors fossil-fuelled reserves absorb" is
  deleted**, together with the §1 sentence "Staleness grows with horizon and
  concentrates in the ramps and regime transitions that drive reserves and
  curtailment", and §4 is restated with no ramp mechanism — the climate
  argument falls back to "this default ships widely" alone, which is weaker and
  will be labelled as such.

**SECONDARY 1.** Monotone increase across the three terciles.
**SECONDARY 2.** The primary sign holds on at least three of the five
covariate-capable backbones.
**SECONDARY 3.** The same tercile split applied to the *level* error of the
window mean versus the NWP-implied level (Figure 1's two curves) — a check that
the bins mean what they are supposed to mean.

**Analysis rules, fixed now.** Seed-average per origin before binning. Equal
weight over cells. If a cell's test origins are fewer than 60 the cell is
dropped and the drop is reported. No alternative ramp definition, bin count or
cell base will be substituted after seeing the result; if any is added it will
be labelled exploratory and the pre-registered version will keep the headline.

---

## Provenance

This file is committed before `experiments/g11_footing.py` and
`experiments/g12_ramp.py` produce any row. The G10 disclosure above is the
honest statement of what was already known at writing time.

---

## G13 — is the boundary just an implementation oversight? (added 2026-07-29)

**Written before the grid, after one smoke seed, and the smoke seed is disclosed.**

### Why

A reviewer of the previous draft asked the question that most threatens the whole
framing, and we had not run it. The structural claim is that
`yhat = ybar + s*f((x-ybar)/s, g)` never exposes `ybar` or `s` to the backbone, so
the level cannot interact with the covariates. If that is the mechanism, the
two-line fix is to hand the backbone `ybar` and `s` as extra input channels. If
doing so closes the gap to covariate-conditioned level handling, then there is no
boundary — there is an implementation oversight, and the paper should say so.

### Design

Arm `revin+stats`: identical to the RevIN parity arm, plus the lookback window mean
and scale appended as two constant covariate channels. Control `raw+stats`: the
same two channels added to the RAW arm, which isolates whatever the extra channels
do on their own. Linear mixer (dropout-free, bit-reproducible), pinned threads,
GEFCom-Wind and GEFCom-Solar at h in {24,96,336} plus Jeju h=24, five seeds. Every
other setting is the frozen parity path.

### Disclosure

One smoke seed was run before this text (GEFCom-Wind, h=24, seed 0, linear mixer):
RAW 0.2480, RevIN 0.3200, RevIN+stats 0.3269, RAW+stats 0.2449, CondNorm 0.1789.
So the author already knows the direction on one cell. The endpoints below were
fixed knowing that, which is weaker than a blind pre-registration and is why it is
recorded rather than presented as one.

### Endpoints

**PRIMARY.** Does `revin+stats` close the gap to CondNorm? Measured as the share of
the RevIN-to-CondNorm gap that is removed, per cell, equally weighted:
`(RevIN - RevIN_stats) / (RevIN - CN)`.

- **If the share exceeds 50%**, the boundary is substantially an implementation
  oversight. **We withdraw the title, the structural argument's status as an
  explanation, and the LPS rule's motivation**, and the paper is rewritten around
  "the shipped layer omits two inputs it could cheaply be given".
- **If the share is near zero or negative**, the structural claim is strengthened:
  the level information is not recoverable by handing the statistics back, because
  what is missing is the level-covariate interaction, not the statistics
  themselves. That result goes in the body.

**SECONDARY.** `raw+stats - raw`, to confirm the extra channels are not simply
harmful. If RAW is also damaged, the PRIMARY is uninterpretable and both are
reported as inconclusive.

---

## G14 — level-isolating footings (added 2026-07-29, after round-2 review)

**Why.** A reviewer showed that our SECONDARY-2 evidence compared
`RAW: window − global` against `RevIN: window − scale`, which is not the same
manipulation: `window` is `(cov − mean_w)/sd_w` and `scale` is `cov_z/s`, so moving
between them changes the centring *and* the divisor. We had written "once units are
matched, removing the covariate level costs it nothing"; the units were not matched.
The reviewer further noted that our stated reason for discarding the `RAW_S`
endpoint applies verbatim to `RAW_WINDOW`, the arm carrying our headline
+0.039/+0.101/+0.060.

**Design.** Two footings that remove the covariate level and change nothing else:
`center_global = cov_z − mean_w(cov_z)` pairs with `global` (divisor: global for
both), `center_scale = (cov_z − mean_w(cov_z))/s` pairs with `scale` (divisor: `/s`
for both). Linear mixer, both GEFCom sets, h ∈ {24,96,336}, five seeds, pinned
threads: 120 runs.

**Disclosure.** One smoke seed was run first and is recorded in the ledger. The
endpoint below was fixed knowing it.

**Endpoint.** The level-isolating cost is larger for the globally scaled arm than
for the instance-normalized arm, per cell, equally weighted.

- *Confirmed* → the asymmetry claim stands, restated at whatever size the clean
  contrast gives, and the two-factor numbers are removed from the manuscript
  regardless of how much better they look.
- *Not confirmed* → the sentence "only the first was using it" is deleted and the
  paper keeps only the interaction and the boundary.

**Outcome: confirmed, and much smaller than the contrast it replaces.**
RAW `center_global − global` = +0.0009 [−0.0109, +0.0141], 3/6 — pooled null,
positive on all three wind cells (+0.013/+0.026/+0.008), negative on all three solar
cells. RevIN `center_scale − scale` = −0.0130 [−0.0229, −0.0040], 1/6 — it improves.
Difference = **+0.0139 [+0.0011, +0.0283], 4/6**. The manuscript now prints these
and no longer prints +0.039/+0.101/+0.060 as a level effect.
