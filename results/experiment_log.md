# Experiment Log — running lab journal

**Purpose.** Per-turn, chronological record of experiments (setup → result → verdict
→ open threads) so work persists across sessions and is never confined to one
session's context. Records results **as-is**, including failures and
point-prediction misses. Polished paper-facing summaries live in
`results/summary.md`; governance/pre-registration in `evidence/`; state + next
steps in `HANDOFF.md`. **Update this file every experiment turn.**

Prior phases (G0–G4, gates, LPS inference) are summarized in `results/summary.md`,
`results/gate1.md`, `results/gate2.md`, `results/lps_inference.md`. This log starts
at the Phase-2 shrinkage/ARIMAX + IJF-reframe cycle.

---

## Current focus / open threads (update each turn)

- **train-holdout α̂ — REQUIRED, next.** The val-split α̂ failed as a safety valve
  (see G8 below). A lower-variance / leakage-free estimator (train-internal
  holdout; alternatives: CV-fold, significance gate, shrink-α̂). If it works, the
  valve is implementable and Tier 1 properly triggers; if it *also* fails, we can
  say empirically "shrinkage is hard." Both outcomes beat the current inconclusive
  state.
- **§7 shrinkage narrative — DO NOT finalize until train-holdout lands.** Current
  honest status = Option C (the block did not adjudicate the shrinkage objection).
- **ARIMAX (classical baseline)** — pending `uv add statsmodels`; Fourier + ARIMA
  errors, exogenous-first. Feeds Table 1 promotion (Phase 4).
- **IJF reframe** — §2 rewritten (`a5b28b4`); §1 rewrite unblocked (framing→first
  para, title→instrument status); §4.2 third conservatism ground (LPS=λ−s biased
  low); §7 shrinkage + asymmetry headline; Table 1 promotion — all pending.
- **PDFs** — main.pdf stale (pre-§2, main branch behind by 4 commits); CCAI PDF
  ~current with tex; rebuild via tectonic at a checkpoint. See earlier status.

---

## 2026-07-24 — G8 shrinkage safety-valve (COMPLETE, 204/204)

**Pre-reg** `evidence/prereg_shrinkage_arimax.md` §A (frozen `9bd4c1f`; matched-cell
amendment `69b4635`; disclosure precision `5d068f9`). **Runner**
`experiments/g8_shrinkage.py` (`7a0d564`). **Verdict** `experiments/g8_tier_verdict.py`
(committed pre-result; reports both coefficients).

**Question.** Does a level-shrinkage safety valve on CondNorm remove the need for
the LPS diagnostic?

**Coefficients (per channel).** PRIMARY α̂ = clip(Cov(ȳ_val,ĝ_val)/Var(ĝ_val),0,1)
(validation-split recalibration). SECONDARY clip(LPS,0,1). Method A (re-train on
shrunk-level residuals). Scope: exog {gefcom_wind, jeju} + standard {etth1, etth2,
weather, electricity}, backbones {rlinear, lgbm_dms}, h{24,96,336}.

**Results — matched, 6-cell means (rlinear+lgbm × 3h):**

| dataset | LPS | CN-run | α̂ | shrunk-α̂ | shrunk-LPS | Raw | best-IN |
|---|---|---|---|---|---|---|---|
| etth1 | −0.72 | 2.751 | 0.484 | 0.511 | 0.387 | 0.386 | 0.379 |
| etth2 | −0.20 | 8.781 | 0.521 | 0.778 | 0.451 | 0.445 | 0.281 |
| weather | 0.11 | 0.842 | 0.450 | 0.190 | 0.176 | 0.185 | 0.170 |
| electricity | 0.28 | 0.173 | 0.885 | 0.155 | 0.138 | 0.144 | 0.143 |

**Tier verdict (matched):**

| | α̂ (PRIMARY, A.6 operative) | clip(LPS) (SECONDARY) |
|---|---|---|
| Tier 1 concession (within 10% of best-IN, ≥3/4) | **0/4 — no** | **3/4 — YES** (etth1 +2.1%, weather +4.7%, electricity −0.6%; etth2 +62% out) |
| Tier 2 falsification (≤ best-IN, ≥2/4) | 0/4 — no | 1/4 — no (electricity only) |
| Tier 3 exog cost (>5% CN degradation) | no cost (gefcom +0.0%, jeju +1.7%) | **BOTH cost** (gefcom +33.4%, jeju +14.1%) |

**Key finding — α̂ FAILED as a safety valve (estimator, not concept).** On the
standard group α̂ overfits: shrunk-CN lands **+2.5 to +75% ABOVE Raw** (worse than a
plain baseline; etth2/lgbm blows up to 1.1–1.2). Diagnosis: where the first stage
fits noise (LPS<0), α̂=Cov(ȳ_val,ĝ_val)/Var(ĝ_val) has a noise–noise numerator with
exploding sample variance, worsened by per-channel estimation (electricity 321ch).
So α̂ is most unstable exactly where it must be ≈0. Fixable (train-holdout / CV /
significance gate / shrink-α̂). The naïve "standard recalibration ⇒ can't be called
a weak implementation" rationale did not hold — this *is* a weak-implementation
result.

**Interpretation — Option C: the block DID NOT ADJUDICATE the objection.** The two
coefficients contradict on Tier 1, and neither is clean: α̂'s Tier-1 non-trigger is
an artifact of the valve failing (not a sharp boundary); clip(LPS) triggers Tier 1
but costs exogenous (Tier 3). So "Tier 1 non-trigger ⇒ catastrophe framing
survives" is **false**. A safety valve that doesn't function can't adjudicate.

**clip(LPS) reading (careful).** etth1/etth2 shrunk-LPS≈Raw (+0.2%, +1.4%) is
*arithmetic* (LPS<0 → α=0 → Raw), not a finding. The genuine cases are weather
(−4.6% vs Raw) and electricity (−3.7% vs Raw, and −0.6% vs best-IN): α≠0 yet beats
Raw — clip(LPS) actually does something there. electricity beating best-IN is a
secondary-coefficient, boundary-dataset (LPS 0.283≈τ) result; reported as-is.

**Point-prediction miss (A.5).** Predicted standard α̂≈0, shrunk-CN→Raw, Tier 1
trigger. Actual α̂=0.45–0.89, shrunk-CN>Raw, Tier 1 no (primary). All three missed —
**all in the thesis-favorable direction**, which is a warning sign the experiment
did not measure what it was designed to (α̂ estimator failure), not a win. Recorded
as-is per A.5 ("value is the rules, not the point values").

**Surviving arguments (independent of this experiment).** (1) Cost — shrinkage must
train once to estimate the coefficient; LPS is pre-training. Intact. (2) Asymmetry
(§7.5) — rests on Prop 1 + Block B, not on G8. Intact; this is the headline. (3)
Conceptual (weak) — the shrinkage coefficient's estimand is essentially what LPS
measures (from the α*=λ/(λ+s) derivation, not G8) → "shrinkage needs the same
information," NOT "shrinkage is impossible."

**Also confirmed.** Seed-symmetric (both sides 5-seed means; lgbm deterministic).
gefcom_wind + jeju Tier 3 clean for α̂ (exogenous no self-harm). clip(LPS)
under-shrinks exogenous (gefcom +33%, jeju +14%) — the pre-registered reason it is
secondary, now shown empirically.

**Next.** train-holdout α̂ (required). §7 narrative after that.
