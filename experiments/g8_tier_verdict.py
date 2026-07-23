"""Executable A.6 / A.6a interpretation rule for the shrinkage pre-registration
(evidence/prereg_shrinkage_arimax.md). Committed BEFORE the standard-group
shrinkage rows exist, so the Tier verdict is deterministic, not hand-computed
post-hoc.

MATCHED comparison (A.6a): shrunk-CN at cell (dataset, backbone, horizon) is
compared to the best instance-norm arm *in the same cell* — never the Table 1
pooled mean. Per-dataset scalar = mean over the in-scope cells (both sides over
the SAME cells). IN family present in g4_grid.csv = {revin, san, fan}; on
lgbm_dms only revin is present, so best-IN there = revin.

This version reports, per standard dataset:
  * matched best-IN   (A.6a evaluation baseline; the operative Tier test)
  * matched Raw       (Tier 1's real claim is "shrunk-CN >= Raw"; and if
                       shrunk-CN < matched Raw, alpha-hat is doing more than
                       shrinkage -> estimation-asymmetry, needs train-holdout)
  * pooled best-IN    (pre-reg A.5 PREDICTION scale; the point predictions
                       0.387/0.287/... were derived here, NOT the matched scale)
so the prediction/evaluation scale split (A.5 pooled vs A.6 matched) is explicit.

Reads results/g4_grid.csv (frozen Block A) + results/g8_shrinkage.csv (this run).
Safe on partial g8 output. Usage:  uv run python -m experiments.g8_tier_verdict
"""
import csv
import os
from collections import defaultdict

ROOT = os.path.join(os.path.dirname(__file__), "..")
G4 = os.path.join(ROOT, "results", "g4_grid.csv")
G8 = os.path.join(ROOT, "results", "g8_shrinkage.csv")

IN_NORMS = ("revin", "san", "fan")          # instance-normalization family (g4)
STANDARD = ("etth1", "etth2", "weather", "electricity")
EXOG = ("gefcom_wind", "jeju_wind")
BACKBONES = ("rlinear", "lgbm_dms")
SCOPE = {"gefcom_wind": (24, 96, 336), "jeju_wind": (24, 48),
         "etth1": (24, 96, 336), "etth2": (24, 96, 336),
         "weather": (24, 96, 336), "electricity": (24, 96, 336)}
# pre-reg A.5 "best instance-norm" POOLED reference (Table 1 scale, 4 backbones x
# 3 horizons x 5 seeds). The A.5 point predictions were made against THESE; the
# A.6/A.6a Tier test uses matched cells instead. Shown only to expose the split.
POOLED_BEST_IN = {"etth1": 0.3771, "etth2": 0.2869,
                  "weather": 0.1666, "electricity": 0.1375}


def _mean(xs):
    return sum(xs) / len(xs)


def load(path, key_cols):
    acc = defaultdict(list)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        for r in csv.DictReader(f):
            k = tuple(int(r[c]) if c == "h" else r[c] for c in key_cols)
            acc[k].append(float(r["mse"]))
    return {k: _mean(v) for k, v in acc.items()}


def load_alpha():
    acc = defaultdict(list)
    if not os.path.exists(G8):
        return {}
    with open(G8) as f:
        for r in csv.DictReader(f):
            acc[(r["dataset"], r["alpha_kind"])].append(float(r["alpha_mean"]))
    return {k: _mean(v) for k, v in acc.items()}


def best_in(g4, ds, bk, h):
    vals = [g4[(ds, n, bk, h)] for n in IN_NORMS if (ds, n, bk, h) in g4]
    return min(vals) if vals else None


def run_verdict(kind, g4, g8, amean):
    role = "PRIMARY (A.6 operative)" if kind == "alphahat" else "SECONDARY (reported per pre-reg A.2)"
    print(f"\n########## alpha_kind = {kind}  [{role}] ##########")
    print("gaps are relative to the MATCHED baseline unless labelled pooled.\n")

    w10_matched, w10_pooled, beats, mech_flags = [], [], [], []
    for ds in STANDARD:
        cells = [(bk, h) for bk in BACKBONES for h in SCOPE[ds]]
        rows = [(bk, h, g8[(ds, kind, bk, h)], best_in(g4, ds, bk, h),
                 g4.get((ds, "raw", bk, h)))
                for bk, h in cells
                if (ds, kind, bk, h) in g8 and best_in(g4, ds, bk, h) is not None
                and (ds, "raw", bk, h) in g4]
        if not rows:
            print(f"{ds:12s} (no shrunk-CN rows yet)")
            continue
        sc = _mean([r[2] for r in rows])
        bi = _mean([r[3] for r in rows])
        rw = _mean([r[4] for r in rows])
        pooled = POOLED_BEST_IN[ds]
        g_bi = (sc - bi) / bi * 100          # matched: vs best-IN (operative)
        g_rw = (sc - rw) / rw * 100          # matched: vs Raw (Tier-1 mechanism)
        g_pl = (sc - pooled) / pooled * 100   # pooled : vs A.5 reference (predict)
        wm = abs(g_bi) <= 10
        wp = abs(g_pl) <= 10
        beat = sc <= bi
        mech = sc <= 0.97 * rw                # beats matched Raw by >3% -> flag
        w10_matched.append(wm); w10_pooled.append(wp); beats.append(beat)
        mech_flags.append(mech)
        a = amean.get((ds, kind), float("nan"))
        print(f"{ds:12s} shrunk-CN={sc:.4f}  a~{a:.3f}  ({len(rows)}/{len(cells)} cells)")
        print(f"    matched best-IN={bi:.4f} (gap {g_bi:+.1f}%, within10={wm}, beatsIN={beat})")
        print(f"    matched Raw    ={rw:.4f} (gap {g_rw:+.1f}%"
              + (", shrunk-CN BEATS Raw >3% -> more than shrinkage" if mech else "")
              + ")")
        print(f"    pooled best-IN ={pooled:.4f} (gap {g_pl:+.1f}%, within10_pooled={wp})  [A.5 predict scale]")
        for bk, h, s, b, r in rows:
            print(f"      {bk:9s} h{h:<4} sc={s:.4f} IN={b:.4f} Raw={r:.4f}")

    n = len(w10_matched)
    if n:
        print(f"\n--- standard-group verdict ({n}/4 datasets present) ---")
        print(f"Tier 1 MATCHED  (within 10% of matched best-IN on >=3/4): "
              f"{sum(w10_matched)}/{n} -> {'TRIGGERS' if sum(w10_matched) >= 3 else 'does NOT trigger'}")
        print(f"Tier 1 pooled   (within 10% of pooled A.5 best-IN on >=3/4): "
              f"{sum(w10_pooled)}/{n}  [prediction-scale reference only]")
        print(f"Tier 2 MATCHED  (shrunk-CN <= matched best-IN on >=2/4): "
              f"{sum(beats)}/{n} -> {'TRIGGERS (boundary wrong on standard side)' if sum(beats) >= 2 else 'does NOT trigger'}")
        if any(mech_flags):
            print("Mechanism: >=1 dataset has shrunk-CN beating matched Raw by >3% "
                  "-> alpha-hat exceeds pure shrinkage; run train-holdout alpha-hat "
                  "(A.6a estimation-asymmetry check).")
        if n < 4:
            print(f"  [partial: {n}/4 standard datasets have rows]")

    print("\n--- Tier 3 (exogenous safety-valve cost, >5% CN degradation) ---")
    for ds in EXOG:
        worst = None
        for bk in BACKBONES:
            for h in SCOPE[ds]:
                if (ds, kind, bk, h) not in g8 or (ds, "condnorm", bk, h) not in g4:
                    continue
                s, c = g8[(ds, kind, bk, h)], g4[(ds, "condnorm", bk, h)]
                deg = (s - c) / c * 100
                print(f"    {ds:12s} {bk:9s} h{h:<4} shrunk-CN={s:.4f} "
                      f"CN-as-run={c:.4f} deg={deg:+.1f}%")
                worst = deg if worst is None else max(worst, deg)
        if worst is not None:
            a = amean.get((ds, kind), float("nan"))
            print(f"  {ds}: a~{a:.2f}, worst deg {worst:+.1f}% -> "
                  f"{'COST (Tier 3 triggers)' if worst > 5 else 'no material cost'}")


def main():
    g4 = load(G4, ("dataset", "norm", "backbone", "h"))
    g8 = load(G8, ("dataset", "alpha_kind", "backbone", "h"))
    amean = load_alpha()
    print("=== A.6/A.6a matched Tier verdict — BOTH coefficients ===")
    print("Tier thresholds are defined on the PRIMARY (alphahat) per A.6; the")
    print("SECONDARY (lpsclip) is reported because the pre-reg mandates both and")
    print("referees will compute it. CAVEAT: a Tier-1 non-trigger by a coefficient")
    print("that failed as a safety valve (alpha-hat overfits when the first stage")
    print("fits noise) does NOT by itself establish the catastrophe framing -- a")
    print("working valve is a prerequisite for this block to adjudicate anything.")
    for kind in ("alphahat", "lpsclip"):
        run_verdict(kind, g4, g8, amean)


if __name__ == "__main__":
    main()
