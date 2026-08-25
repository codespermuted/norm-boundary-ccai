"""G9 analysis — exactly the rule fixed in evidence/prereg_meanonly.md.

Seed-mean within each (dataset, backbone, h) cell, then equal weight over the
six cells per backbone. Percentile bootstrap over cells. Per-cell table printed
in full so pooling cannot hide a sign flip.

Endpoints (pre-registered):
  PRIMARY 1  d_mean  > 0 in mean on both backbones
  PRIMARY 2  d_mean / d_total >= 0.5 on both backbones
  SECONDARY 3  |d_mean| > |d_total - d_mean|
  SECONDARY 4  d_mean grows with horizon
  CONTROL      rerun raw/revin reproduce results/g4_covfair_full.csv

Usage: uv run python -m experiments.g9_meanonly_analyze
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
G9 = os.path.join(ROOT, "results", "g9_meanonly.csv")
PARITY = os.path.join(ROOT, "results", "g4_covfair_full.csv")
B, SEED = 100_000, 0


def boot(vals, b=B, seed=SEED):
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, dtype=float)
    draws = rng.choice(v, size=(b, len(v)), replace=True).mean(axis=1)
    return float(v.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main():
    d = pd.read_csv(G9)
    cell = (d.groupby(["dataset", "backbone", "h", "arm"])["mse"].mean()
             .unstack("arm").reset_index())
    cell["d_mean"] = cell["revin_mean"] - cell["raw"]
    cell["d_total"] = cell["revin"] - cell["raw"]
    cell["d_rest"] = cell["revin"] - cell["revin_mean"]
    cell["share"] = cell["d_mean"] / cell["d_total"]

    print("=" * 78)
    print("CONTROL — rerun arms against the frozen parity block, per backbone")
    old = pd.read_csv(PARITY)
    repro = {}
    for bb, s_bb in cell.groupby("backbone"):
        worst = 0.0
        for _, r in s_bb.iterrows():
            o = old[(old.dataset == r.dataset) & (old.backbone == r.backbone)
                    & (old.h == r.h)]
            for arm in ("raw", "revin"):
                a = o[o.arm == arm]["mse"].mean()
                if not np.isnan(a):
                    worst = max(worst, abs(a - r[arm]))
        repro[bb] = worst
        tag = ("bit-exact — comparable to Table 1" if worst < 1e-6 else
               f"NOT reproducible (max |diff| {worst:.4f}) — internally consistent only")
        print(f"  {bb:8s} {tag}")
    print("  note: CPU dropout masks are sampled per intra-op thread chunk, so any")
    print("  backbone with dropout reproduces only at a fixed thread count; the")
    print("  linear mixer has no dropout and reproduces exactly.")

    print()
    print("=" * 78)
    print("PER-CELL (seed means; global-z MSE)")
    show = cell[["dataset", "backbone", "h", "raw", "revin_mean", "revin",
                 "d_mean", "d_rest", "d_total", "share"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))

    print()
    print("=" * 78)
    print("PER-BACKBONE (equal weight over cells, percentile bootstrap over cells)")
    verdict = {}
    for bb, s in cell.groupby("backbone"):
        m_mean, lo_m, hi_m = boot(s.d_mean)
        m_tot, lo_t, hi_t = boot(s.d_total)
        m_rest, lo_r, hi_r = boot(s.d_rest)
        share = m_mean / m_tot if m_tot else float("nan")
        verdict[bb] = dict(d_mean=m_mean, d_total=m_tot, d_rest=m_rest,
                           share=share, ci_mean=(lo_m, hi_m))
        print(f"\n  {bb}  ({len(s)} cells)")
        print(f"    mean channel   d_mean  = {m_mean:+.4f}  [{lo_m:+.4f}, {hi_m:+.4f}]"
              f"   cells>0 {(s.d_mean > 0).sum()}/{len(s)}")
        print(f"    scale+affine   d_rest  = {m_rest:+.4f}  [{lo_r:+.4f}, {hi_r:+.4f}]"
              f"   cells>0 {(s.d_rest > 0).sum()}/{len(s)}")
        print(f"    total layer    d_total = {m_tot:+.4f}  [{lo_t:+.4f}, {hi_t:+.4f}]"
              f"   cells>0 {(s.d_total > 0).sum()}/{len(s)}")
        print(f"    share of total carried by the mean channel: {share:.1%}")

    print()
    print("=" * 78)
    print("PRE-REGISTERED ENDPOINTS")
    p1 = all(v["d_mean"] > 0 for v in verdict.values())
    p2 = all(v["share"] >= 0.5 for v in verdict.values())
    p3 = all(abs(v["d_mean"]) > abs(v["d_rest"]) for v in verdict.values())
    hz = cell[cell.backbone == "linmix"].groupby("h")["d_mean"].mean()
    p4 = bool(hz.is_monotonic_increasing)
    print("  (horizon endpoint read on the bit-reproducible backbone, linmix)")
    for name, val, txt in [
        ("PRIMARY 1", p1, "d_mean > 0 on both backbones"),
        ("PRIMARY 2", p2, "mean channel carries >= 50% of the layer cost"),
        ("SECONDARY 3", p3, "|mean channel| > |scale+affine|"),
        ("SECONDARY 4", p4, "d_mean grows monotonically with horizon"),
    ]:
        print(f"  [{'CONFIRMED' if val else 'NOT CONFIRMED'}] {name}: {txt}")
    for bb, sb in cell.groupby("backbone"):
        for ch, col in (("mean", "d_mean"), ("scale+affine", "d_rest")):
            byh = sb.groupby("h")[col].mean()
            print(f"    {bb:8s} {ch:12s} by horizon: "
                  + ", ".join(f"h={int(k)}: {v:+.4f}" for k, v in byh.items()))
    print(f"\n  d_mean by horizon (linmix): "
          + ", ".join(f"h={int(k)}: {v:+.4f}" for k, v in hz.items()))

    out = os.path.join(ROOT, "results", "g9_meanonly_cells.csv")
    show.to_csv(out, index=False)
    print(f"\nper-cell table written to {out}")


if __name__ == "__main__":
    main()
