"""G9 — the mean-only ablation: is the layer's cost the *window mean*?

Pre-registration: evidence/prereg_meanonly.md (written before any run here).

Table 1 of the workshop paper toggles RevIN, which removes window mean, window
scale and a learnable affine together. Section 2 and Figure 1 attribute the cost
to the first of those. This runner isolates it by adding one arm that removes
and restores only the window mean, giving

    RevIN - RAW        total layer cost      (Table 1)
    RevINMean - RAW    mean channel alone
    RevIN - RevINMean  scale + affine

Everything else is the information-parity block byte-for-byte: the same nn_run,
the same lookback per (dataset, backbone, h) read from the frozen
results/g4_covfair_full.csv, the same splits, optimizer, early stopping and
global-z loss scale. The rerun `raw` and `revin` cells therefore double as an
implementation control — they must reproduce the frozen numbers.

Output: results/g9_meanonly.csv (one row per run) and a per-cell summary on
stdout. Resumable: existing rows are skipped.

Usage: uv run python -m experiments.g9_meanonly [--datasets ...] [--dry-run]
"""

import argparse
import csv
import os
import time

import numpy as np
import pandas as pd

# nn_run enforces GPU1 and determinism through the same import chain as the
# parity block; keep the import order identical to it.
from experiments.g4_covfair_full import nn_run
from experiments.g4_grid import build_frame, firststage

ROOT = os.path.join(os.path.dirname(__file__), "..")
PARITY_CSV = os.path.join(ROOT, "results", "g4_covfair_full.csv")
OUT_CSV = os.path.join(ROOT, "results", "g9_meanonly.csv")

DATASETS = ("gefcom_wind", "gefcom_solar")
BACKBONES = ("linmix", "mlpmix")
ARMS = ("raw", "revin_mean", "revin")
HORIZONS = (24, 96, 336)
SEEDS = range(5)
FIELDS = ["dataset", "arm", "backbone", "h", "seed", "L", "mse", "wall_s"]


def frozen_lookbacks() -> dict:
    """L actually used by the parity block, per (dataset, backbone, h)."""
    df = pd.read_csv(PARITY_CSV)
    lb = df.groupby(["dataset", "backbone", "h"])["L"].agg(
        lambda c: int(c.mode().iloc[0]))
    return lb.to_dict()


def done_keys(path) -> set:
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path)
    return set(zip(df.dataset, df.arm, df.backbone, df.h, df.seed))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--backbones", nargs="*", default=list(BACKBONES))
    ap.add_argument("--horizons", nargs="*", type=int, default=list(HORIZONS))
    ap.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    lb = frozen_lookbacks()
    done = done_keys(OUT_CSV)
    plan = [(d, a, b, h, s)
            for d in args.datasets
            for b in args.backbones
            for h in args.horizons
            for a in ARMS
            for s in args.seeds
            if (d, a, b, h, s) not in done]
    print(f"{len(plan)} runs to go ({len(done)} already in {OUT_CSV})")
    if args.dry_run:
        for row in plan[:10]:
            print("  ", row, "L=", lb.get((row[0], row[2], row[3])))
        return

    new = not os.path.exists(OUT_CSV)
    fh = open(OUT_CSV, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if new:
        w.writeheader()

    frames = {}
    for dataset, arm, backbone, h, seed in plan:
        if dataset not in frames:
            fr = build_frame(dataset)
            frames[dataset] = (fr, firststage(fr))
        frame, level = frames[dataset]
        L = lb.get((dataset, backbone, h))
        if L is None:
            print(f"  no frozen L for {dataset}/{backbone}/h={h}; skipping")
            continue
        t0 = time.time()
        r = nn_run(frame, h, arm, backbone, seed, level, int(L))
        w.writerow({"dataset": dataset, "arm": arm, "backbone": backbone,
                    "h": h, "seed": seed, "L": int(L),
                    "mse": round(r["mse"], 6),
                    "wall_s": round(time.time() - t0, 1)})
        fh.flush()
        print(f"  {dataset:13s} {arm:11s} {backbone:7s} h={h:3d} s={seed} "
              f"L={L} mse={r['mse']:.6f} ({time.time()-t0:.0f}s)")
    fh.close()


if __name__ == "__main__":
    main()
