"""G10 — is the measured layer cost a units mismatch?

The parity block globally z-scores the covariate channels on train statistics
while RevIN window-normalizes only the target. A reviewer can therefore argue
that part of the measured "layer cost" is a scale mismatch between a
window-normalized target and globally scaled covariates, rather than level
handling.

This closes it by putting both on the same footing. Four arms, on the linear
mixer only (the one backbone that reproduces the frozen parity block
bit-exactly, and the class the analysis is matched to):

    raw            no instance normalization anywhere            (parity baseline)
    revin          target window-normalized, covariates global   (parity block)
    raw_cn         covariates window-normalized, target global
    revin_cn       both window-normalized                        (matched footing)

The confound is closed if the matched contrast (revin_cn - raw_cn) reproduces
the baseline contrast (revin - raw): the layer still costs when the units
mismatch is removed.

Output: results/g10_covscale.csv.
Usage: uv run python -m experiments.g10_covscale
"""

import argparse
import csv
import os
import time

import pandas as pd

from experiments.g4_covfair_full import nn_run
from experiments.g4_grid import build_frame, firststage

ROOT = os.path.join(os.path.dirname(__file__), "..")
PARITY = os.path.join(ROOT, "results", "g4_covfair_full.csv")
OUT = os.path.join(ROOT, "results", "g10_covscale.csv")

DATASETS = ("gefcom_wind", "gefcom_solar")
BACKBONE = "linmix"
HORIZONS = (24, 96, 336)
SEEDS = range(5)
# (row label, norm arm passed to nn_run, covariates window-normalized?)
ARMS = (("raw", "raw", False),
        ("revin", "revin", False),
        ("raw_cn", "raw", True),
        ("revin_cn", "revin", True))
FIELDS = ["dataset", "arm", "backbone", "h", "seed", "L", "mse", "wall_s"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    lb = (pd.read_csv(PARITY).groupby(["dataset", "backbone", "h"])["L"]
          .agg(lambda c: int(c.mode().iloc[0])).to_dict())
    done = set()
    if os.path.exists(OUT):
        d = pd.read_csv(OUT)
        done = set(zip(d.dataset, d.arm, d.h, d.seed))

    plan = [(ds, lab, nrm, cn, h, s)
            for ds in DATASETS for h in HORIZONS
            for lab, nrm, cn in ARMS for s in SEEDS
            if (ds, lab, h, s) not in done]
    print(f"{len(plan)} runs to go")
    if args.dry_run:
        return

    new = not os.path.exists(OUT)
    fh = open(OUT, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if new:
        w.writeheader()
    frames = {}
    for ds, lab, nrm, cn, h, s in plan:
        if ds not in frames:
            fr = build_frame(ds)
            frames[ds] = (fr, firststage(fr))
        frame, level = frames[ds]
        L = int(lb[(ds, BACKBONE, h)])
        t0 = time.time()
        r = nn_run(frame, h, nrm, BACKBONE, s, level, L, cov_instance_norm=cn)
        w.writerow({"dataset": ds, "arm": lab, "backbone": BACKBONE, "h": h,
                    "seed": s, "L": L, "mse": round(r["mse"], 6),
                    "wall_s": round(time.time() - t0, 1)})
        fh.flush()
    fh.close()
    print("done")


if __name__ == "__main__":
    main()
