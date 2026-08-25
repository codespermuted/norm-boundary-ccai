"""G11 -- covariate footing: is the audited layer cost footing-robust?

Pre-registered in evidence/prereg_ramp_footing.md.

The parity block globally z-scores the covariate channels while RevIN
window-normalizes only the target, so a reviewer can argue that part of the
measured layer cost is a units mismatch rather than level handling. G10 tried
to settle that by window-normalizing the covariates too -- but with their own
lookback statistics, which *destroys the covariate level*, i.e. exactly the
signal this paper says instance normalization discards. G10 therefore changes
the units and the information at once and cannot adjudicate either.

This block separates them. Three covariate footings (see nn_run):

    G  global        cov_z                     level kept, global units
    W  window        (cov-mean_w)/sd_w         level destroyed, own units   (= G10)
    Wf window_floor  ditto, sd_w floored       readable on the solar set
    S  scale         cov_z / s                 level kept, target's units

S is the units-only control the confound actually calls for. Both target arms
{raw, revin} are run under every footing, so the isolated layer toggle can be
read within each footing.

The (raw, G) and (revin, G) cells must reproduce the frozen parity block --
that is the implementation check, asserted by --verify.

Output: results/g11_footing.csv
Usage: uv run python -m experiments.g11_footing [--backbones linmix mlpmix]
"""

import argparse
import csv
import os
import time

import pandas as pd
import torch

from experiments.g4_covfair_full import nn_run
from experiments.g4_grid import build_frame, firststage

ROOT = os.path.join(os.path.dirname(__file__), "..")
PARITY = os.path.join(ROOT, "results", "g4_covfair_full.csv")
OUT = os.path.join(ROOT, "results", "g11_footing.csv")

DATASETS = ("gefcom_wind", "gefcom_solar")
# jeju_wind's forecast band only covers h<=48 and its h=48 cells consume the
# h<=24 band (a disclosed defect), so only h=24 is admissible there.
DATASET_HORIZONS = {"jeju_wind": (24,)}
HORIZONS = (24, 96, 336)
SEEDS = range(5)
FOOTINGS = ("global", "window", "window_floor", "scale",
            "center_global", "center_scale")
TARGET_ARMS = ("raw", "revin")
FIELDS = ["dataset", "arm", "footing", "backbone", "h", "seed", "L", "mse",
          "threads", "wall_s"]
THREADS = 8          # pinned: dropout backbones are thread-reproducible only


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", default=["linmix", "mlpmix"])
    ap.add_argument("--footings", nargs="+", default=list(FOOTINGS))
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--arms", nargs="+", default=list(TARGET_ARMS))
    ap.add_argument("--threads", type=int, default=THREADS)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify", action="store_true",
                    help="check the global-footing cells against the frozen "
                         "parity block and exit")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)

    parity = pd.read_csv(PARITY)
    lb = (parity.groupby(["dataset", "backbone", "h"])["L"]
          .agg(lambda c: int(c.mode().iloc[0])).to_dict())

    if args.verify:
        d = pd.read_csv(OUT)
        d = d[d.footing == "global"]
        ref = (parity[parity.arm.isin(TARGET_ARMS)]
               .groupby(["dataset", "arm", "backbone", "h"]).mse.mean())
        got = d.groupby(["dataset", "arm", "backbone", "h"]).mse.mean()
        j = pd.concat([ref.rename("frozen"), got.rename("rerun")], axis=1,
                      join="inner")
        j["absdiff"] = (j.frozen - j.rerun).abs()
        print(j.to_string())
        print(f"\nmax |diff| = {j.absdiff.max():.2e} over {len(j)} cells")
        return

    done = set()
    if os.path.exists(OUT):
        d = pd.read_csv(OUT)
        done = set(zip(d.dataset, d.arm, d.footing, d.backbone, d.h, d.seed))

    plan = [(ds, arm, ft, bb, h, s)
            for ds in args.datasets for bb in args.backbones
            for h in DATASET_HORIZONS.get(ds, HORIZONS)
            for ft in args.footings for arm in args.arms for s in SEEDS
            if (ds, arm, ft, bb, h, s) not in done]
    print(f"{len(plan)} runs to go (threads={args.threads})")
    if args.dry_run:
        return

    new = not os.path.exists(OUT)
    fh = open(OUT, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if new:
        w.writeheader()
    frames = {}
    for ds, arm, ft, bb, h, s in plan:
        if ds not in frames:
            fr = build_frame(ds)
            frames[ds] = (fr, firststage(fr))
        frame, level = frames[ds]
        lookback = int(lb[(ds, bb, h)])
        t0 = time.time()
        r = nn_run(frame, h, arm, bb, s, level, lookback, cov_footing=ft)
        w.writerow({"dataset": ds, "arm": arm, "footing": ft, "backbone": bb,
                    "h": h, "seed": s, "L": lookback,
                    "mse": round(r["mse"], 6), "threads": args.threads,
                    "wall_s": round(time.time() - t0, 1)})
        fh.flush()
    fh.close()
    print("done")


if __name__ == "__main__":
    main()
