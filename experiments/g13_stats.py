"""G13 -- does handing the backbone the statistics RevIN removes close the gap?

Pre-registered in evidence/prereg_ramp_footing.md (with the smoke seed disclosed).

The structural claim behind this paper is that an instance-normalized forecaster
    yhat = ybar + s * f((x - ybar)/s, g)
never exposes ybar or s to f, so f cannot weigh the covariate-implied level
against the window level. If that is really the mechanism, the obvious fix is to
give f those two numbers. This block does exactly that and measures how much of
the gap to covariate-conditioned level handling it closes.

Output: results/g13_stats.csv
Usage: uv run python -m experiments.g13_stats
"""

import csv
import os
import time

import pandas as pd
import torch

from experiments.g4_covfair_full import nn_run
from experiments.g4_grid import build_frame, firststage

ROOT = os.path.join(os.path.dirname(__file__), "..")
PARITY = os.path.join(ROOT, "results", "g4_covfair_full.csv")
OUT = os.path.join(ROOT, "results", "g13_stats.csv")

CELLS = [("gefcom_wind", 24), ("gefcom_wind", 96), ("gefcom_wind", 336),
         ("gefcom_solar", 24), ("gefcom_solar", 96), ("gefcom_solar", 336),
         ("jeju_wind", 24)]
ARMS = ("raw", "revin")
SEEDS = range(5)
BACKBONE = "linmix"
THREADS = 8
FIELDS = ["dataset", "arm", "feed_stats", "backbone", "h", "seed", "L", "mse",
          "threads", "wall_s"]


def main():
    torch.set_num_threads(THREADS)
    lb = (pd.read_csv(PARITY).groupby(["dataset", "backbone", "h"])["L"]
          .agg(lambda c: int(c.mode().iloc[0])).to_dict())
    done = set()
    if os.path.exists(OUT):
        d = pd.read_csv(OUT)
        done = set(zip(d.dataset, d.arm, d.feed_stats, d.h, d.seed))

    plan = [(ds, h, arm, s) for ds, h in CELLS for arm in ARMS for s in SEEDS
            if (ds, arm, True, h, s) not in done]
    print(f"{len(plan)} runs to go")
    new = not os.path.exists(OUT)
    fh = open(OUT, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if new:
        w.writeheader()
    frames = {}
    for ds, h, arm, s in plan:
        if ds not in frames:
            fr = build_frame(ds)
            frames[ds] = (fr, firststage(fr))
        frame, level = frames[ds]
        L = int(lb[(ds, BACKBONE, h)])
        t0 = time.time()
        r = nn_run(frame, h, arm, BACKBONE, s, level, L, feed_stats=True)
        w.writerow({"dataset": ds, "arm": arm, "feed_stats": True,
                    "backbone": BACKBONE, "h": h, "seed": s, "L": L,
                    "mse": round(r["mse"], 6), "threads": THREADS,
                    "wall_s": round(time.time() - t0, 1)})
        fh.flush()
    fh.close()
    print("done")


if __name__ == "__main__":
    main()
