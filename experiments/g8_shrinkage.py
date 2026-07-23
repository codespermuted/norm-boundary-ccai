"""Phase-2 shrinkage safety-valve experiment (pre-registered:
evidence/prereg_shrinkage_arimax.md). Post-hoc; changes no Block A number.

Shrunk level  l~_c = mu_c^train + alpha_c * (ghat_c - mu_c^train), fed to the
CondNorm path of torch_run/lgbm_run as `level` -> the backbone RE-TRAINS on the
shrunk residual series (method A) and the test path denormalizes with l~.

alpha variants (per channel):
  alphahat : clip( Cov(y_val, ghat_val) / Var(ghat_val), 0, 1 )   [primary]
  lpsclip  : clip( official LPS (frozen), 0, 1 ) broadcast          [secondary]

Scope: exogenous {gefcom_wind, jeju_wind} + standard 4; backbones {rlinear,
lgbm_dms}; h {24,96,336} (jeju {24,48}); Block-A seeds (lgbm deterministic).
Writes only results/g8_shrinkage.csv; reuses frozen lookbacks from g4_grid.csv.
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import csv

import numpy as np
import torch

from experiments.g4_grid import (CSV_PATH, LGBM_L, build_frame, firststage,
                                  lgbm_run, torch_run)

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "results", "g8_shrinkage.csv")

SCOPE = {"gefcom_wind": (24, 96, 336), "jeju_wind": (24, 48),
         "etth1": (24, 96, 336), "etth2": (24, 96, 336),
         "weather": (24, 96, 336), "electricity": (24, 96, 336)}
# frozen official LPS (results/gate2 / tab:tau), for the secondary alpha
LPS = {"jeju_wind": 0.745, "gefcom_wind": 0.744, "etth1": -0.717,
       "etth2": -0.205, "weather": 0.110, "electricity": 0.283}
FIELDS = ["dataset", "alpha_kind", "backbone", "h", "seed", "L", "mse",
          "alpha_mean"]


def alpha_hat(frame, level):
    """Per-channel validation-split recalibration slope, clipped to [0,1]."""
    t1, t2, v = frame["t1"], frame["t2"], frame["values"]
    yv, gv = v[t1:t2], level[t1:t2]
    yc, gc = yv - yv.mean(0), gv - gv.mean(0)
    var = (gc * gc).mean(0)
    a = np.where(var > 0, (yc * gc).mean(0) / var, 0.0)
    return np.clip(a, 0.0, 1.0)


def shrunk(frame, level, kind):
    t1, v = frame["t1"], frame["values"]
    mu = v[:t1].mean(0)
    if kind == "alphahat":
        a = alpha_hat(frame, level)                    # (C,)
    else:  # lpsclip: scalar official LPS, broadcast to all channels
        a = np.full(v.shape[1], float(np.clip(LPS[frame["name"]], 0.0, 1.0)))
    return mu + a * (level - mu), float(a.mean())


def frozen_L():
    """{(dataset,backbone,h): L} for condnorm rows in g4_grid.csv."""
    out = {}
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            if r["norm"] == "condnorm":
                out[(r["dataset"], r["backbone"], int(r["h"]))] = int(r["L"])
    return out


def done_keys():
    if not os.path.exists(OUT):
        return set()
    with open(OUT) as f:
        return {(r["dataset"], r["alpha_kind"], r["backbone"], r["h"], r["seed"])
                for r in csv.DictReader(f)}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Ls = frozen_L()
    done = done_keys()
    new = not os.path.exists(OUT)
    f = open(OUT, "a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
        w.writeheader()
    for name, horizons in SCOPE.items():
        frame = build_frame(name)
        level = firststage(frame)
        for kind in ("alphahat", "lpsclip"):
            slevel, amean = shrunk(frame, level, kind)
            for h in horizons:
                # rlinear (5 seeds)
                L = Ls.get((name, "rlinear", h))
                if L is not None:
                    for seed in range(5):
                        key = (name, kind, "rlinear", str(h), str(seed))
                        if key in done:
                            continue
                        r = torch_run(frame, L, h, "rlinear", "condnorm", seed,
                                      device, slevel)
                        w.writerow({"dataset": name, "alpha_kind": kind,
                                    "backbone": "rlinear", "h": h, "seed": seed,
                                    "L": L, "mse": round(r["mse"], 6),
                                    "alpha_mean": round(amean, 4)})
                        f.flush()
                        print(f"{name} {kind} rlinear h{h} s{seed}: "
                              f"mse={r['mse']:.5f} (a~{amean:.3f})", flush=True)
                # lgbm_dms (deterministic, 1)
                key = (name, kind, "lgbm_dms", str(h), "0")
                if key not in done:
                    r = lgbm_run(frame, h, "condnorm", slevel)
                    w.writerow({"dataset": name, "alpha_kind": kind,
                                "backbone": "lgbm_dms", "h": h, "seed": 0,
                                "L": LGBM_L, "mse": round(r["mse"], 6),
                                "alpha_mean": round(amean, 4)})
                    f.flush()
                    print(f"{name} {kind} lgbm_dms h{h}: mse={r['mse']:.5f} "
                          f"(a~{amean:.3f})", flush=True)
    f.close()
    print("g8 shrinkage pass complete", flush=True)


if __name__ == "__main__":
    main()
