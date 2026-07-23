"""Targeted verification (evidence/condnorm_val_scale_diagnosis.md, §6.3).

Question: does selecting CondNorm's early-stopping checkpoint on the *global-z*
(channel-reweighted) validation instead of the as-run r-space validation change
electricity's test MSE enough to flip its RevIN-CondNorm sign (as-run gap only
-0.0268)? electricity is the sole at-risk cell: multivariate (321 channels, the
largest reweighting), thinnest margin, lowest-confidence pre-registered sign.

Runs electricity CondNorm with G4_VAL_GLOBALZ on (corrected) and, for rlinear,
also off (a same-process determinism control). Reuses the frozen lookback L.
Writes ONLY results/g4_val_globalz_electricity.csv; never touches g4_grid.csv.
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import csv

import numpy as np
import torch

from experiments.g4_grid import CSV_PATH, build_frame, firststage, torch_run

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "results", "g4_val_globalz_electricity.csv")
NAME = "electricity"
BACKBONES = ("rlinear", "patchtst", "segrnn")
HORIZONS = (24, 96, 336)
SEEDS = range(5)


def frozen():
    """{(backbone,h,seed): (L, as-run test_mse)} for electricity condnorm."""
    out = {}
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            if (r["dataset"] == NAME and r["norm"] == "condnorm"
                    and r["backbone"] in BACKBONES):
                out[(r["backbone"], int(r["h"]), int(r["seed"]))] = (
                    int(r["L"]), float(r["mse"]))
    return out


def asrun_means():
    acc = {"revin": [], "condnorm": []}
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            if r["dataset"] == NAME and r["norm"] in acc:
                acc[r["norm"]].append(float(r["mse"]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fr = build_frame(NAME)
    level = firststage(fr)
    froz = frozen()
    rows = []
    for backbone in BACKBONES:
        for h in HORIZONS:
            for seed in SEEDS:
                key = (backbone, h, seed)
                if key not in froz:
                    continue
                L, asrun_mse = froz[key]
                res = {}
                for flag in (("on", "off") if backbone == "rlinear" else ("on",)):
                    os.environ["G4_VAL_GLOBALZ"] = "1" if flag == "on" else "0"
                    res[flag] = torch_run(fr, L, h, backbone, "condnorm",
                                          seed, device, level)["mse"]
                rows.append({"backbone": backbone, "h": h, "seed": seed, "L": L,
                             "asrun_mse": round(asrun_mse, 6),
                             "gz_on_mse": round(res["on"], 6),
                             "gz_off_mse": (round(res["off"], 6)
                                            if "off" in res else "")})
                print(f"{backbone} h{h} s{seed}: asrun={asrun_mse:.5f} "
                      f"gz_on={res['on']:.5f} "
                      f"gz_off={res.get('off', float('nan')):.5f}"
                      if "off" in res else
                      f"{backbone} h{h} s{seed}: asrun={asrun_mse:.5f} "
                      f"gz_on={res['on']:.5f}", flush=True)
                os.makedirs(os.path.dirname(OUT), exist_ok=True)
                with open(OUT, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=["backbone", "h", "seed",
                                       "L", "asrun_mse", "gz_on_mse", "gz_off_mse"])
                    w.writeheader()
                    w.writerows(rows)
    os.environ.pop("G4_VAL_GLOBALZ", None)

    d = np.array([r["gz_on_mse"] - r["asrun_mse"] for r in rows])
    on_mean = float(np.mean([r["gz_on_mse"] for r in rows]))
    ar = asrun_means()
    print("\n=== SUMMARY ===", flush=True)
    print(f"configs: {len(rows)}   |gz_on - asrun| max={np.abs(d).max():.5f} "
          f"mean={np.abs(d).mean():.5f}", flush=True)
    print(f"electricity CondNorm test-mse, torch backbones: as-run vs gz_on "
          f"means -> {np.mean([r['asrun_mse'] for r in rows]):.5f} vs "
          f"{on_mean:.5f}", flush=True)
    print(f"as-run dataset means: RevIN={ar['revin']:.5f} "
          f"CondNorm={ar['condnorm']:.5f}  gap RevIN-CondNorm="
          f"{ar['revin'] - ar['condnorm']:+.5f} (predicted <0; IN wins)", flush=True)
    print("A sign flip needs CondNorm to drop below RevIN "
          f"({ar['revin']:.4f}); i.e. ~15% improvement. Compare to |delta| above.",
          flush=True)


if __name__ == "__main__":
    main()
