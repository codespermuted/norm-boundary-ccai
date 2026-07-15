"""Self-healing overnight supervisor for the G4 experiment queues.

Every CYCLE seconds: find the first incomplete PHASE, ensure each of its
lanes has a live worker (relaunch with the lane's exact GPU/env recipe if
missing — OOM/crash recovery), and advance phases in order:

  A     pre-registered grid remainder (electricity h24 / h336)
  Bcpu  covfair-full linmix/mlpmix/lgbmcov remainder (CPU lane)
  Bdeep covfair-full patchtstcov / segrnncov (one GPU each)
  C     SOTA extension: itransformer(+_ms), timexer(+_ms)

A lane relaunches at most MAX_RELAUNCH times (runaway guard); everything is
resumable so restarts only lose the in-flight run. Prints one status line
per cycle; exits with 'ALL PHASES COMPLETE' when nothing remains.

Usage: uv run python -m experiments.g4_supervisor
"""

import os
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CYCLE = 300
MAX_RELAUNCH = 12

ENV_BASE = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "G4_LGBM_JOBS": "16"}


def grid_done():
    import pandas as pd

    df = pd.read_csv(os.path.join(ROOT, "results", "g4_grid.csv"))
    return df.drop_duplicates(["dataset", "norm", "backbone", "h", "seed"])


def covfair_done():
    import pandas as pd

    p = os.path.join(ROOT, "results", "g4_covfair_full.csv")
    if not os.path.exists(p):
        return pd.DataFrame(columns=["dataset", "arm", "backbone", "h", "seed"])
    return pd.read_csv(p)


EXOG = "jeju_wind,gefcom_wind,gefcom_load,gefcom_solar"


def n_missing_grid(backbones, datasets=None, horizons=None):
    df = grid_done()
    df = df[df.backbone.isin(backbones)]
    total = 0
    from experiments.g4_grid import DATASETS, NORM_ORDER

    for name, meta in DATASETS.items():
        if datasets and name not in datasets:
            continue
        for h in meta["horizons"]:
            if horizons and h not in horizons:
                continue
            for bb in backbones:
                if bb == "lgbm_dms":
                    arms = 3
                    have = len(df[(df.dataset == name) & (df.h == h)
                                  & (df.backbone == bb)])
                    total += max(0, arms - have)
                else:
                    have = len(df[(df.dataset == name) & (df.h == h)
                                  & (df.backbone == bb)])
                    total += max(0, len(NORM_ORDER) * 5 - have)
    return total


def n_missing_covfair(backbones):
    df = covfair_done()
    total = 0
    from experiments.g4_grid import DATASETS

    for name in EXOG.split(","):
        for h in DATASETS[name]["horizons"]:
            for bb in backbones:
                have = len(df[(df.dataset == name) & (df.h == h)
                              & (df.backbone == bb)])
                want = 3 if bb == "lgbmcov" else 25
                total += max(0, want - have)
    return total


def alive(marker):
    r = subprocess.run(["pgrep", "-f", marker], capture_output=True, text=True)
    return bool(r.stdout.strip())


def launch(cmd, gpu=None, extra_env=None, log="supervisor_worker"):
    env = dict(os.environ)
    env.update(ENV_BASE)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if extra_env:
        env.update(extra_env)
    logf = open(os.path.join(ROOT, "results", f"{log}.log"), "a")
    subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=logf,
                     stderr=subprocess.STDOUT, start_new_session=True)


UV = ["uv", "run", "python", "-m"]

LANES = [
    # (phase, name, missing_fn, process_marker, launch_cmd, gpu, extra_env)
    ("A", "elec_h24",
     lambda: n_missing_grid(["rlinear", "patchtst", "segrnn"],
                            {"electricity"}, {24}),
     "g4_grid --datasets electricity --horizons 24 --skip-lgbm",
     UV + ["experiments.g4_grid", "--datasets", "electricity",
           "--horizons", "24", "--skip-lgbm"], 0, None),
    ("A", "elec_h24_gpu1",
     lambda: n_missing_grid(["rlinear", "patchtst", "segrnn"],
                            {"electricity"}, {24}),
     "g4_grid --datasets electricity --horizons 24 --skip-lgbm --norms san,fan",
     UV + ["experiments.g4_grid", "--datasets", "electricity",
           "--horizons", "24", "--skip-lgbm", "--norms", "san,fan"], 1, None),
    ("A", "elec_h336",
     lambda: n_missing_grid(["rlinear", "patchtst", "segrnn"],
                            {"electricity"}, {336}),
     "g4_grid --datasets electricity --horizons 336",
     UV + ["experiments.g4_grid", "--datasets", "electricity",
           "--horizons", "336", "--skip-lgbm"], 1,
     {"G4_BATCH_CAP": "16"}),
    # CPU-only lane: safe to run concurrently with phase A (label shares "A")
    ("A", "covfair_cpu",
     lambda: n_missing_covfair(["linmix", "mlpmix", "lgbmcov"]),
     "g4_covfair_full --backbones linmix,mlpmix",
     UV + ["experiments.g4_covfair_full", "--backbones", "linmix,mlpmix"],
     None, {"CUDA_VISIBLE_DEVICES": ""}),
    ("B", "covfair_patchtstcov",
     lambda: n_missing_covfair(["patchtstcov"]),
     "g4_covfair_full --backbones patchtstcov",
     UV + ["experiments.g4_covfair_full", "--backbones", "patchtstcov",
           "--skip-lgbm"], 0, None),
    ("B", "covfair_segrnncov",
     lambda: n_missing_covfair(["segrnncov"]),
     "g4_covfair_full --backbones segrnncov",
     UV + ["experiments.g4_covfair_full", "--backbones", "segrnncov",
           "--skip-lgbm"], 1, None),
    ("C", "sota_itr_exog_ms",
     lambda: n_missing_grid(["itransformer", "itransformer_ms"],
                            set(EXOG.split(","))),
     "g4_grid --backbones itransformer,itransformer_ms --datasets jeju",
     UV + ["experiments.g4_grid", "--backbones",
           "itransformer,itransformer_ms", "--datasets", EXOG,
           "--skip-lgbm"], 1, None),
    ("C", "sota_timexer_ms",
     lambda: n_missing_grid(["timexer_ms"], set(EXOG.split(","))),
     "g4_grid --backbones timexer_ms",
     UV + ["experiments.g4_grid", "--backbones", "timexer_ms",
           "--datasets", EXOG, "--skip-lgbm"], 0, None),
    ("C2", "sota_itr_std",
     lambda: n_missing_grid(["itransformer"],
                            {"etth1", "etth2", "weather", "electricity"}),
     "g4_grid --backbones itransformer --datasets etth1",
     UV + ["experiments.g4_grid", "--backbones", "itransformer",
           "--datasets", "etth1,etth2,weather,electricity", "--skip-lgbm"],
     1, {"G4_BATCH_CAP": "32"}),
    ("C2", "sota_timexer_ett",
     lambda: n_missing_grid(["timexer"], {"etth1", "etth2"}),
     "g4_grid --backbones timexer --datasets etth1,etth2",
     UV + ["experiments.g4_grid", "--backbones", "timexer",
           "--datasets", "etth1,etth2", "--skip-lgbm"], 0, None),
]


def main():
    relaunches = {name: 0 for _, name, *_ in LANES}
    while True:
        phases = []
        status = []
        for phase, name, missing_fn, marker, cmd, gpu, extra in LANES:
            try:
                miss = missing_fn()
            except Exception as exc:
                status.append(f"{name}: check-error {exc}")
                continue
            if miss > 0:
                phases.append(phase)
                running = alive(marker)
                status.append(f"{name}: {miss} left, "
                              f"{'RUN' if running else 'DEAD'}")
        if not phases:
            print("ALL PHASES COMPLETE", flush=True)
            return
        active_phase = sorted(set(phases))[0]
        for phase, name, missing_fn, marker, cmd, gpu, extra in LANES:
            if phase != active_phase:
                continue
            try:
                if missing_fn() <= 0 or alive(marker):
                    continue
            except Exception:
                continue
            if relaunches[name] >= MAX_RELAUNCH:
                print(f"[supervisor] {name}: relaunch cap hit — manual "
                      f"attention needed", flush=True)
                continue
            relaunches[name] += 1
            print(f"[supervisor] (re)launching {name} "
                  f"(#{relaunches[name]})", flush=True)
            launch(cmd, gpu=gpu, extra_env=extra, log=f"worker_{name}")
        print(f"[supervisor] phase={active_phase} | " + " | ".join(status),
              flush=True)
        time.sleep(CYCLE)


if __name__ == "__main__":
    main()
