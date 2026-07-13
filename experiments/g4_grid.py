"""G4 factorial grid (plan §5.2), resumable, pre-registration-faithful.

norms 5 x backbones 4 x datasets 8 x horizons (dataset-adjusted) x seeds 5
  - lookback tuned per (dataset, backbone, h) on RevIN seed0 val MSE over
    {96,192,336,720}, then FIXED across every norm arm (capacity identical
    across arms — theory-experiment fidelity invariant)
  - CondNorm: per-channel LightGBM first stage (exog+calendar / calendar),
    fitted on train rows only, cached; exactly invertible restoration
  - per-origin test losses saved to results/g4_errors/*.npy for DM/MCS
  - priority: exogenous datasets first; within a cell, revin/condnorm arms
    first (GATE 2 sign data lands earliest); heavy LTSF datasets last
  - MLflow run name {dataset}_{norm}_{backbone}_{h}_{seed}

Usage: uv run python -m experiments.g4_grid [--only-exog] [--max-runs N]
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import csv
import time

import numpy as np
import torch

from src.data.covariate import segment_ids
from src.data.curation import BUILDERS
from src.data.etth import _TEST_END, _TRAIN_END, _VAL_END, load_etth_frame
from src.data.ltsf import load_ltsf_frame
from src.models import build_backbone
from src.models.lgbm_dms import LgbmDMS, window_znorm
from src.models.rlinear import NormWrapper
from src.norms import build_norm
from src.norms.condnorm import first_stage_level
from src.theory.lps import calendar_features

ROOT = os.path.join(os.path.dirname(__file__), "..")
CSV_PATH = os.path.join(ROOT, "results", "g4_grid.csv")
LOOKBACK_CSV = os.path.join(ROOT, "results", "g4_lookback.csv")
ERR_DIR = os.path.join(ROOT, "results", "g4_errors")
FS_DIR = os.path.join(ROOT, "curated", "firststage")

# priority order: exogenous first (plan), heavy LTSF last
DATASETS = {
    "jeju_wind": {"kind": "curated", "horizons": (24, 48)},
    "gefcom_wind": {"kind": "curated", "horizons": (24, 96, 336)},
    "gefcom_load": {"kind": "curated", "horizons": (24, 96, 336)},
    "gefcom_solar": {"kind": "curated", "horizons": (24, 96, 336)},
    "etth1": {"kind": "etth", "horizons": (24, 96, 336)},
    "etth2": {"kind": "etth", "horizons": (24, 96, 336)},
    "weather": {"kind": "ltsf", "horizons": (24, 96, 336)},
    "electricity": {"kind": "ltsf", "horizons": (24, 96, 336)},
}
NORM_ORDER = ("revin", "condnorm", "raw", "san", "fan")  # GATE-2 arms first
BACKBONES = ("rlinear", "patchtst", "segrnn")            # + lgbm_dms separately
LOOKBACKS = (96, 192, 336, 720)
SEEDS = range(5)
LGBM_L = 336
LR = {"rlinear": 5e-3, "patchtst": 1e-4, "segrnn": 1e-3}
FIELDS = ["dataset", "norm", "backbone", "h", "seed", "L", "mse", "mae",
          "val_mse", "epochs", "wall_s"]


# ------------------------------------------------------------------ frames
def build_frame(name: str) -> dict:
    kind = DATASETS[name]["kind"]
    if kind == "etth":
        df = load_etth_frame(name.replace("etth", "ETTh"))
        values = df.values[:_TEST_END]
        t1, t2 = _TRAIN_END, _VAL_END
        index = df.index[:_TEST_END]
        exog = None
    elif kind == "ltsf":
        df = load_ltsf_frame(name)
        values = df.values
        T = len(values)
        t1, t2 = int(T * 0.7), int(T * 0.8)
        index = df.index
        exog = None
    else:
        df = BUILDERS[name]()
        values = df[["y"]].values
        T = len(values)
        t1, t2 = int(T * 0.6), int(T * 0.8)
        index = df.index
        exog = df.drop(columns=["y"]).values if df.shape[1] > 1 else None
    seg = segment_ids(index)
    return {"name": name, "values": values.astype(np.float64), "seg": seg,
            "t1": t1, "t2": t2, "index": index, "exog": exog}


def firststage(frame: dict) -> np.ndarray:
    """(T, C) CondNorm level matrix, train-fitted, cached."""
    os.makedirs(FS_DIR, exist_ok=True)
    path = os.path.join(FS_DIR, f"{frame['name']}.npy")
    if os.path.exists(path):
        return np.load(path)
    cal = calendar_features(frame["index"])
    feats = (cal if frame["exog"] is None
             else np.column_stack([frame["exog"], cal]))
    T, C = frame["values"].shape
    level = np.empty((T, C))
    for c in range(C):
        level[:, c] = first_stage_level(feats, frame["values"][:, c],
                                        train_end=frame["t1"])
    np.save(path, level)
    return level


# ------------------------------------------------------------------ windows
def valid_starts(seg: np.ndarray, lo: int, hi: int, span: int) -> np.ndarray:
    starts = np.arange(max(lo, 0), hi - span + 1)
    return starts[seg[starts] == seg[starts + span - 1]]


def split_starts(frame: dict, L: int, h: int) -> dict:
    T = len(frame["values"])
    t1, t2 = frame["t1"], frame["t2"]
    span = L + h
    return {
        "train": valid_starts(frame["seg"], 0, t1, span),
        "val": valid_starts(frame["seg"], t1 - L, t2, span),
        "test": valid_starts(frame["seg"], t2 - L, T, span),
    }


# ------------------------------------------------------------------ torch run
def torch_run(frame: dict, L: int, h: int, backbone: str, norm_name: str,
              seed: int, device, level: np.ndarray | None = None,
              epochs: int = 12, patience: int = 3) -> dict:
    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    values = frame["values"]
    t1 = frame["t1"]
    C = values.shape[1]
    mu_g, sd_g = values[:t1].mean(0), values[:t1].std(0)

    if norm_name == "condnorm":
        resid = values - level
        mu_r, sd_r = resid[:t1].mean(0), resid[:t1].std(0)
        series = (resid - mu_r) / sd_r
        model_norm = "raw"
    else:
        series = (values - mu_g) / sd_g
        model_norm = norm_name

    st = torch.tensor(series, dtype=torch.float32, device=device)
    starts = {k: torch.tensor(v, dtype=torch.long, device=device)
              for k, v in split_starts(frame, L, h).items()}
    ar_L = torch.arange(L, device=device)
    ar_h = torch.arange(h, device=device)

    def gather(s_idx):
        x = st[s_idx[:, None] + ar_L[None, :]]
        y = st[s_idx[:, None] + L + ar_h[None, :]]
        return x, y

    norm = build_norm(model_norm, num_features=C, lookback=L, horizon=h)
    bb = build_backbone(backbone, lookback=L, horizon=h, num_features=C)
    model = NormWrapper(bb, norm).to(device)

    batch = max(16, min(256, int(2_000_000 / (L * C))))
    tr = starts["train"]

    if getattr(norm, "requires_pretrain", False):
        stat_opt = torch.optim.Adam(list(norm.stats_parameters()), lr=1e-4)
        for _ in range(3):
            perm = tr[torch.randperm(len(tr), device=device)]
            for i in range(0, len(perm), batch):
                x, y = gather(perm[i : i + batch])
                stat_opt.zero_grad()
                norm(x, "norm")
                norm.stats_loss(y).backward()
                stat_opt.step()
        for p in norm.stats_parameters():
            p.requires_grad_(False)

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=LR[backbone])

    def eval_mse(which: str) -> float:
        model.eval()
        tot = cnt = 0
        with torch.no_grad():
            s = starts[which]
            for i in range(0, len(s), batch):
                x, y = gather(s[i : i + batch])
                tot += torch.sum((model(x) - y) ** 2).item()
                cnt += y.numel()
        return tot / cnt

    best, best_state, bad, ep_run = float("inf"), None, 0, 0
    for ep in range(epochs):
        model.train()
        perm = tr[torch.randperm(len(tr), device=device)]
        for i in range(0, len(perm), batch):
            x, y = gather(perm[i : i + batch])
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(model(x), y)
            if hasattr(norm, "aux_loss"):
                loss = loss + norm.aux_loss(y)
            loss.backward()
            opt.step()
        v = eval_mse("val")
        ep_run = ep + 1
        if v < best:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)

    # test in per-channel global-z scale, per-origin losses for DM/MCS
    lv = (torch.tensor(level, dtype=torch.float32, device=device)
          if norm_name == "condnorm" else None)
    mu_r_t = (torch.tensor(mu_r, dtype=torch.float32, device=device)
              if norm_name == "condnorm" else None)
    sd_r_t = (torch.tensor(sd_r, dtype=torch.float32, device=device)
              if norm_name == "condnorm" else None)
    mu_g_t = torch.tensor(mu_g, dtype=torch.float32, device=device)
    sd_g_t = torch.tensor(sd_g, dtype=torch.float32, device=device)
    vt = torch.tensor(values, dtype=torch.float32, device=device)

    model.eval()
    per_origin, ae_sum, n_elems = [], 0.0, 0
    with torch.no_grad():
        s = starts["test"]
        for i in range(0, len(s), batch):
            sb = s[i : i + batch]
            x, _ = gather(sb)
            pred = model(x)
            tgt_pos = sb[:, None] + L + ar_h[None, :]
            if norm_name == "condnorm":
                pred_y = pred * sd_r_t + mu_r_t + lv[tgt_pos]
                pred_s = (pred_y - mu_g_t) / sd_g_t
            else:
                pred_s = pred
            true_s = (vt[tgt_pos] - mu_g_t) / sd_g_t
            se = (pred_s - true_s) ** 2
            per_origin.append(se.mean(dim=(1, 2)).cpu().numpy())
            ae_sum += torch.sum(torch.abs(pred_s - true_s)).item()
            n_elems += se.numel()
    losses = np.concatenate(per_origin)
    return {"mse": float(losses.mean()), "mae": ae_sum / n_elems,
            "epochs": ep_run, "wall_s": round(time.time() - t0, 2),
            "losses": losses, "val_mse": best}


# ------------------------------------------------------------------ lgbm run
def lgbm_run(frame: dict, h: int, arm: str, level: np.ndarray | None,
             max_rows: int = 250_000) -> dict:
    t0 = time.time()
    L = LGBM_L
    values = frame["values"]
    t1 = frame["t1"]
    C = values.shape[1]
    mu_g, sd_g = values[:t1].mean(0), values[:t1].std(0)
    if arm == "condnorm":
        resid = values - level
        mu_r, sd_r = resid[:t1].mean(0), resid[:t1].std(0)
        series = (resid - mu_r) / sd_r
    else:
        series = (values - mu_g) / sd_g

    sp = split_starts(frame, L, h)

    def pooled(which):
        s = sp[which]
        x = np.stack([series[i : i + L] for i in s])          # (n, L, C)
        y = np.stack([series[i + L : i + L + h] for i in s])  # (n, h, C)
        return (x.transpose(0, 2, 1).reshape(-1, L),
                y.transpose(0, 2, 1).reshape(-1, h), s)

    xtr, ytr, _ = pooled("train")
    if len(xtr) > max_rows:
        sel = np.random.default_rng(0).choice(len(xtr), max_rows, replace=False)
        xtr, ytr = xtr[sel], ytr[sel]
    xte, yte, s_te = pooled("test")

    if arm == "winz":
        xtr, (mtr, strd) = window_znorm(xtr)
        ytr = (ytr - mtr) / strd
        xte, (mte, ste) = window_znorm(xte)
    model = LgbmDMS(horizon=h, n_estimators=100).fit(xtr, ytr)
    pred = model.predict(xte)
    if arm == "winz":
        pred = pred * ste + mte

    n = len(s_te)
    pred = pred.reshape(n, C, h).transpose(0, 2, 1)   # (n, h, C)
    true = yte.reshape(n, C, h).transpose(0, 2, 1)
    if arm == "condnorm":
        tgt = s_te[:, None] + L + np.arange(h)[None, :]
        pred_y = pred * sd_r + mu_r + level[tgt]
        true_y = true * sd_r + mu_r + level[tgt]
        pred_s = (pred_y - mu_g) / sd_g
        true_s = (true_y - mu_g) / sd_g
    else:
        pred_s, true_s = pred, true
    se = (pred_s - true_s) ** 2
    losses = se.mean(axis=(1, 2))
    return {"mse": float(losses.mean()), "mae": float(np.mean(np.abs(pred_s - true_s))),
            "val_mse": float("nan"), "epochs": 0,
            "wall_s": round(time.time() - t0, 2), "losses": losses}


# ------------------------------------------------------------------ driver
def read_done(path, keyfields):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {tuple(r[k] for k in keyfields): r for r in csv.DictReader(f)}


def tune_lookback(frame, h, backbone, device, writer, done):
    key = (frame["name"], backbone, str(h))
    if key in done:
        return int(done[key]["L"])
    best_L, best_v = None, float("inf")
    for L in LOOKBACKS:
        sp = split_starts(frame, L, h)
        if min(len(sp[k]) for k in sp) < 10:
            continue
        r = torch_run(frame, L, h, backbone, "revin", seed=0, device=device,
                      epochs=8, patience=2)
        v = r["val_mse"]  # selection strictly on validation — never test
        if v < best_v:
            best_L, best_v = L, v
    writer.writerow({"dataset": frame["name"], "backbone": backbone,
                     "h": h, "L": best_L, "val_mse": round(best_v, 6)})
    return best_L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-exog", action="store_true")
    ap.add_argument("--max-runs", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(ERR_DIR, exist_ok=True)
    done = read_done(CSV_PATH, ("dataset", "norm", "backbone", "h", "seed"))
    lb_done = read_done(LOOKBACK_CSV, ("dataset", "backbone", "h"))

    import mlflow
    mlflow.set_tracking_uri(os.environ.get(
        "MLFLOW_TRACKING_URI", f"sqlite:///{os.path.join(ROOT, 'mlflow.db')}"))
    mlflow.set_experiment("norm-boundary")

    new_grid = not os.path.exists(CSV_PATH)
    new_lb = not os.path.exists(LOOKBACK_CSV)
    grid_f = open(CSV_PATH, "a", newline="")
    lb_f = open(LOOKBACK_CSV, "a", newline="")
    gw = csv.DictWriter(grid_f, fieldnames=FIELDS)
    lw = csv.DictWriter(lb_f, fieldnames=["dataset", "backbone", "h", "L",
                                          "val_mse"])
    if new_grid:
        gw.writeheader()
    if new_lb:
        lw.writeheader()

    n_run = 0
    names = list(DATASETS)
    if args.only_exog:
        names = names[:4]
    for name in names:
        frame = build_frame(name)
        level = firststage(frame)
        for h in DATASETS[name]["horizons"]:
            # lookback per backbone (fixed across norm arms afterwards)
            Ls = {}
            for backbone in BACKBONES:
                Ls[backbone] = tune_lookback(frame, h, backbone, device,
                                             lw, lb_done)
                lb_f.flush()
            for norm_name in NORM_ORDER:
                for backbone in BACKBONES:
                    L = Ls[backbone]
                    if L is None:
                        continue
                    for seed in SEEDS:
                        key = (name, norm_name, backbone, str(h), str(seed))
                        if key in done:
                            continue
                        r = torch_run(frame, L, h, backbone, norm_name, seed,
                                      device,
                                      level if norm_name == "condnorm" else None)
                        tag = f"{name}_{norm_name}_{backbone}_{h}_{seed}"
                        np.save(os.path.join(ERR_DIR, tag + ".npy"),
                                r.pop("losses"))
                        gw.writerow({"dataset": name, "norm": norm_name,
                                     "backbone": backbone, "h": h,
                                     "seed": seed, "L": L,
                                     **{k: round(v, 6) if isinstance(v, float)
                                        else v for k, v in r.items()}})
                        grid_f.flush()
                        try:
                            with mlflow.start_run(run_name=tag):
                                mlflow.log_params({"phase": "G4", "dataset": name,
                                                   "norm": norm_name,
                                                   "backbone": backbone, "h": h,
                                                   "seed": seed, "L": L})
                                mlflow.log_metrics({"test_mse": r["mse"],
                                                    "test_mae": r["mae"]})
                        except Exception as exc:
                            print(f"mlflow skip: {exc}", flush=True)
                        n_run += 1
                        print(f"[{n_run}] {tag} L={L}: mse={r['mse']:.4f} "
                              f"({r['wall_s']}s)", flush=True)
                        if args.max_runs and n_run >= args.max_runs:
                            return
                # lgbm arms: raw/winz/condnorm mapped from norm order
                lg_arm = {"raw": "raw", "revin": "winz",
                          "condnorm": "condnorm"}.get(norm_name)
                if lg_arm:
                    key = (name, norm_name, "lgbm_dms", str(h), "0")
                    if key not in done:
                        r = lgbm_run(frame, h, lg_arm,
                                     level if lg_arm == "condnorm" else None)
                        tag = f"{name}_{norm_name}_lgbm_dms_{h}_0"
                        np.save(os.path.join(ERR_DIR, tag + ".npy"),
                                r.pop("losses"))
                        gw.writerow({"dataset": name, "norm": norm_name,
                                     "backbone": "lgbm_dms", "h": h, "seed": 0,
                                     "L": LGBM_L,
                                     **{k: round(v, 6) if isinstance(v, float)
                                        else v for k, v in r.items()}})
                        grid_f.flush()
                        n_run += 1
                        print(f"[{n_run}] {tag}: mse={r['mse']:.4f} "
                              f"({r['wall_s']}s)", flush=True)
    print("grid pass complete", flush=True)


if __name__ == "__main__":
    main()
