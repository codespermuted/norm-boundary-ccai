"""G2 sweep: lambda(11) x h{24,96,336} x L{96,336} x seed(10) x norm(4), RLinear.

Theory-experiment fidelity contract:
  - backbone is the single linear map of the theory, identical across arms;
    the ONLY difference between arms is the normalization
  - arms: raw        global z-score only
          revin      RevIN module inside the model
          cn_oracle  subtract true m_t, add back at target time
          cn_est     subtract LightGBM first-stage m_hat(x_t), add back
  - all test MSEs are reported on the SAME global-z scale of y

Resumable: rows already in results/synth_grid.csv are skipped.

Usage: uv run python -m src.synth.runner [--quick]
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import csv
import itertools
import time

import numpy as np
import torch

from src.models.rlinear import NormWrapper, RLinear
from src.norms import build_norm
from src.synth.dgp import DEFAULTS, generate_series, r2_level

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
CSV_PATH = os.path.join(ROOT, "results", "synth_grid.csv")
CACHE_DIR = os.path.join(ROOT, "curated", "synth")

LAMS = [round(0.1 * i, 1) for i in range(11)]
HORIZONS = (24, 96, 336)
LOOKBACKS = (96, 336)
SEEDS = range(10)
NORMS = ("raw", "revin", "cn_oracle", "cn_est")

TRAIN_FRAC, VAL_FRAC = 0.6, 0.2  # plan §4.1: 6:2:2 chronological

FIELDS = ["lam", "seed", "L", "h", "norm", "mse", "mae", "epochs",
          "r2_level", "sigma_est2", "th_level_term", "wall_s"]


# ---------------------------------------------------------------- first stage
def first_stage_mhat(series: dict, train_end: int) -> np.ndarray:
    """LightGBM g(x_t) -> y_t fitted on train indices only; predicts all t."""
    import lightgbm as lgb

    X = series["x"].reshape(-1, 1)
    model = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        min_child_samples=50, verbose=-1, random_state=0,
    )
    model.fit(X[:train_end], series["y"][:train_end])
    return model.predict(X)


def load_or_build_series(lam: float, seed: int) -> dict:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"lam{lam:.1f}_seed{seed}.npz")
    if os.path.exists(path):
        d = np.load(path)
        return {k: d[k] for k in ("y", "m", "x", "mhat")} | {
            "lam": lam, "seed": seed, "r2": float(d["r2"])}
    s = generate_series(lam, seed)
    train_end = int(len(s["y"]) * TRAIN_FRAC)
    mhat = first_stage_mhat(s, train_end)
    r2 = r2_level(s)
    np.savez(path, y=s["y"], m=s["m"], x=s["x"], mhat=mhat, r2=r2)
    return {"y": s["y"], "m": s["m"], "x": s["x"], "mhat": mhat,
            "lam": lam, "seed": seed, "r2": r2}


# ---------------------------------------------------------------- windowing
def strided_windows(arr: torch.Tensor, L: int, h: int):
    """(N, L) inputs and (N, h) targets from a 1-D tensor, stride 1."""
    T = arr.shape[0]
    n = T - L - h + 1
    idx = torch.arange(n, device=arr.device)
    xw = arr.unfold(0, L, 1)[:n]
    yw = arr.unfold(0, h, 1)[L : L + n]
    return xw, yw, idx  # idx i -> input [i, i+L), target [i+L, i+L+h)


def split_borders(T: int, L: int):
    t1, t2 = int(T * TRAIN_FRAC), int(T * (TRAIN_FRAC + VAL_FRAC))
    return (0, t1), (t1 - L, t2), (t2 - L, T)


# ---------------------------------------------------------------- one run
def run_one(series: dict, L: int, h: int, norm: str, device: torch.device,
            lr: float = 5e-3, epochs: int = 15, patience: int = 3,
            batch: int = 256) -> dict:
    t0 = time.time()
    torch.manual_seed(series["seed"])
    y, m, mhat = series["y"], series["m"], series["mhat"]
    T = len(y)
    (a0, a1), (b0, b1), (c0, c1) = split_borders(T, L)

    mu_g, sd_g = y[a0:a1].mean(), y[a0:a1].std()

    level = {"raw": np.zeros(T), "revin": np.zeros(T),
             "cn_oracle": m, "cn_est": mhat}[norm]
    resid = y - level
    mu_r, sd_r = resid[a0:a1].mean(), resid[a0:a1].std()
    r = (resid - mu_r) / sd_r

    rt = torch.tensor(r, dtype=torch.float32, device=device)
    lv = torch.tensor(level, dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.float32, device=device)

    segs = {}
    for name, (s0, s1) in zip(("train", "val", "test"), ((a0, a1), (b0, b1), (c0, c1))):
        xw, yw, idx = strided_windows(rt[s0:s1], L, h)
        segs[name] = (xw.unsqueeze(-1), yw.unsqueeze(-1), idx + s0)

    backbone = RLinear(lookback=L, horizon=h, num_features=1)
    norm_mod = build_norm("revin" if norm == "revin" else "raw", num_features=1)
    model = NormWrapper(backbone, norm_mod).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def val_mse() -> float:
        model.eval()
        with torch.no_grad():
            xw, yw, _ = segs["val"]
            return torch.mean((model(xw) - yw) ** 2).item()

    xw_tr, yw_tr, _ = segs["train"]
    n_tr = xw_tr.shape[0]
    best, best_state, bad, ep_run = float("inf"), None, 0, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_tr, device=device)
        for i in range(0, n_tr, batch):
            j = perm[i : i + batch]
            opt.zero_grad()
            loss = torch.mean((model(xw_tr[j]) - yw_tr[j]) ** 2)
            loss.backward()
            opt.step()
        v = val_mse()
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

    # test in the common global-z scale of y
    model.eval()
    with torch.no_grad():
        xw, yw, idx = segs["test"]
        pred_r = model(xw).squeeze(-1)
        tgt_pos = idx.unsqueeze(1) + L + torch.arange(h, device=device).unsqueeze(0)
        pred_y = pred_r * sd_r + mu_r + lv[tgt_pos]
        true_y = yt[tgt_pos]
        pred_s = (pred_y - mu_g) / sd_g
        true_s = (true_y - mu_g) / sd_g
        mse = torch.mean((pred_s - true_s) ** 2).item()
        mae = torch.mean(torch.abs(pred_s - true_s)).item()

        # measured M1 level terms on test (docs/theory_g1.md §5 mapping)
        m_t = torch.tensor(m, dtype=torch.float32, device=device)
        m_tgt = m_t[tgt_pos]
        if norm == "raw":
            th = torch.mean(((m_tgt - mu_g) / sd_g) ** 2).item()
        elif norm == "revin":
            ybar_w = yt.unfold(0, L, 1).mean(dim=1)[idx]
            th = torch.mean(((m_tgt - ybar_w[:, None]) / sd_g) ** 2).item()
        elif norm == "cn_oracle":
            th = 0.0
        else:
            mh = torch.tensor(mhat, dtype=torch.float32, device=device)
            th = torch.mean(((m_tgt - mh[tgt_pos]) / sd_g) ** 2).item()

        sigma_est2 = (
            torch.mean(((m_t - torch.tensor(mhat, dtype=torch.float32,
                                            device=device))[c0:] / sd_g) ** 2).item()
            if norm == "cn_est" else float("nan")
        )

    return {"mse": mse, "mae": mae, "epochs": ep_run, "sigma_est2": sigma_est2,
            "th_level_term": th, "wall_s": round(time.time() - t0, 2)}


# ---------------------------------------------------------------- sweep
def done_keys() -> set:
    if not os.path.exists(CSV_PATH):
        return set()
    with open(CSV_PATH) as f:
        return {(r["lam"], r["seed"], r["L"], r["h"], r["norm"])
                for r in csv.DictReader(f)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="3 lams x 1 seed x L=96 smoke sweep")
    args = parser.parse_args()

    lams, seeds, lbs = (LAMS, SEEDS, LOOKBACKS)
    if args.quick:
        lams, seeds, lbs = [0.0, 0.5, 1.0], range(1), (96,)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    done = done_keys()
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    new_file = not os.path.exists(CSV_PATH)
    total = len(lams) * len(list(seeds)) * len(lbs) * len(HORIZONS) * len(NORMS)
    print(f"sweep: {total} runs on {device}, {len(done)} already done", flush=True)

    import mlflow
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI",
                                           f"sqlite:///{os.path.join(ROOT, 'mlflow.db')}"))
    mlflow.set_experiment("norm-boundary")

    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        k = 0
        for lam, seed in itertools.product(lams, seeds):
            series = load_or_build_series(lam, seed)
            for L, h, norm in itertools.product(lbs, HORIZONS, NORMS):
                k += 1
                key = (f"{lam:.1f}", str(seed), str(L), str(h), norm)
                if key in done:
                    continue
                res = run_one(series, L, h, norm, device)
                row = {"lam": f"{lam:.1f}", "seed": seed, "L": L, "h": h,
                       "norm": norm, "r2_level": round(series["r2"], 4), **{
                           k2: (round(v, 6) if isinstance(v, float) else v)
                           for k2, v in res.items()}}
                writer.writerow(row)
                f.flush()
                try:
                    with mlflow.start_run(
                        run_name=f"synth{lam:.1f}_{norm}_rlinear_{h}_{seed}_L{L}"
                    ):
                        mlflow.log_params({"phase": "G2", "lam": lam, "seed": seed,
                                           "L": L, "h": h, "norm": norm})
                        mlflow.log_metrics({"test_mse": res["mse"],
                                            "test_mae": res["mae"]})
                except Exception as exc:  # csv is the primary record
                    print(f"mlflow skip: {exc}", flush=True)
                if k % 60 == 0:
                    print(f"[{k}/{total}] lam={lam:.1f} seed={seed} L={L} h={h} "
                          f"{norm}: mse={res['mse']:.4f}", flush=True)
    print("sweep complete", flush=True)


if __name__ == "__main__":
    main()
