"""G8 — post-hoc quantile recalibration of the Block F probabilistic runs.

Question: Block F found CondNorm's intervals sharp but under-covered on the
exogenous group (cov80 ~0.60 at nominal 0.80; first-stage uncertainty not
propagated). Does a STANDARD split-conformal wrapper — fitted on the
validation segment, applied to every arm identically — restore nominal
coverage, and does CondNorm keep its pinball lead once every arm is
recalibrated?

Method (per (dataset, arm, h, seed), all in the per-channel global-z space
where Block F metrics live):
  1. train exactly as Block F (same arms, budgets, frozen Block A lookbacks;
     experiments/g7_blockf.py is the reference implementation),
  2. predict the VALIDATION segment, compute per-quantile-level additive
     conformal offsets: delta_k = empirical q_k-quantile of the validation
     residuals y - q_hat_k  (so P_val(y <= q_hat_k + delta_k) = q_k exactly),
  3. apply the offsets to the TEST predictions, re-sort (rearrangement),
     re-evaluate pinball / crps / cov80 / cov_lo / cov_hi.
  Variants: "none"    = Block F as-is (reproduction control),
            "pooled"  = one offset per quantile level (primary — the
                        cheapest wrapper a pipeline would deploy),
            "perstep" = offsets per (lead step, level) (sensitivity: the
                        theory says staleness grows with lead).

Caveats (disclosed in the paper): the validation segment is reused — it
early-stops the backbone AND fits the offsets; the test segment is untouched.
Scope: exogenous group (the regime where the paper recommends CN and where
the under-coverage was found). lgbm_q is restricted to h<=48 — Block F logs
show h=96/336 quantile DMS costs 1.3-4.5 h per run (h x Q boosters), and the
GBM row is a cross-check, not the primary endpoint.

Output: results/g8_recal.csv (+ per-origin pinball in results/g8_errors/).
Block A-F artifacts are read-only here.

Usage: uv run python -m experiments.g8_recal \
         [--datasets ...] [--backbones rlinear_q,lgbm_q] [--arms ...]
         [--horizons ...] [--max-runs N]
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import csv
import time

import numpy as np
import torch

from experiments.g4_grid import (DATASETS, LGBM_L, build_frame, firststage,
                                 split_starts)
from experiments.g7_blockf import (EPOCHS, LR, PATIENCE, QUANTILES_LGBM,
                                   QUANTILES_NN, RLinearQ, blocka_lookback,
                                   pinball)
from src.models.lgbm_dms import window_znorm
from src.norms import build_norm

ROOT = os.path.join(os.path.dirname(__file__), "..")
# G15 re-runs this block only to add the interval-width columns; it writes to
# its own file so the frozen g8_recal.csv artifact is never appended to with a
# different header.
CSV_PATH = os.path.join(ROOT, "results",
                        os.environ.get("G8_OUT", "g8_recal.csv"))
ERR_DIR = os.path.join(ROOT, "results", "g8_errors")

EXO_DATASETS = ("jeju_wind", "gefcom_wind", "gefcom_load", "gefcom_solar")
NN_ARMS = ("raw", "revin", "san", "fan", "condnorm")
LGBM_ARMS = ("raw", "winz", "condnorm")
SEEDS = range(5)
VARIANTS = ("none", "pooled", "perstep")
FIELDS = ["dataset", "arm", "backbone", "h", "seed", "variant",
          "width80", "upper_margin",
          "pinball", "crps", "cov80", "cov_lo", "cov_hi", "val_cov80"]


# ------------------------------------------------------------- recalibration
def conformal_offsets(pred_s: np.ndarray, true_s: np.ndarray,
                      qs: np.ndarray, per_step: bool = False) -> np.ndarray:
    """Additive per-quantile-level conformal offsets from a calibration set.

    pred_s (n, h, C, Q) sorted quantile predictions, true_s (n, h, C);
    delta_k is the q_k-quantile of the residuals y - q_hat_k, so adding it
    makes the calibration-set exceedance of level k exactly q_k.
    Returns (Q,) or, with per_step, (h, Q)."""
    resid = true_s[..., None] - pred_s                       # (n, h, C, Q)
    if per_step:
        h, Q = resid.shape[1], resid.shape[3]
        out = np.empty((h, Q))
        for k, q in enumerate(qs):
            out[:, k] = np.quantile(resid[:, :, :, k], q, axis=(0, 2))
        return out
    return np.array([np.quantile(resid[..., k], q)
                     for k, q in enumerate(qs)])


def apply_offsets(pred_s: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Shift each quantile slice and restore monotonicity (rearrangement)."""
    if delta.ndim == 2:                                      # (h, Q)
        shifted = pred_s + delta[None, :, None, :]
    else:                                                    # (Q,)
        shifted = pred_s + delta[None, None, None, :]
    return np.sort(shifted, axis=-1)


def q_metrics(pred_s: np.ndarray, true_s: np.ndarray, qs: np.ndarray) -> dict:
    """Block F metric block on numpy arrays (global-z space)."""
    diff = true_s[..., None] - pred_s
    pb = np.maximum(qs * diff, (qs - 1.0) * diff)
    q_lo, q_hi = pred_s[..., 0], pred_s[..., -1]             # q10, q90
    pb_mean = float(pb.mean())
    q_med = pred_s[..., pred_s.shape[-1] // 2]
    return {"pinball": pb_mean, "crps": 2 * pb_mean,
            "cov80": float(((true_s >= q_lo) & (true_s <= q_hi)).mean()),
            "cov_lo": float((true_s <= q_lo).mean()),
            "cov_hi": float((true_s <= q_hi).mean()),
            # G15: the width a reserve decision actually consumes. cov80 is
            # location; two arms can both cover 0.80 while one interval is
            # materially wider. Reported in global-z units.
            "width80": float((q_hi - q_lo).mean()),
            "upper_margin": float((q_hi - q_med).mean()),
            "losses": pb.mean(axis=(1, 2, 3))}


def recalibrated_rows(val_pred, val_true, te_pred, te_true, qs) -> dict:
    """All three variants evaluated on test, with post-hoc val cov80 sanity."""
    rows = {}
    for variant in VARIANTS:
        if variant == "none":
            te_p, va_p = te_pred, val_pred
        else:
            delta = conformal_offsets(val_pred, val_true, qs,
                                      per_step=(variant == "perstep"))
            te_p, va_p = (apply_offsets(te_pred, delta),
                          apply_offsets(val_pred, delta))
        r = q_metrics(te_p, te_true, qs)
        r["val_cov80"] = q_metrics(va_p, val_true, qs)["cov80"]
        rows[variant] = r
    return rows


# ------------------------------------------------------------- rlinear_q run
def torch_q_recal_run(frame: dict, L: int, h: int, arm: str, seed: int,
                      device, level: np.ndarray | None = None) -> dict:
    """Block F's torch_q_run (mirrored) + validation predictions + conformal.

    Training/early-stopping is byte-for-byte the Block F protocol; the only
    addition is that val AND test predictions are exported to global-z space
    and the offsets of `recalibrated_rows` are applied."""
    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    values = frame["values"]
    t1 = frame["t1"]
    C = values.shape[1]
    mu_g, sd_g = values[:t1].mean(0), values[:t1].std(0)

    if arm == "condnorm":
        resid = values - level
        mu_r, sd_r = resid[:t1].mean(0), resid[:t1].std(0)
        series = (resid - mu_r) / sd_r
        model_norm = "raw"
    else:
        series = (values - mu_g) / sd_g
        model_norm = arm

    st = torch.tensor(series, dtype=torch.float32, device=device)
    starts = {k: torch.tensor(v, dtype=torch.long, device=device)
              for k, v in split_starts(frame, L, h).items()}
    ar_L = torch.arange(L, device=device)
    ar_h = torch.arange(h, device=device)
    qs = torch.tensor(QUANTILES_NN, dtype=torch.float32, device=device)
    Q = len(QUANTILES_NN)
    q_med = Q // 2

    def gather(s_idx):
        x = st[s_idx[:, None] + ar_L[None, :]]
        y = st[s_idx[:, None] + L + ar_h[None, :]]
        return x, y

    norm = build_norm(model_norm, num_features=C, lookback=L, horizon=h)
    net = RLinearQ(L, h, Q)
    norm, net = norm.to(device), net.to(device)

    def forward(x):
        xn = norm(x, "norm")
        pred = net(xn)
        return torch.stack([norm(pred[..., q], "denorm") for q in range(Q)],
                           dim=-1)

    batch_cap = int(os.environ.get("G7_BATCH_CAP", 256))
    batch = max(16, min(batch_cap, int(2_000_000 / (L * C))))
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

    params = list(net.parameters()) + [p for p in norm.parameters()
                                       if p.requires_grad]
    opt = torch.optim.Adam(params, lr=LR)

    def eval_pinball(which: str) -> float:
        net.eval(); norm.eval()
        tot = cnt = 0
        with torch.no_grad():
            s = starts[which]
            for i in range(0, len(s), batch):
                x, y = gather(s[i : i + batch])
                p = forward(x)
                diff = y.unsqueeze(-1) - p
                tot += torch.sum(torch.maximum(qs * diff,
                                               (qs - 1.0) * diff)).item()
                cnt += diff.numel()
        return tot / cnt

    best, best_state, bad = float("inf"), None, 0
    for ep in range(EPOCHS):
        net.train(); norm.train()
        perm = tr[torch.randperm(len(tr), device=device)]
        for i in range(0, len(perm), batch):
            x, y = gather(perm[i : i + batch])
            opt.zero_grad()
            pred = forward(x)
            loss = pinball(pred, y, qs)
            if hasattr(norm, "aux_loss"):
                norm._pred_residual = pred[..., q_med] - norm._pred_main
                loss = loss + norm.aux_loss(y)
            loss.backward()
            opt.step()
        v = eval_pinball("val")
        if v < best:
            best, bad = v, 0
            best_state = ({k: t.detach().clone() for k, t in net.state_dict().items()},
                          {k: t.detach().clone() for k, t in norm.state_dict().items()})
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state:
        net.load_state_dict(best_state[0])
        norm.load_state_dict(best_state[1])

    # ---------- export val/test predictions in global-z space ----------
    lv = (torch.tensor(level, dtype=torch.float32, device=device)
          if arm == "condnorm" else None)
    mu_r_t = (torch.tensor(mu_r, dtype=torch.float32, device=device)
              if arm == "condnorm" else None)
    sd_r_t = (torch.tensor(sd_r, dtype=torch.float32, device=device)
              if arm == "condnorm" else None)
    mu_g_t = torch.tensor(mu_g, dtype=torch.float32, device=device)
    sd_g_t = torch.tensor(sd_g, dtype=torch.float32, device=device)
    vt = torch.tensor(values, dtype=torch.float32, device=device)

    def predict_globalz(which: str):
        net.eval(); norm.eval()
        preds, trues = [], []
        with torch.no_grad():
            s = starts[which]
            for i in range(0, len(s), batch):
                sb = s[i : i + batch]
                x, _ = gather(sb)
                pred = forward(x)                            # (B, h, C, Q)
                tgt_pos = sb[:, None] + L + ar_h[None, :]
                if arm == "condnorm":
                    pred_y = (pred * sd_r_t.view(1, 1, -1, 1)
                              + mu_r_t.view(1, 1, -1, 1)
                              + lv[tgt_pos].unsqueeze(-1))
                    pred_s = (pred_y - mu_g_t.view(1, 1, -1, 1)) \
                        / sd_g_t.view(1, 1, -1, 1)
                else:
                    pred_s = pred
                pred_s = torch.sort(pred_s, dim=-1).values   # rearrangement
                true_s = (vt[tgt_pos] - mu_g_t) / sd_g_t
                preds.append(pred_s.cpu().numpy())
                trues.append(true_s.cpu().numpy())
        return np.concatenate(preds), np.concatenate(trues)

    val_pred, val_true = predict_globalz("val")
    te_pred, te_true = predict_globalz("test")
    rows = recalibrated_rows(val_pred, val_true, te_pred, te_true,
                             np.asarray(QUANTILES_NN))
    rows["wall_s"] = round(time.time() - t0, 2)
    return rows


# --------------------------------------------------------------- lgbm_q run
def lgbm_q_recal_run(frame: dict, h: int, arm: str,
                     level: np.ndarray | None) -> dict:
    """Block F's lgbm_q_run (mirrored) + validation predictions + conformal."""
    import lightgbm as lgb

    max_rows = int(os.environ.get("G7_LGBM_MAX_ROWS", 250_000))
    n_jobs = int(os.environ.get("G7_LGBM_JOBS", 8))
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
        x = np.stack([series[i : i + L] for i in s])
        y = np.stack([series[i + L : i + L + h] for i in s])
        return (x.transpose(0, 2, 1).reshape(-1, L),
                y.transpose(0, 2, 1).reshape(-1, h), s)

    xtr, ytr, _ = pooled("train")
    if len(xtr) > max_rows:
        sel = np.random.default_rng(0).choice(len(xtr), max_rows, replace=False)
        xtr, ytr = xtr[sel], ytr[sel]
    xva, yva, s_va = pooled("val")
    xte, yte, s_te = pooled("test")

    sw = None
    if arm == "winz":
        xtr, (mtr, strd) = window_znorm(xtr)
        ytr = (ytr - mtr) / strd
        sw = strd.ravel()
        xva, (mva, sva) = window_znorm(xva)
        xte, (mte, ste) = window_znorm(xte)

    qs = np.asarray(QUANTILES_LGBM)
    Q = len(qs)
    pred_va = np.empty((len(xva), h, Q))
    pred_te = np.empty((len(xte), h, Q))
    for j, q in enumerate(qs):
        for step in range(h):
            m = lgb.LGBMRegressor(objective="quantile", alpha=q,
                                  n_estimators=100, learning_rate=0.07,
                                  num_leaves=31, verbose=-1, random_state=0,
                                  n_jobs=n_jobs)
            m.fit(xtr, ytr[:, step], sample_weight=sw)
            pred_va[:, step, j] = m.predict(xva)
            pred_te[:, step, j] = m.predict(xte)
    if arm == "winz":
        pred_va = pred_va * sva[..., None] + mva[..., None]
        pred_te = pred_te * ste[..., None] + mte[..., None]

    def to_globalz(pred, ytrue, s_idx):
        n = len(s_idx)
        pred = pred.reshape(n, C, h, Q).transpose(0, 2, 1, 3)  # (n, h, C, Q)
        true = ytrue.reshape(n, C, h).transpose(0, 2, 1)       # (n, h, C)
        if arm == "condnorm":
            tgt = s_idx[:, None] + L + np.arange(h)[None, :]
            pred_y = (pred * sd_r[None, None, :, None]
                      + mu_r[None, None, :, None] + level[tgt][..., None])
            true_y = true * sd_r + mu_r + level[tgt]
            pred_s = (pred_y - mu_g[None, None, :, None]) \
                / sd_g[None, None, :, None]
            true_s = (true_y - mu_g) / sd_g
        else:
            pred_s, true_s = pred, true
        return np.sort(pred_s, axis=-1), true_s

    val_pred, val_true = to_globalz(pred_va, yva, s_va)
    te_pred, te_true = to_globalz(pred_te, yte, s_te)
    rows = recalibrated_rows(val_pred, val_true, te_pred, te_true, qs)
    rows["wall_s"] = round(time.time() - t0, 2)
    return rows


# ------------------------------------------------------------------ driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(EXO_DATASETS),
                    help="comma list (default: exogenous group)")
    ap.add_argument("--backbones", default="rlinear_q",
                    help="subset of rlinear_q,lgbm_q")
    ap.add_argument("--arms", default="",
                    help="comma subset of arms; empty = backbone defaults")
    ap.add_argument("--horizons", default="",
                    help="comma list; empty = dataset defaults "
                         "(lgbm_q additionally capped at h<=48, see header)")
    ap.add_argument("--max-runs", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(ERR_DIR, exist_ok=True)

    done = set()
    new = not os.path.exists(CSV_PATH)
    if not new:
        with open(CSV_PATH) as f:
            done = {(r["dataset"], r["arm"], r["backbone"], r["h"], r["seed"])
                    for r in csv.DictReader(f) if r["variant"] == "perstep"}
    out_f = open(CSV_PATH, "a", newline="")
    w = csv.DictWriter(out_f, fieldnames=FIELDS)
    if new:
        w.writeheader()

    try:
        import mlflow
        mlflow.set_tracking_uri(os.environ.get(
            "MLFLOW_TRACKING_URI", f"sqlite:///{os.path.join(ROOT, 'mlflow.db')}"))
        mlflow.set_experiment("norm-boundary")
    except Exception as exc:
        mlflow = None
        print(f"mlflow disabled: {exc}", flush=True)

    def emit(name, arm, backbone, h, seed, rows):
        wall = rows.pop("wall_s")
        for variant, r in rows.items():
            np.save(os.path.join(
                ERR_DIR, f"{name}_{arm}_{backbone}_{h}_{seed}_{variant}.npy"),
                r.pop("losses"))
            w.writerow({"dataset": name, "arm": arm, "backbone": backbone,
                        "h": h, "seed": seed, "variant": variant,
                        **{k: round(r[k], 6) for k in
                           ("width80", "upper_margin",
                            "pinball", "crps", "cov80", "cov_lo", "cov_hi",
                            "val_cov80")}})
        out_f.flush()
        tag = f"g8_{name}_{arm}_{backbone}_{h}_{seed}"
        if mlflow is not None:
            try:
                with mlflow.start_run(run_name=tag):
                    mlflow.log_params({"phase": "G8-recal", "dataset": name,
                                       "arm": arm, "backbone": backbone,
                                       "h": h, "seed": seed})
                    for variant, r in rows.items():
                        mlflow.log_metrics({f"{variant}_pinball": r["pinball"],
                                            f"{variant}_cov80": r["cov80"]})
            except Exception as exc:
                print(f"mlflow skip: {exc}", flush=True)
        print(f"{tag}: cov80 {rows['none']['cov80']:.3f} -> "
              f"{rows['pooled']['cov80']:.3f} (perstep "
              f"{rows['perstep']['cov80']:.3f}), pinball "
              f"{rows['none']['pinball']:.4f} -> "
              f"{rows['pooled']['pinball']:.4f} ({wall}s)", flush=True)

    n_run = 0
    names = [n for n in DATASETS if n in args.datasets.split(",")]
    backbones = args.backbones.split(",")
    h_filter = ({int(x) for x in args.horizons.split(",")}
                if args.horizons else None)
    for name in names:
        frame = build_frame(name)
        level = firststage(frame)
        for h in DATASETS[name]["horizons"]:
            if h_filter and h not in h_filter:
                continue
            if "rlinear_q" in backbones:
                L = blocka_lookback(name, h)
                if L is None:
                    print(f"skip {name} h={h}: no Block A rlinear L", flush=True)
                else:
                    arms = ([a for a in NN_ARMS if a in args.arms.split(",")]
                            if args.arms else NN_ARMS)
                    for arm in arms:
                        for seed in SEEDS:
                            key = (name, arm, "rlinear_q", str(h), str(seed))
                            if key in done:
                                continue
                            rows = torch_q_recal_run(
                                frame, L, h, arm, seed, device,
                                level if arm == "condnorm" else None)
                            emit(name, arm, "rlinear_q", h, seed, rows)
                            n_run += 1
                            if args.max_runs and n_run >= args.max_runs:
                                return
            if "lgbm_q" in backbones and h <= 48:
                arms = ([a for a in LGBM_ARMS if a in args.arms.split(",")]
                        if args.arms else LGBM_ARMS)
                for arm in arms:
                    key = (name, arm, "lgbm_q", str(h), "0")
                    if key in done:
                        continue
                    rows = lgbm_q_recal_run(
                        frame, h, arm, level if arm == "condnorm" else None)
                    emit(name, arm, "lgbm_q", h, 0, rows)
                    n_run += 1
                    if args.max_runs and n_run >= args.max_runs:
                        return
    print("g8 recal pass complete", flush=True)


if __name__ == "__main__":
    main()
