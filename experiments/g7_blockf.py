"""G7 Block F — probabilistic (quantile) extension of the Block A boundary.

Question: does the RevIN/CondNorm boundary persist (or amplify) when the
target is the predictive DISTRIBUTION rather than the conditional mean?

Design (docs/blockf_design.md has the full rationale):
  backbones : rlinear_q (linear, 9-quantile head, pinball loss)
              lgbm_q   (LightGBM objective='quantile', 3 quantiles)
  arms      : rlinear_q -> raw/revin/san/fan/condnorm (Block A arms;
              affine/predicted-statistics denorm applied to EVERY quantile —
              all denorms here are monotone increasing per element, so
              quantile equivariance holds)
              lgbm_q   -> raw/winz/condnorm (SAN/FAN structurally N/A,
              exactly as in Block A)
  metrics   : per-channel GLOBAL z-score space (the Block A grid scale)
              pinball = mean pinball loss over the quantile set
              crps    = 2 * pinball  (quantile approximation: CRPS equals
                        2x the integral of pinball over q in (0,1); with a
                        finite grid this is the standard Riemann proxy —
                        comparable ACROSS ARMS at fixed quantile set only)
              cov80   = P(q10 <= y <= q90), cov_lo = P(y <= q10),
              cov_hi  = P(y <= q90)
  budget    : rlinear_q epochs 12 / patience 3 / seeds 0-4 (Block A budget),
              lookback = Block A tuned L for (dataset, rlinear, h) read from
              results/g4_grid.csv (capacity identical to Block A rlinear);
              lgbm_q L=336, deterministic seed 0 (as Block A lgbm_dms)
  non-cross : quantiles sorted at inference (rearrangement; training uses
              the unsorted head as is standard)

Output: results/g7_blockf.csv + per-origin pinball in results/g7_errors/.
FROZEN Block A-D artifacts (results/g4_*) are read-only here.

Usage: uv run python -m experiments.g7_blockf \
         [--datasets ...] [--backbones ...] [--arms ...] [--horizons ...]
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
from src.models.lgbm_dms import window_znorm
from src.norms import build_norm

ROOT = os.path.join(os.path.dirname(__file__), "..")
CSV_PATH = os.path.join(ROOT, "results", "g7_blockf.csv")
ERR_DIR = os.path.join(ROOT, "results", "g7_errors")
GRID_CSV = os.path.join(ROOT, "results", "g4_grid.csv")

QUANTILES_NN = tuple(round(0.1 * k, 1) for k in range(1, 10))   # 9: 0.1..0.9
# LightGBM reduced set: h=336 direct multi-step is the known cost bottleneck
# (h x Q separate boosters); {.1,.5,.9} suffices for cov80 and a coarse
# pinball/crps. Documented in docs/blockf_design.md §4.
QUANTILES_LGBM = (0.1, 0.5, 0.9)

NN_ARMS = ("raw", "revin", "san", "fan", "condnorm")
LGBM_ARMS = ("raw", "winz", "condnorm")
BACKBONES = ("rlinear_q", "lgbm_q")
SEEDS = range(5)
EPOCHS, PATIENCE, LR = 12, 3, 5e-3      # Block A rlinear budget
FIELDS = ["dataset", "arm", "backbone", "h", "seed", "pinball", "crps",
          "cov80", "cov_lo", "cov_hi"]


# ------------------------------------------------------------------ lookback
def blocka_lookback(dataset: str, h: int) -> int | None:
    """Tuned L for (dataset, rlinear, h) frozen in the Block A grid."""
    with open(GRID_CSV) as f:
        Ls = {int(r["L"]) for r in csv.DictReader(f)
              if r["dataset"] == dataset and r["backbone"] == "rlinear"
              and r["h"] == str(h) and r["L"]}
    if len(Ls) != 1:  # 0 = cell never ran; >1 would be a grid inconsistency
        return None
    return Ls.pop()


# ------------------------------------------------------------------ model
class RLinearQ(torch.nn.Module):
    """RLinear with a Q-quantile head: shared Linear(L -> h*Q), CI.

    Each quantile level owns its own linear map of the window — per-quantile
    capacity identical to Block A's rlinear (one Linear(L -> h))."""

    def __init__(self, lookback: int, horizon: int, num_quantiles: int):
        super().__init__()
        self.horizon = horizon
        self.Q = num_quantiles
        self.linear = torch.nn.Linear(lookback, horizon * num_quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x.permute(0, 2, 1))              # (B, C, h*Q)
        B, C, _ = out.shape
        return out.reshape(B, C, self.horizon, self.Q).permute(0, 2, 1, 3)


def pinball(pred: torch.Tensor, y: torch.Tensor, qs: torch.Tensor) -> torch.Tensor:
    """pred (..., Q), y (...), qs (Q,) -> scalar mean pinball."""
    diff = y.unsqueeze(-1) - pred
    return torch.maximum(qs * diff, (qs - 1.0) * diff).mean()


# ------------------------------------------------------------------ torch run
def torch_q_run(frame: dict, L: int, h: int, arm: str, seed: int, device,
                level: np.ndarray | None = None, epochs: int = EPOCHS,
                patience: int = PATIENCE) -> dict:
    """rlinear_q under one norm arm. Mirrors g4_grid.torch_run except the
    head/loss are quantile; denorm is applied to every quantile slice (the
    stored/predicted statistics of the single 'norm' call are reused)."""
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
    q_med = Q // 2  # index of q=0.5

    def gather(s_idx):
        x = st[s_idx[:, None] + ar_L[None, :]]
        y = st[s_idx[:, None] + L + ar_h[None, :]]
        return x, y

    norm = build_norm(model_norm, num_features=C, lookback=L, horizon=h)
    net = RLinearQ(L, h, Q)
    norm, net = norm.to(device), net.to(device)

    def forward(x):
        xn = norm(x, "norm")
        pred = net(xn)                                     # (B, h, C, Q)
        # denorm every quantile with the SAME stored/predicted statistics
        # (RevIN affine+window stats / SAN predicted slice stats / FAN
        # predicted main-freq shift): all monotone increasing per element,
        # so quantile levels are preserved.
        return torch.stack([norm(pred[..., q], "denorm") for q in range(Q)],
                           dim=-1)

    batch_cap = int(os.environ.get("G7_BATCH_CAP", 256))
    batch = max(16, min(batch_cap, int(2_000_000 / (L * C))))
    tr = starts["train"]

    if getattr(norm, "requires_pretrain", False):  # SAN: unchanged protocol
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

    best, best_state, bad, ep_run = float("inf"), None, 0, 0
    for ep in range(epochs):
        net.train(); norm.train()
        perm = tr[torch.randperm(len(tr), device=device)]
        for i in range(0, len(perm), batch):
            x, y = gather(perm[i : i + batch])
            opt.zero_grad()
            pred = forward(x)
            loss = pinball(pred, y, qs)
            if hasattr(norm, "aux_loss"):
                # FAN adaptation: the official residual-MSE term regresses
                # the backbone output on the true residual — its quantile
                # analogue is the MEDIAN head (pinball@0.5 targets the
                # median). _pred_residual currently holds the LAST denormed
                # quantile; point it at the median before aux_loss.
                norm._pred_residual = pred[..., q_med] - norm._pred_main
                loss = loss + norm.aux_loss(y)
            loss.backward()
            opt.step()
        v = eval_pinball("val")
        ep_run = ep + 1
        if v < best:
            best, bad = v, 0
            best_state = ({k: t.detach().clone() for k, t in net.state_dict().items()},
                          {k: t.detach().clone() for k, t in norm.state_dict().items()})
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state:
        net.load_state_dict(best_state[0])
        norm.load_state_dict(best_state[1])

    # ---------------- test in per-channel global-z scale ------------------
    lv = (torch.tensor(level, dtype=torch.float32, device=device)
          if arm == "condnorm" else None)
    mu_r_t = (torch.tensor(mu_r, dtype=torch.float32, device=device)
              if arm == "condnorm" else None)
    sd_r_t = (torch.tensor(sd_r, dtype=torch.float32, device=device)
              if arm == "condnorm" else None)
    mu_g_t = torch.tensor(mu_g, dtype=torch.float32, device=device)
    sd_g_t = torch.tensor(sd_g, dtype=torch.float32, device=device)
    vt = torch.tensor(values, dtype=torch.float32, device=device)

    net.eval(); norm.eval()
    pb_sum, n_pb, per_origin = 0.0, 0, []
    in80 = below_lo = below_hi = n_pts = 0
    with torch.no_grad():
        s = starts["test"]
        for i in range(0, len(s), batch):
            sb = s[i : i + batch]
            x, _ = gather(sb)
            pred = forward(x)                              # (B, h, C, Q)
            tgt_pos = sb[:, None] + L + ar_h[None, :]
            if arm == "condnorm":
                pred_y = pred * sd_r_t.view(1, 1, -1, 1) + mu_r_t.view(1, 1, -1, 1) \
                    + lv[tgt_pos].unsqueeze(-1)
                pred_s = (pred_y - mu_g_t.view(1, 1, -1, 1)) / sd_g_t.view(1, 1, -1, 1)
            else:
                pred_s = pred
            # non-crossing: rearrangement (sort) at inference; sorted index
            # k maps to level QUANTILES_NN[k]
            pred_s = torch.sort(pred_s, dim=-1).values
            true_s = (vt[tgt_pos] - mu_g_t) / sd_g_t
            diff = true_s.unsqueeze(-1) - pred_s
            pb = torch.maximum(qs * diff, (qs - 1.0) * diff)
            pb_sum += pb.sum().item()
            n_pb += pb.numel()
            per_origin.append(pb.mean(dim=(1, 2, 3)).cpu().numpy())
            q_lo, q_hi = pred_s[..., 0], pred_s[..., -1]   # q10, q90
            in80 += ((true_s >= q_lo) & (true_s <= q_hi)).sum().item()
            below_lo += (true_s <= q_lo).sum().item()
            below_hi += (true_s <= q_hi).sum().item()
            n_pts += true_s.numel()
    pb_mean = pb_sum / n_pb
    return {"pinball": pb_mean, "crps": 2 * pb_mean,
            "cov80": in80 / n_pts, "cov_lo": below_lo / n_pts,
            "cov_hi": below_hi / n_pts,
            "losses": np.concatenate(per_origin),
            "wall_s": round(time.time() - t0, 2), "epochs": ep_run}


# ------------------------------------------------------------------ lgbm run
def lgbm_q_run(frame: dict, h: int, arm: str, level: np.ndarray | None,
               max_rows: int | None = None) -> dict:
    """LightGBM quantile DMS: one booster per (step, quantile level).

    Pooled CI windows and the row cap follow g4_grid.lgbm_run exactly
    (identical cap for every arm within a dataset — capacity-fair).
    winz sample weight: pinball is 1-homogeneous in scale, so weighting by
    sd (NOT sd^2 — that was the MSE 2-homogeneity convention) makes the
    normalized-space objective equal the original-scale pinball."""
    import lightgbm as lgb

    max_rows = max_rows or int(os.environ.get("G7_LGBM_MAX_ROWS", 250_000))
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
        x = np.stack([series[i : i + L] for i in s])          # (n, L, C)
        y = np.stack([series[i + L : i + L + h] for i in s])  # (n, h, C)
        return (x.transpose(0, 2, 1).reshape(-1, L),
                y.transpose(0, 2, 1).reshape(-1, h), s)

    xtr, ytr, _ = pooled("train")
    if len(xtr) > max_rows:
        sel = np.random.default_rng(0).choice(len(xtr), max_rows, replace=False)
        xtr, ytr = xtr[sel], ytr[sel]
    xte, yte, s_te = pooled("test")

    sw = None
    if arm == "winz":
        xtr, (mtr, strd) = window_znorm(xtr)
        ytr = (ytr - mtr) / strd
        sw = strd.ravel()  # 1-homogeneous pinball: sd, not sd^2
        xte, (mte, ste) = window_znorm(xte)

    qs = np.asarray(QUANTILES_LGBM)
    Q = len(qs)
    pred = np.empty((len(xte), h, Q))
    for j, q in enumerate(qs):
        for step in range(h):
            m = lgb.LGBMRegressor(objective="quantile", alpha=q,
                                  n_estimators=100, learning_rate=0.07,
                                  num_leaves=31, verbose=-1, random_state=0,
                                  n_jobs=n_jobs)
            m.fit(xtr, ytr[:, step], sample_weight=sw)
            pred[:, step, j] = m.predict(xte)
    if arm == "winz":
        pred = pred * ste[..., None] + mte[..., None]

    n = len(s_te)
    pred = pred.reshape(n, C, h, Q).transpose(0, 2, 1, 3)   # (n, h, C, Q)
    true = yte.reshape(n, C, h).transpose(0, 2, 1)          # (n, h, C)
    if arm == "condnorm":
        tgt = s_te[:, None] + L + np.arange(h)[None, :]
        pred_y = pred * sd_r[None, None, :, None] + mu_r[None, None, :, None] \
            + level[tgt][..., None]
        true_y = true * sd_r + mu_r + level[tgt]
        pred_s = (pred_y - mu_g[None, None, :, None]) / sd_g[None, None, :, None]
        true_s = (true_y - mu_g) / sd_g
    else:
        pred_s, true_s = pred, true
    pred_s = np.sort(pred_s, axis=-1)   # non-crossing rearrangement
    diff = true_s[..., None] - pred_s
    pb = np.maximum(qs * diff, (qs - 1.0) * diff)
    q_lo, q_hi = pred_s[..., 0], pred_s[..., -1]            # q10, q90
    pb_mean = float(pb.mean())
    return {"pinball": pb_mean, "crps": 2 * pb_mean,
            "cov80": float(((true_s >= q_lo) & (true_s <= q_hi)).mean()),
            "cov_lo": float((true_s <= q_lo).mean()),
            "cov_hi": float((true_s <= q_hi).mean()),
            "losses": pb.mean(axis=(1, 2, 3)),
            "wall_s": round(time.time() - t0, 2), "epochs": 0}


# ------------------------------------------------------------------ driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="",
                    help="comma list; empty = all 8 (worker partitioning)")
    ap.add_argument("--backbones", default=",".join(BACKBONES),
                    help="subset of rlinear_q,lgbm_q")
    ap.add_argument("--arms", default="",
                    help="comma subset of arms; empty = backbone defaults")
    ap.add_argument("--horizons", default="",
                    help="comma list; empty = dataset defaults")
    ap.add_argument("--max-runs", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(ERR_DIR, exist_ok=True)

    done = set()
    new = not os.path.exists(CSV_PATH)
    if not new:
        with open(CSV_PATH) as f:
            done = {(r["dataset"], r["arm"], r["backbone"], r["h"], r["seed"])
                    for r in csv.DictReader(f)}
    out_f = open(CSV_PATH, "a", newline="")
    w = csv.DictWriter(out_f, fieldnames=FIELDS)
    if new:
        w.writeheader()

    try:
        import mlflow
        mlflow.set_tracking_uri(os.environ.get(
            "MLFLOW_TRACKING_URI", f"sqlite:///{os.path.join(ROOT, 'mlflow.db')}"))
        mlflow.set_experiment("norm-boundary")
    except Exception as exc:  # MLflow optional for Block F
        mlflow = None
        print(f"mlflow disabled: {exc}", flush=True)

    def emit(name, arm, backbone, h, seed, r):
        tag = f"g7_{name}_{arm}_{backbone}_{h}_{seed}"
        np.save(os.path.join(ERR_DIR, f"{name}_{arm}_{backbone}_{h}_{seed}.npy"),
                r.pop("losses"))
        w.writerow({"dataset": name, "arm": arm, "backbone": backbone,
                    "h": h, "seed": seed,
                    **{k: round(r[k], 6) for k in
                       ("pinball", "crps", "cov80", "cov_lo", "cov_hi")}})
        out_f.flush()
        if mlflow is not None:
            try:
                with mlflow.start_run(run_name=tag):
                    mlflow.log_params({"phase": "G7-blockF", "dataset": name,
                                       "arm": arm, "backbone": backbone,
                                       "h": h, "seed": seed})
                    mlflow.log_metrics({"pinball": r["pinball"],
                                        "crps": r["crps"],
                                        "cov80": r["cov80"]})
            except Exception as exc:
                print(f"mlflow skip: {exc}", flush=True)
        print(f"{tag}: pinball={r['pinball']:.4f} cov80={r['cov80']:.3f} "
              f"({r['wall_s']}s)", flush=True)

    n_run = 0
    names = ([n for n in DATASETS if n in args.datasets.split(",")]
             if args.datasets else list(DATASETS))
    backbones = args.backbones.split(",")
    h_filter = ({int(x) for x in args.horizons.split(",")}
                if args.horizons else None)
    for name in names:
        frame = build_frame(name)
        level = firststage(frame)   # cached Block A first stage (read-only)
        for h in DATASETS[name]["horizons"]:
            if h_filter and h not in h_filter:
                continue
            if "rlinear_q" in backbones:
                L = blocka_lookback(name, h)
                if L is None:
                    print(f"skip {name} h={h}: no Block A rlinear L",
                          flush=True)
                else:
                    arms = ([a for a in NN_ARMS if a in args.arms.split(",")]
                            if args.arms else NN_ARMS)
                    for arm in arms:
                        for seed in SEEDS:
                            key = (name, arm, "rlinear_q", str(h), str(seed))
                            if key in done:
                                continue
                            r = torch_q_run(frame, L, h, arm, seed, device,
                                            level if arm == "condnorm" else None)
                            emit(name, arm, "rlinear_q", h, seed, r)
                            n_run += 1
                            if args.max_runs and n_run >= args.max_runs:
                                return
            if "lgbm_q" in backbones:
                arms = ([a for a in LGBM_ARMS if a in args.arms.split(",")]
                        if args.arms else LGBM_ARMS)
                for arm in arms:  # deterministic: seed 0 only
                    key = (name, arm, "lgbm_q", str(h), "0")
                    if key in done:
                        continue
                    r = lgbm_q_run(frame, h, arm,
                                   level if arm == "condnorm" else None)
                    emit(name, arm, "lgbm_q", h, 0, r)
                    n_run += 1
                    if args.max_runs and n_run >= args.max_runs:
                        return
    print("block F pass complete", flush=True)


if __name__ == "__main__":
    main()
