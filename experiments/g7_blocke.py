"""G7 Block E — baselines & published-default ablation (audit pass v2).

SELF-CONTAINED block: every arm runs on the SAME chronological splits and
window admissibility rules as the main grid (experiments/g4_grid.py
build_frame / split_starts), and every metric (mse, mae) is reported in the
SAME per-channel global z-score space as Block A — scaler fitted on the
train split only — so rows follow Block A conventions and are directly
comparable to them in scale. Never compare Block E rows against Block B
numbers: revin_all's comparators (linmix_raw / linmix_revin /
linmix_condnorm) are run in-block on the identical setup.

Arms (results/g7_blocke.csv, key dataset,arm,h,seed):
  seasonal_naive   y_hat(t+j) = y(t+j - k*m); m = weekly lag when it fits in
                   the fixed BASE_L lookback, else daily lag; k = smallest
                   multiple that reaches strictly past the forecast origin.
                   Lags are scaled by the dataset step (weather is 10-min:
                   daily = 144 steps, weekly = 1008 > BASE_L -> daily lag).
  climatology      per-channel train mean conditional on (hour-of-day,
                   day-of-week); fallback: hour-of-day mean, then train mean.
                   10-min rows pool within their hour.
  first_stage_only the CondNorm first stage m_hat(x_t) alone as the forecast
                   (src/norms/condnorm.py first_stage_level, train-fitted,
                   grid covariate sets: domain covariates + calendar for the
                   exogenous group, calendar only for the standard group) —
                   isolates how much of CondNorm's win is the first stage.
  dynreg           dynamic-regression class (lagged-dependent + exogenous
                   regressors, ridge; statsmodels is NOT a dependency): 24
                   lags of y at hourly spacing (scaled to the dataset step)
                   + the same covariates at the target time, direct
                   multi-step via per-horizon-step closed-form ridge
                   (alpha=1.0, the LPS ridge sensitivity choice; intercept
                   column penalized too — negligible at these n), solved
                   vectorized across steps, per channel.
  revin_all        published-default ablation on the Block B linmix
                   backbone: instance normalization (lookback-window
                   mean/std per channel, no affine — published use_norm
                   behavior) applied to the COVARIATE channels as well as
                   the target; future covariates are normalized with the
                   lookback-window stats (exactly the window-statistic
                   extrapolation this paper characterizes). Seeds 0-4,
                   epochs 15 (Block B budget).
  linmix_raw / linmix_revin / linmix_condnorm
                   in-block comparators on the identical linmix setup:
                   global-z input / target-only RevIN / CondNorm residual.

Lookbacks: deterministic arms use BASE_L = 336 (= main-grid LGBM_L) so their
test-origin sets coincide across arms 1-4; linmix arms reuse the Block B
tuned L per (dataset, h) recovered from results/g4_covfair_full.csv, and for
the 4 standard datasets absent from Block B tune once on linmix_revin seed-0
val MSE over {96,192,336,720} (grid protocol), frozen across arms
(results/g7_blocke_lookback.csv).

nmae column (jeju_wind only, empty elsewhere): raw-scale MAE divided by the
train-split max of y — a capacity proxy (nameplate capacity is not part of
the curated frame).

Per-origin test losses are saved to results/g7_errors/{dataset}_{arm}_{h}_{seed}.npy
for later DM/MCS use. Rows already in the csv are skipped (resumable).

Usage: uv run python -m experiments.g7_blocke
         [--datasets a,b] [--arms x,y] [--horizons 24,96] [--device auto]
G7_EPOCHS overrides the NN epoch budget (smoke tests only — never for rows
that land in the csv).
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import csv
import time

import numpy as np
import torch

from experiments.g4_covfair_full import (BATCH, EPOCHS as B_EPOCHS, EXOG,
                                         MixBackbone, NN_LR, PATIENCE)
from experiments.g4_grid import (DATASETS, LOOKBACKS, build_frame, firststage,
                                 read_done, split_starts)
from src.norms import build_norm
from src.theory.lps import calendar_features

ROOT = os.path.join(os.path.dirname(__file__), "..")
CSV_PATH = os.path.join(ROOT, "results", "g7_blocke.csv")
LB_CSV = os.path.join(ROOT, "results", "g7_blocke_lookback.csv")
ERR_DIR = os.path.join(ROOT, "results", "g7_errors")
BLOCKB_CSV = os.path.join(ROOT, "results", "g4_covfair_full.csv")

BASE_L = 336                 # deterministic arms: fixed lookback (= LGBM_L)
RIDGE_ALPHA = 1.0            # matches the LPS ridge sensitivity choice
DYN_CHUNK = 32               # horizon steps solved per vectorized block
PW_CHUNK = 128               # test windows per chunk (bounds memory)
DET_ARMS = ("seasonal_naive", "climatology", "first_stage_only", "dynreg")
NN_ARMS = ("revin_all", "linmix_raw", "linmix_revin", "linmix_condnorm")
ARM_ORDER = DET_ARMS + NN_ARMS
SEEDS = range(5)
EPOCHS_E = int(os.environ.get("G7_EPOCHS", B_EPOCHS))   # smoke hook only
FIELDS = ["dataset", "arm", "h", "seed", "mse", "mae", "nmae"]


# ------------------------------------------------------------------ helpers
def steps_per_hour(index) -> int:
    """Dataset step from the index (weather is 10-min -> 6 steps/hour)."""
    step = float(np.median(np.diff(index.values) / np.timedelta64(1, "s")))
    return max(1, round(3600.0 / step))


def global_z(frame):
    """Per-channel global z-score, train stats only (Block A convention)."""
    v = frame["values"]
    t1 = frame["t1"]
    mu, sd = v[:t1].mean(0), v[:t1].std(0)
    return (v - mu) / sd, mu, sd


def cov_matrix(frame) -> np.ndarray:
    """Grid covariate sets: domain exog + calendar (exogenous group),
    calendar only (standard group) — identical to g4_grid.firststage."""
    cal = calendar_features(frame["index"])
    return cal if frame["exog"] is None else np.column_stack([frame["exog"],
                                                              cal])


# ------------------------------------------------------ deterministic arms
def _pointwise(series, frame, h, pred_at, t0):
    """Chunked evaluation of a pointwise predictor pred_at(tgt)->(b,h,C) in
    global-z space over the BASE_L test windows; per-origin losses kept."""
    s_te = split_starts(frame, BASE_L, h)["test"]
    j = np.arange(h)
    C = series.shape[1]
    se_o = np.zeros(len(s_te))
    ae = 0.0
    for i in range(0, len(s_te), PW_CHUNK):
        sb = s_te[i : i + PW_CHUNK]
        tgt = sb[:, None] + BASE_L + j[None, :]
        diff = pred_at(tgt) - series[tgt]
        se_o[i : i + len(sb)] = (diff ** 2).mean(axis=(1, 2))
        ae += np.abs(diff).sum()
    return {"mse": float(se_o.mean()), "mae": float(ae / (len(s_te) * h * C)),
            "losses": se_o, "wall_s": round(time.time() - t0, 2)}


def seasonal_naive_run(frame, h):
    t0 = time.time()
    series, _, _ = global_z(frame)
    sph = steps_per_hour(frame["index"])
    m = 168 * sph if 168 * sph <= BASE_L else 24 * sph
    j = np.arange(h)
    lag = (j // m + 1) * m     # smallest multiple of m strictly past origin
    assert (lag - j).max() <= BASE_L, "seasonal lag escapes the lookback"
    return _pointwise(series, frame, h,
                      lambda tgt: series[tgt - lag[None, :]], t0)


def climatology_run(frame, h):
    t0 = time.time()
    series, _, _ = global_z(frame)
    idx = frame["index"]
    t1 = frame["t1"]
    C = series.shape[1]
    hour = np.asarray(idx.hour)
    key = np.asarray(idx.dayofweek) * 24 + hour
    sums, hsums = np.zeros((168, C)), np.zeros((24, C))
    cnt = np.bincount(key[:t1], minlength=168)
    hcnt = np.bincount(hour[:t1], minlength=24)
    np.add.at(sums, key[:t1], series[:t1])
    np.add.at(hsums, hour[:t1], series[:t1])
    gmean = series[:t1].mean(0)
    hmean = np.where(hcnt[:, None] > 0,
                     hsums / np.maximum(hcnt, 1)[:, None], gmean)
    table = np.where(cnt[:, None] > 0,
                     sums / np.maximum(cnt, 1)[:, None],
                     hmean[np.arange(168) % 24])
    return _pointwise(series, frame, h, lambda tgt: table[key[tgt]], t0)


def first_stage_run(frame, h, level):
    t0 = time.time()
    series, mu, sd = global_z(frame)
    assert level.shape == frame["values"].shape
    level_z = (level - mu) / sd
    return _pointwise(series, frame, h, lambda tgt: level_z[tgt], t0)


def dynreg_run(frame, h, cov_z):
    """Direct multi-step ridge: y(o+j) ~ [24 hourly-spaced lags of y at the
    origin o ; covariates at target time o+j ; 1], closed-form normal
    equations assembled per horizon-step chunk, per channel."""
    t0 = time.time()
    series, _, _ = global_z(frame)
    sph = steps_per_hour(frame["index"])
    lags = np.arange(1, 25) * sph            # one day of hourly-spaced lags
    assert lags.max() <= BASE_L
    sp = split_starts(frame, BASE_L, h)
    o_tr, o_te = sp["train"] + BASE_L, sp["test"] + BASE_L
    d = cov_z.shape[1]
    K = len(lags)
    p = K + d + 1
    C = series.shape[1]
    n_tr, n_te = len(o_tr), len(o_te)
    eye = np.eye(p) * RIDGE_ALPHA
    se_o = np.zeros(n_te)
    ae = 0.0
    for c in range(C):
        Xl = series[o_tr[:, None] - lags[None, :], c]        # (n_tr, K)
        Xl_te = series[o_te[:, None] - lags[None, :], c]
        G_ll = Xl.T @ Xl
        g_l1 = Xl.sum(0)
        for j0 in range(0, h, DYN_CHUNK):
            jc = np.arange(j0, min(j0 + DYN_CHUNK, h))
            m = len(jc)
            Cv = cov_z[jc[:, None] + o_tr[None, :]]          # (m, n_tr, d)
            y = series[jc[:, None] + o_tr[None, :], c]       # (m, n_tr)
            G_lc = np.einsum("nk,mnd->mkd", Xl, Cv, optimize=True)
            G_cc = np.einsum("mnd,mne->mde", Cv, Cv, optimize=True)
            g_c1 = Cv.sum(1)
            A = np.zeros((m, p, p))
            A[:, :K, :K] = G_ll
            A[:, :K, K:K + d] = G_lc
            A[:, K:K + d, :K] = G_lc.transpose(0, 2, 1)
            A[:, K:K + d, K:K + d] = G_cc
            A[:, :K, -1] = g_l1
            A[:, -1, :K] = g_l1
            A[:, K:K + d, -1] = g_c1
            A[:, -1, K:K + d] = g_c1
            A[:, -1, -1] = n_tr
            A += eye
            rhs = np.concatenate(
                [np.einsum("nk,mn->mk", Xl, y, optimize=True),
                 np.einsum("mnd,mn->md", Cv, y, optimize=True),
                 y.sum(1, keepdims=True)], axis=1)
            beta = np.linalg.solve(A, rhs[:, :, None])[:, :, 0]  # (m, p)
            Cv_te = cov_z[jc[:, None] + o_te[None, :]]
            pred = ((Xl_te @ beta[:, :K].T).T
                    + np.einsum("mnd,md->mn", Cv_te, beta[:, K:K + d],
                                optimize=True)
                    + beta[:, -1:])
            diff = pred - series[jc[:, None] + o_te[None, :], c]
            se_o += (diff ** 2).sum(0)
            ae += np.abs(diff).sum()
    losses = se_o / (h * C)
    return {"mse": float(losses.mean()), "mae": float(ae / (n_te * h * C)),
            "losses": losses, "wall_s": round(time.time() - t0, 2)}


# ------------------------------------------------------------ linmix arms
def linmix_run(frame, h, arm, seed, level, L, cov_z, device):
    """Block B linmix protocol (MixBackbone / NN_LR / epochs / patience from
    g4_covfair_full), generalized to multivariate frames by folding
    (window, channel) pairs into the batch — linmix stays channel-
    independent, the covariate block is shared across channels."""
    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    values = frame["values"]
    t1 = frame["t1"]
    C = values.shape[1]
    mu_g, sd_g = values[:t1].mean(0), values[:t1].std(0)
    if arm == "linmix_condnorm":
        resid = values - level
        mu_r, sd_r = resid[:t1].mean(0), resid[:t1].std(0)
        series = (resid - mu_r) / sd_r
    else:
        series = (values - mu_g) / sd_g

    st = torch.tensor(series, dtype=torch.float32, device=device)
    ct = torch.tensor(cov_z, dtype=torch.float32, device=device)
    sp = split_starts(frame, L, h)
    starts = {k: torch.tensor(v, dtype=torch.long, device=device)
              for k, v in sp.items()}
    ar_L = torch.arange(L, device=device)
    ar_h = torch.arange(h, device=device)
    d = cov_z.shape[1]

    model = MixBackbone(L, h, d, "linmix").to(device)
    norm = (build_norm("revin", num_features=1, lookback=L, horizon=h)
            .to(device) if arm == "linmix_revin" else None)
    eps = 1e-5

    def gather(s_idx, c_idx):
        x = st[s_idx[:, None] + ar_L[None, :], c_idx[:, None]]
        cp = ct[s_idx[:, None] + ar_L[None, :]]
        cf = ct[s_idx[:, None] + L + ar_h[None, :]]
        tgt = st[s_idx[:, None] + L + ar_h[None, :], c_idx[:, None]]
        return x, cp, cf, tgt

    def forward(x, cp, cf):
        if arm == "revin_all":
            # published use_norm: lookback-window mean/std per channel on
            # target AND covariates, no affine; future covs get the same
            # window stats (the extrapolation under study)
            xm = x.mean(1, keepdim=True)
            xs = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + eps)
            cm = cp.mean(1, keepdim=True)
            cs = torch.sqrt(cp.var(1, keepdim=True, unbiased=False) + eps)
            out = model((x - xm) / xs, (cp - cm) / cs, (cf - cm) / cs)
            return out * xs + xm
        if arm == "linmix_revin":
            xn = norm(x.unsqueeze(-1), "norm").squeeze(-1)
            return norm(model(xn, cp, cf).unsqueeze(-1), "denorm").squeeze(-1)
        return model(x, cp, cf)

    bp = BATCH if C == 1 else 4096       # (window, channel)-pair batch
    n_tr, n_va = len(sp["train"]), len(sp["val"])
    params = list(model.parameters())
    if norm is not None:
        params += [q for q in norm.parameters() if q.requires_grad]
    opt = torch.optim.Adam(params, lr=NN_LR["linmix"])
    flat_va = torch.arange(n_va * C, device=device)

    def val_mse():
        model.eval()
        tot = n = 0
        with torch.no_grad():
            for i in range(0, len(flat_va), bp):
                fl = flat_va[i : i + bp]
                x, cp, cf, tgt = gather(starts["val"][fl // C], fl % C)
                tot += torch.sum((forward(x, cp, cf) - tgt) ** 2).item()
                n += tgt.numel()
        return tot / n

    best, best_state, bad, ep_run = float("inf"), None, 0, 0
    for ep in range(EPOCHS_E):
        model.train()
        perm = torch.randperm(n_tr * C, device=device)
        for i in range(0, len(perm), bp):
            fl = perm[i : i + bp]
            x, cp, cf, tgt = gather(starts["train"][fl // C], fl % C)
            opt.zero_grad()
            torch.nn.functional.mse_loss(forward(x, cp, cf), tgt).backward()
            opt.step()
        v = val_mse()
        ep_run = ep + 1
        if v < best:
            best, bad = v, 0
            best_state = (
                {k: t.detach().clone() for k, t in model.state_dict().items()},
                {k: t.detach().clone() for k, t in norm.state_dict().items()}
                if norm is not None else None)
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state:
        model.load_state_dict(best_state[0])
        if norm is not None and best_state[1] is not None:
            norm.load_state_dict(best_state[1])

    # test in per-channel global-z space, per-origin losses for DM/MCS
    mu_g_t = torch.tensor(mu_g, dtype=torch.float32, device=device)
    sd_g_t = torch.tensor(sd_g, dtype=torch.float32, device=device)
    vt = torch.tensor(values, dtype=torch.float32, device=device)
    if arm == "linmix_condnorm":
        lv = torch.tensor(level, dtype=torch.float32, device=device)
        mu_r_t = torch.tensor(mu_r, dtype=torch.float32, device=device)
        sd_r_t = torch.tensor(sd_r, dtype=torch.float32, device=device)
    te = starts["test"]
    bw = max(1, bp // C)
    c_all = torch.arange(C, device=device)
    model.eval()
    per_origin, ae_sum, n_el = [], 0.0, 0
    with torch.no_grad():
        for i in range(0, len(te), bw):
            sb = te[i : i + bw]
            s_rep = sb.repeat_interleave(C)
            c_rep = c_all.repeat(len(sb))
            x, cp, cf, _ = gather(s_rep, c_rep)
            pred = forward(x, cp, cf)
            tgt_time = s_rep[:, None] + L + ar_h[None, :]
            true_s = ((vt[tgt_time, c_rep[:, None]] - mu_g_t[c_rep][:, None])
                      / sd_g_t[c_rep][:, None])
            if arm == "linmix_condnorm":
                pred_y = (pred * sd_r_t[c_rep][:, None]
                          + mu_r_t[c_rep][:, None]
                          + lv[tgt_time, c_rep[:, None]])
                pred_s = ((pred_y - mu_g_t[c_rep][:, None])
                          / sd_g_t[c_rep][:, None])
            else:
                pred_s = pred    # series already IS per-channel global z
            se = (pred_s - true_s) ** 2
            per_origin.append(
                se.view(len(sb), C, h).mean(dim=(1, 2)).cpu().numpy())
            ae_sum += torch.sum(torch.abs(pred_s - true_s)).item()
            n_el += se.numel()
    losses = np.concatenate(per_origin)
    return {"mse": float(losses.mean()), "mae": ae_sum / n_el,
            "val_mse": best, "epochs": ep_run, "losses": losses,
            "wall_s": round(time.time() - t0, 2)}


def tune_blocke_L(frame, h, cov_z, device, lb_done, lb_w, lb_f):
    """Grid protocol for the standard datasets absent from Block B:
    linmix_revin seed-0 val MSE over LOOKBACKS, frozen across arms."""
    key = (frame["name"], str(h))
    if key in lb_done and lb_done[key]["L"]:
        return int(lb_done[key]["L"])
    best_L, best_v = None, float("inf")
    for L_try in LOOKBACKS:
        sp = split_starts(frame, L_try, h)
        if min(len(v) for v in sp.values()) < 10:
            continue
        r = linmix_run(frame, h, "linmix_revin", 0, None, L_try, cov_z,
                       device)
        if r["val_mse"] < best_v:     # selection strictly on validation
            best_L, best_v = L_try, r["val_mse"]
    lb_w.writerow({"dataset": frame["name"], "h": h, "L": best_L,
                   "val_mse": round(best_v, 6)})
    lb_f.flush()
    return best_L


def blockb_linmix_L() -> dict:
    """Tuned L per (dataset, h) for linmix, recovered from Block B rows."""
    out = {}
    if os.path.exists(BLOCKB_CSV):
        with open(BLOCKB_CSV) as f:
            for r in csv.DictReader(f):
                if r["backbone"] == "linmix" and r["L"]:
                    out.setdefault((r["dataset"], r["h"]), int(r["L"]))
    return out


# ------------------------------------------------------------------ driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="",
                    help="comma list; empty = all (worker partitioning)")
    ap.add_argument("--arms", default="",
                    help=f"comma subset of {','.join(ARM_ORDER)}")
    ap.add_argument("--horizons", default="",
                    help="comma list; empty = dataset defaults (partitioning)")
    ap.add_argument("--device", default="auto", help="auto|cuda|cpu (NN arms)")
    args = ap.parse_args()

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    if device.type == "cpu":
        torch.set_num_threads(10)

    arm_list = tuple(args.arms.split(",")) if args.arms else ARM_ORDER
    unknown = set(arm_list) - set(ARM_ORDER)
    if unknown:
        raise SystemExit(f"unknown arms: {sorted(unknown)}")
    names = [n for n in DATASETS
             if not args.datasets or n in args.datasets.split(",")]
    h_filter = ({int(x) for x in args.horizons.split(",")}
                if args.horizons else None)

    os.makedirs(ERR_DIR, exist_ok=True)
    done = read_done(CSV_PATH, ("dataset", "arm", "h", "seed"))
    new = not os.path.exists(CSV_PATH)
    f = open(CSV_PATH, "a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
        w.writeheader()
    lb_done = read_done(LB_CSV, ("dataset", "h"))
    lb_new = not os.path.exists(LB_CSV)
    lb_f = open(LB_CSV, "a", newline="")
    lb_w = csv.DictWriter(lb_f, fieldnames=["dataset", "h", "L", "val_mse"])
    if lb_new:
        lb_w.writeheader()

    bb_L = blockb_linmix_L()
    n_run = 0
    for name in names:
        frame = build_frame(name)
        level = firststage(frame)
        cov = cov_matrix(frame)
        t1 = frame["t1"]
        mu_c, sd_c = cov[:t1].mean(0), cov[:t1].std(0)
        sd_c = np.where(sd_c == 0, 1.0, sd_c)
        cov_z = (cov - mu_c) / sd_c
        ymax_tr = frame["values"][:t1, 0].max()    # jeju capacity proxy
        sd0 = frame["values"][:t1].std(0)[0]
        for h in DATASETS[name]["horizons"]:
            if h_filter and h not in h_filter:
                continue
            L_nn = None      # resolved lazily on the first NN arm
            for arm in arm_list:
                for seed in (SEEDS if arm in NN_ARMS else (0,)):
                    key = (name, arm, str(h), str(seed))
                    if key in done:
                        continue
                    if arm in NN_ARMS:
                        if L_nn is None:
                            L_nn = (bb_L.get((name, str(h)))
                                    if name in EXOG else
                                    tune_blocke_L(frame, h, cov_z, device,
                                                  lb_done, lb_w, lb_f))
                        if L_nn is None:
                            print(f"skip {name} {arm} h={h}: no tuned L",
                                  flush=True)
                            break
                        r = linmix_run(frame, h, arm, seed, level, L_nn,
                                       cov_z, device)
                    elif arm == "seasonal_naive":
                        r = seasonal_naive_run(frame, h)
                    elif arm == "climatology":
                        r = climatology_run(frame, h)
                    elif arm == "first_stage_only":
                        r = first_stage_run(frame, h, level)
                    else:
                        r = dynreg_run(frame, h, cov_z)
                    tag = f"{name}_{arm}_{h}_{seed}"
                    np.save(os.path.join(ERR_DIR, tag + ".npy"),
                            r.pop("losses"))
                    nmae = (round(r["mae"] * sd0 / ymax_tr, 6)
                            if name == "jeju_wind" else "")
                    w.writerow({"dataset": name, "arm": arm, "h": h,
                                "seed": seed, "mse": round(r["mse"], 6),
                                "mae": round(r["mae"], 6), "nmae": nmae})
                    f.flush()
                    n_run += 1
                    print(f"[{n_run}] {tag}: mse={r['mse']:.4f} "
                          f"mae={r['mae']:.4f} ({r['wall_s']}s)", flush=True)
    print("block E pass complete", flush=True)


if __name__ == "__main__":
    main()
