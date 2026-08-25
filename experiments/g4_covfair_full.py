"""FULL information-fair ablation (post-hoc fairness section, NOT
pre-registered): every arm receives past target + past-and-future covariates.

  norms x backbones on the 4 exogenous datasets, 11 (dataset,h) cells, 5 seeds
    arms      : raw+cov, revin+cov, san+cov, fan+cov, condnorm+cov
    backbones : linear-mix (Prop-1 class), mlp-mix (2-layer, nonlinear
                neural representative), lgbm+cov (industry-standard GBM
                with weather features; deterministic, arms raw/winz/condnorm)

SAN/FAN operate on the target channel exactly as in the main grid (stats
pretrain / aux loss); covariates enter through the mixing backbone.

Output: results/g4_covfair_full.csv + per-origin losses.
Usage: uv run python -m experiments.g4_covfair_full [--datasets ...]
"""

import argparse
import csv
import os

import numpy as np
import torch

from experiments.g4_grid import DATASETS, build_frame, firststage, split_starts
from src.models.lgbm_dms import LgbmDMS, window_znorm
from src.norms import build_norm

ROOT = os.path.join(os.path.dirname(__file__), "..")
CSV_PATH = os.path.join(ROOT, "results", "g4_covfair_full.csv")
ERR_DIR = os.path.join(ROOT, "results", "g4_errors")

EXOG = ("jeju_wind", "gefcom_wind", "gefcom_load", "gefcom_solar")
ARMS = ("raw", "revin", "san", "fan", "condnorm")
NN_BACKBONES = ("linmix", "mlpmix")
L = 336          # LGBM+cov fixed L, mirroring the main grid's LGBM_L
LOOKBACKS = (96, 192, 336, 720)   # NN arms: tuned like the main grid
SEEDS = range(5)
EPOCHS, PATIENCE, LR, BATCH = 15, 3, 5e-3, 256
FIELDS = ["dataset", "arm", "backbone", "h", "seed", "L", "mse"]
LB_CSV = os.path.join(ROOT, "results", "g4_covfair_lookback.csv")
# G11: floor on the per-window covariate sd, in globally z-scored units.
# Night windows on GEFCom-Solar have near-zero covariate variance, so the
# unfloored "window" footing divides by ~0 and is unreadable there.
COV_SD_FLOOR = 0.1


class MixBackbone(torch.nn.Module):
    """Linear or MLP map: [target window ; cov block] -> horizon."""

    def __init__(self, lookback, horizon, d_cov, kind):
        super().__init__()
        d_in = lookback + (lookback + horizon) * d_cov
        if kind == "linmix":
            self.net = torch.nn.Linear(d_in, horizon)
        else:
            self.net = torch.nn.Sequential(
                torch.nn.Linear(d_in, 256), torch.nn.ReLU(),
                torch.nn.Dropout(0.1), torch.nn.Linear(256, horizon),
            )

    def forward(self, x_win, cov_past, cov_future):
        cb = torch.cat([cov_past.flatten(1), cov_future.flatten(1)], dim=1)
        return self.net(torch.cat([x_win, cb], dim=1))


def build_cov_backbone(kind, L, h, d_cov):
    if kind in ("linmix", "mlpmix"):
        return MixBackbone(L, h, d_cov, kind)
    from src.models.cov_variants import PatchTSTCov, SegRNNCov

    if kind == "patchtstcov":
        return PatchTSTCov(L, h, d_cov)
    if kind == "segrnncov":
        return SegRNNCov(L, h, d_cov)
    raise KeyError(kind)


NN_LR = {"linmix": 5e-3, "mlpmix": 1e-3, "patchtstcov": 1e-4,
         "segrnncov": 1e-3}


def nn_run(frame, h, arm, backbone, seed, level, L, device="cpu",
           cov_instance_norm=False, cov_footing=None, feed_stats=False):
    """Covariate footing (G11). The series is globally z-scored on train
    statistics before any arm sees it; let s be the per-window standard
    deviation of the target lookback in those units -- the divisor RevIN uses.

        "global"       cov_z                          level kept, global units
        "window"       (cov - mean_w)/sd_w            level DESTROYED, own units
        "window_floor" as "window", sd_w floored      ditto, readable on solar
        "scale"        cov_z / s                      level kept, target's units

    "global" is the frozen parity block; "window" is G10 and changes the units
    *and* the information at once, which is why it cannot adjudicate the
    units-mismatch confound; "scale" is the units-only control that can.

    cov_instance_norm=True is the legacy spelling of cov_footing="window" and
    is kept so the G10 rows reproduce. Default None -> "global", which keeps
    the frozen parity path byte-identical.

    feed_stats=True additionally hands the backbone the two statistics the
    instance-normalization layer removes -- the lookback window mean and scale --
    as extra constant input channels (G13). RevIN's structural cost is that
    yhat = ybar + s * f((x - ybar)/s, g) never exposes ybar or s to f, so if
    simply feeding them back closes the gap to covariate-conditioned level
    handling, the boundary is an implementation oversight rather than a property
    of the layer. That is the test.
    """
    if cov_footing is None:
        cov_footing = "window" if cov_instance_norm else "global"
    if cov_footing not in ("global", "window", "window_floor", "scale",
                           "center_global", "center_scale"):
        raise ValueError(f"unknown cov_footing {cov_footing!r}")
    torch.manual_seed(seed)
    y = frame["values"][:, 0]
    cov = frame["exog"]
    t1 = frame["t1"]
    mu_g, sd_g = y[:t1].mean(), y[:t1].std()
    cov_z = (cov - cov[:t1].mean(0)) / cov[:t1].std(0)

    if arm == "condnorm":
        resid = y - level[:, 0]
        mu_r, sd_r = resid[:t1].mean(), resid[:t1].std()
        series = (resid - mu_r) / sd_r
    else:
        series = (y - mu_g) / sd_g

    st = torch.tensor(series, dtype=torch.float32, device=device)
    ct = torch.tensor(cov_z, dtype=torch.float32, device=device)
    lv = torch.tensor(level[:, 0], dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.float32, device=device)
    sp = split_starts(frame, L, h)

    norm = (build_norm(arm, num_features=1, lookback=L, horizon=h)
            if arm in ("revin", "revin_mean", "san", "fan") else None)
    model = build_cov_backbone(backbone, L, h,
                               cov.shape[1] + (2 if feed_stats else 0)).to(device)
    if norm is not None:
        norm = norm.to(device)
    ar_L = torch.arange(L, device=device)
    ar_h = torch.arange(h, device=device)

    def gather(s_idx):
        x = st[s_idx[:, None] + ar_L[None, :]]
        cp = ct[s_idx[:, None] + ar_L[None, :]]
        cf = ct[s_idx[:, None] + L + ar_h[None, :]]
        tgt = st[s_idx[:, None] + L + ar_h[None, :]]
        return x, (cp, cf), tgt

    def forward(x, cb):
        cp, cf = cb
        if cov_footing in ("window", "window_floor"):
            # per-window, per-channel statistics taken from the lookback block
            m = cp.mean(dim=1, keepdim=True)
            sd = torch.sqrt(cp.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
            if cov_footing == "window_floor":
                # night windows on the solar set have ~zero covariate variance;
                # without a floor the division is numerically meaningless
                sd = torch.clamp(sd, min=COV_SD_FLOOR)
            cp, cf = (cp - m) / sd, (cf - m) / sd
        elif cov_footing == "scale":
            # units-only match: the covariate mean is NOT removed, so the
            # covariate level survives; only the per-instance scaling is put
            # on the same footing as the RevIN-normalized target.
            s = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
            cp, cf = cp / s.unsqueeze(-1), cf / s.unsqueeze(-1)
        elif cov_footing in ("center_global", "center_scale"):
            # LEVEL-ISOLATING footings (G14). Remove the covariate level by
            # subtracting the per-window covariate mean, and leave the divisor
            # alone. "window" changes the centring AND the divisor at once, so
            # the contrast window-vs-scale cannot attribute anything to the
            # level; these two can, because each pairs with a footing that
            # differs from it in the centring only:
            #     RAW  : global -> center_global   (divisor: global, both)
            #     RevIN: scale  -> center_scale    (divisor: /s, both)
            m = cp.mean(dim=1, keepdim=True)
            cp, cf = cp - m, cf - m
            if cov_footing == "center_scale":
                s = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
                cp, cf = cp / s.unsqueeze(-1), cf / s.unsqueeze(-1)
        if feed_stats:
            m = x.mean(dim=1, keepdim=True)
            sd = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
            st2 = torch.stack([m, sd], dim=-1)          # (B, 1, 2)
            cp = torch.cat([cp, st2.expand(-1, cp.shape[1], -1)], dim=-1)
            cf = torch.cat([cf, st2.expand(-1, cf.shape[1], -1)], dim=-1)
        if norm is None:
            return model(x, cp, cf)
        xn = norm(x.unsqueeze(-1), "norm").squeeze(-1)
        out = model(xn, cp, cf)
        return norm(out.unsqueeze(-1), "denorm").squeeze(-1)

    tr = torch.tensor(sp["train"], dtype=torch.long, device=device)
    va = torch.tensor(sp["val"], dtype=torch.long, device=device)
    te = torch.tensor(sp["test"], dtype=torch.long, device=device)

    if norm is not None and getattr(norm, "requires_pretrain", False):
        stat_opt = torch.optim.Adam(list(norm.stats_parameters()), lr=1e-4)
        for _ in range(3):
            perm = tr[torch.randperm(len(tr), device=device)]
            for i in range(0, len(perm), BATCH):
                x, cb, tgt = gather(perm[i : i + BATCH])
                stat_opt.zero_grad()
                norm(x.unsqueeze(-1), "norm")
                norm.stats_loss(tgt.unsqueeze(-1)).backward()
                stat_opt.step()
        for p in norm.stats_parameters():
            p.requires_grad_(False)

    params = list(model.parameters())
    if norm is not None:
        params += [p for p in norm.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=NN_LR[backbone])

    def eval_loss(idx):
        model.eval()
        tot = n = 0
        with torch.no_grad():
            for i in range(0, len(idx), BATCH):
                x, cb, tgt = gather(idx[i : i + BATCH])
                tot += torch.sum((forward(x, cb) - tgt) ** 2).item()
                n += tgt.numel()
        return tot / n

    best, best_state, bad = float("inf"), None, 0
    for _ in range(EPOCHS):
        model.train()
        perm = tr[torch.randperm(len(tr), device=device)]
        for i in range(0, len(perm), BATCH):
            x, cb, tgt = gather(perm[i : i + BATCH])
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(forward(x, cb), tgt)
            if norm is not None and hasattr(norm, "aux_loss"):
                loss = loss + norm.aux_loss(tgt.unsqueeze(-1))
            loss.backward()
            opt.step()
        v = eval_loss(va)
        if v < best:
            best, bad = v, 0
            best_state = ({k: t.clone() for k, t in model.state_dict().items()},
                          {k: t.clone() for k, t in norm.state_dict().items()}
                          if norm is not None else None)
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state[0])
    if norm is not None and best_state[1] is not None:
        norm.load_state_dict(best_state[1])

    model.eval()
    losses = []
    with torch.no_grad():
        for i in range(0, len(te), BATCH):
            sb = te[i : i + BATCH]
            x, cb, _ = gather(sb)
            pred = forward(x, cb)
            tgt_pos = sb[:, None] + L + ar_h[None, :]
            if arm == "condnorm":
                pred_y = pred * sd_r + mu_r + lv[tgt_pos]
            else:
                pred_y = pred * sd_g + mu_g
            se = ((pred_y - yt[tgt_pos]) / sd_g) ** 2
            losses.append(se.mean(dim=1).cpu().numpy())
    losses = np.concatenate(losses)
    return {"mse": float(losses.mean()), "losses": losses, "val_mse": best}


def tune_L(frame, h, backbone, level, lb_done, lb_writer, lb_file):
    """Mirror the main grid: revin+cov seed0 val MSE over LOOKBACKS, frozen
    across every arm afterwards."""
    key = (frame["name"], backbone, str(h))
    if key in lb_done and lb_done[key]["L"]:
        return int(lb_done[key]["L"])
    best_L, best_v = None, float("inf")
    for L_try in LOOKBACKS:
        sp = split_starts(frame, L_try, h)
        if min(len(v) for v in sp.values()) < 10:
            continue
        r = nn_run(frame, h, "revin", backbone, 0, level, L_try)
        if r["val_mse"] < best_v:
            best_L, best_v = L_try, r["val_mse"]
    lb_writer.writerow({"dataset": frame["name"], "backbone": backbone,
                        "h": h, "L": best_L, "val_mse": round(best_v, 6)})
    lb_file.flush()
    return best_L


def lgbm_cov_run(frame, h, arm, level):
    y = frame["values"][:, 0]
    cov = frame["exog"]
    t1 = frame["t1"]
    mu_g, sd_g = y[:t1].mean(), y[:t1].std()
    cov_z = (cov - cov[:t1].mean(0)) / cov[:t1].std(0)
    if arm == "condnorm":
        resid = y - level[:, 0]
        mu_r, sd_r = resid[:t1].mean(), resid[:t1].std()
        series = (resid - mu_r) / sd_r
    else:
        series = (y - mu_g) / sd_g
    sp = split_starts(frame, L, h)

    def feats(idx):
        x = np.stack([series[i : i + L] for i in idx])
        cb = np.stack([cov_z[i : i + L + h].ravel() for i in idx])
        tgt = np.stack([series[i + L : i + L + h] for i in idx])
        return x, cb, tgt

    xtr, ctr, ytr = feats(sp["train"])
    xte, cte, yte = feats(sp["test"])
    sw = None
    if arm == "revin":
        xtr, (mtr, strd) = window_znorm(xtr)
        ytr = (ytr - mtr) / strd
        sw = (strd ** 2).ravel()
        xte, (mte, ste) = window_znorm(xte)
    model = LgbmDMS(horizon=h, n_estimators=100,
                    n_jobs=int(os.environ.get("G4_LGBM_JOBS", -1)))
    model.fit(np.hstack([xtr, ctr]), ytr, sample_weight=sw)
    pred = model.predict(np.hstack([xte, cte]))
    if arm == "revin":
        pred = pred * ste + mte
    tgt_pos = sp["test"][:, None] + L + np.arange(h)[None, :]
    if arm == "condnorm":
        pred_y = pred * sd_r + mu_r + level[tgt_pos, 0]
        true_y = frame["values"][tgt_pos, 0]
        se = ((pred_y - true_y) / sd_g) ** 2
    else:
        se = (pred - yte) ** 2
    losses = se.mean(axis=1)
    return {"mse": float(losses.mean()), "losses": losses}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(EXOG))
    ap.add_argument("--backbones", default=",".join(NN_BACKBONES),
                    help="subset of linmix,mlpmix,patchtstcov,segrnncov")
    ap.add_argument("--skip-lgbm", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(10)

    done = set()
    new = not os.path.exists(CSV_PATH)
    if not new:
        with open(CSV_PATH) as f:
            done = {(r["dataset"], r["arm"], r["backbone"], r["h"], r["seed"])
                    for r in csv.DictReader(f)}
    f = open(CSV_PATH, "a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
        w.writeheader()

    lb_done = {}
    lb_new = not os.path.exists(LB_CSV)
    if not lb_new:
        with open(LB_CSV) as lf:
            lb_done = {(r["dataset"], r["backbone"], r["h"]): r
                       for r in csv.DictReader(lf)}
    lb_file = open(LB_CSV, "a", newline="")
    lb_writer = csv.DictWriter(lb_file, fieldnames=["dataset", "backbone",
                                                    "h", "L", "val_mse"])
    if lb_new:
        lb_writer.writeheader()

    import mlflow
    mlflow.set_tracking_uri(os.environ.get(
        "MLFLOW_TRACKING_URI", f"sqlite:///{os.path.join(ROOT, 'mlflow.db')}"))
    mlflow.set_experiment("norm-boundary")

    for name in args.datasets.split(","):
        frame = build_frame(name)
        level = firststage(frame)
        for h in DATASETS[name]["horizons"]:
            for backbone in args.backbones.split(","):
                L_star = tune_L(frame, h, backbone, level, lb_done,
                                lb_writer, lb_file)
                for arm in ARMS:
                    for seed in SEEDS:
                        key = (name, arm, backbone, str(h), str(seed))
                        if key in done:
                            continue
                        r = nn_run(frame, h, arm, backbone, seed, level, L_star)
                        tag = f"{name}_{arm}+cov_{backbone}_{h}_{seed}"
                        np.save(os.path.join(ERR_DIR, tag + ".npy"),
                                r.pop("losses"))
                        w.writerow({"dataset": name, "arm": arm,
                                    "backbone": backbone, "h": h, "seed": seed,
                                    "L": L_star, "mse": round(r["mse"], 6)})
                        f.flush()
                        try:
                            with mlflow.start_run(run_name=tag):
                                mlflow.log_params({"phase": "G4covfair",
                                                   "dataset": name, "arm": arm,
                                                   "backbone": backbone,
                                                   "h": h, "seed": seed,
                                                   "L": L_star})
                                mlflow.log_metrics({"test_mse": r["mse"]})
                        except Exception as exc:
                            print(f"mlflow skip: {exc}", flush=True)
                        print(f"{tag} L={L_star}: {r['mse']:.4f}", flush=True)
            for arm in ("raw", "revin", "condnorm"):
                if args.skip_lgbm:
                    break
                key = (name, arm, "lgbmcov", str(h), "0")
                if key in done:
                    continue
                r = lgbm_cov_run(frame, h, arm, level)
                tag = f"{name}_{arm}+cov_lgbmcov_{h}_0"
                np.save(os.path.join(ERR_DIR, tag + ".npy"), r.pop("losses"))
                w.writerow({"dataset": name, "arm": arm, "backbone": "lgbmcov",
                            "h": h, "seed": 0, "L": L,
                            "mse": round(r["mse"], 6)})
                f.flush()
                print(f"{tag}: {r['mse']:.4f}", flush=True)
    print("covfair-full complete", flush=True)


if __name__ == "__main__":
    main()
