"""Information-fair ablation (NOT pre-registered — reported separately).

Question: does CondNorm win on exogenous datasets merely because it SEES
covariates, or because covariate-based RESTORATION is the right injection
mechanism? Every arm here receives the SAME information: past target window
+ past-and-future covariates (lead-matched block over [t-L+1, t+h]).

Backbone: channel-mixing linear (the unconstrained linear class of
Prop 1 / Toner & Darlow):  y_hat = W [x_window ; cov_block] + b

Arms (identical features, differing only in target normalization):
  raw+cov      target globally z-scored
  revin+cov    target window-standardized, restoration = window stats
               (the Prop-1 constrained class: window-mean restoration cannot
               express covariate-dependent level moves)
  condnorm+cov target residualized by first-stage m_hat, restoration = m_hat

Output: results/g4_covfair.csv (+ per-origin losses for DM).
Usage: uv run python -m experiments.g4_covfair
"""

import os

import numpy as np
import torch

from experiments.g4_grid import DATASETS, build_frame, firststage, split_starts

ROOT = os.path.join(os.path.dirname(__file__), "..")
CSV_PATH = os.path.join(ROOT, "results", "g4_covfair.csv")
ERR_DIR = os.path.join(ROOT, "results", "g4_errors")

EXOG = ("jeju_wind", "gefcom_wind", "gefcom_load", "gefcom_solar")
ARMS = ("raw+cov", "revin+cov", "condnorm+cov")
L = 336
SEEDS = range(5)
EPOCHS, PATIENCE, LR, BATCH = 15, 3, 5e-3, 256


class CovMixLinear(torch.nn.Module):
    def __init__(self, lookback, horizon, d_cov):
        super().__init__()
        self.linear = torch.nn.Linear(lookback + (lookback + horizon) * d_cov,
                                      horizon)

    def forward(self, x_win, cov_block):
        return self.linear(torch.cat([x_win, cov_block], dim=1))


def run_one(frame, h, arm, seed, level, device="cpu"):
    torch.manual_seed(seed)
    y = frame["values"][:, 0]
    cov = frame["exog"]
    t1 = frame["t1"]
    mu_g, sd_g = y[:t1].mean(), y[:t1].std()
    cmu, csd = cov[:t1].mean(0), cov[:t1].std(0)
    cov_z = (cov - cmu) / csd

    if arm == "condnorm+cov":
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
    d_cov = cov.shape[1]
    model = CovMixLinear(L, h, d_cov).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    ar_L = torch.arange(L, device=device)
    ar_Lh = torch.arange(L + h, device=device)
    ar_h = torch.arange(h, device=device)

    def gather(s_idx):
        x = st[s_idx[:, None] + ar_L[None, :]]
        cb = ct[s_idx[:, None] + ar_Lh[None, :]].reshape(len(s_idx), -1)
        tgt = st[s_idx[:, None] + L + ar_h[None, :]]
        if arm == "revin+cov":
            m = x.mean(1, keepdim=True)
            s = x.std(1, keepdim=True) + 1e-5
            return (x - m) / s, cb, (tgt - m) / s, (m, s)
        return x, cb, tgt, None

    tr = torch.tensor(sp["train"], dtype=torch.long, device=device)
    va = torch.tensor(sp["val"], dtype=torch.long, device=device)
    te = torch.tensor(sp["test"], dtype=torch.long, device=device)

    def eval_loss(idx):
        model.eval()
        tot = n = 0
        with torch.no_grad():
            for i in range(0, len(idx), BATCH):
                x, cb, tgt, _ = gather(idx[i : i + BATCH])
                tot += torch.sum((model(x, cb) - tgt) ** 2).item()
                n += tgt.numel()
        return tot / n

    best, best_state, bad = float("inf"), None, 0
    for _ in range(EPOCHS):
        model.train()
        perm = tr[torch.randperm(len(tr), device=device)]
        for i in range(0, len(perm), BATCH):
            x, cb, tgt, _ = gather(perm[i : i + BATCH])
            opt.zero_grad()
            torch.nn.functional.mse_loss(model(x, cb), tgt).backward()
            opt.step()
        v = eval_loss(va)
        if v < best:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)

    model.eval()
    losses = []
    with torch.no_grad():
        for i in range(0, len(te), BATCH):
            sb = te[i : i + BATCH]
            x, cb, _, restore = gather(sb)
            pred = model(x, cb)
            tgt_pos = sb[:, None] + L + ar_h[None, :]
            if arm == "revin+cov":
                m, s = restore
                pred_y = (pred * s + m) * sd_g + mu_g
            elif arm == "condnorm+cov":
                pred_y = pred * sd_r + mu_r + lv[tgt_pos]
            else:
                pred_y = pred * sd_g + mu_g
            se = ((pred_y - yt[tgt_pos]) / sd_g) ** 2
            losses.append(se.mean(dim=1).cpu().numpy())
    losses = np.concatenate(losses)
    return {"mse": float(losses.mean()), "losses": losses}


def main():
    import csv

    torch.set_num_threads(12)
    new = not os.path.exists(CSV_PATH)
    done = set()
    if not new:
        with open(CSV_PATH) as f:
            done = {(r["dataset"], r["arm"], r["h"], r["seed"])
                    for r in csv.DictReader(f)}
    f = open(CSV_PATH, "a", newline="")
    w = csv.DictWriter(f, fieldnames=["dataset", "arm", "h", "seed", "L", "mse"])
    if new:
        w.writeheader()
    for name in EXOG:
        frame = build_frame(name)
        level = firststage(frame)
        for h in DATASETS[name]["horizons"]:
            for arm in ARMS:
                for seed in SEEDS:
                    key = (name, arm, str(h), str(seed))
                    if key in done:
                        continue
                    r = run_one(frame, h, arm, seed, level)
                    tag = f"{name}_{arm}_rlinearmix_{h}_{seed}"
                    np.save(os.path.join(ERR_DIR, tag + ".npy"),
                            r.pop("losses"))
                    w.writerow({"dataset": name, "arm": arm, "h": h,
                                "seed": seed, "L": L, "mse": round(r["mse"], 6)})
                    f.flush()
                    print(f"{tag}: {r['mse']:.4f}", flush=True)
    print("covfair complete", flush=True)


if __name__ == "__main__":
    main()
