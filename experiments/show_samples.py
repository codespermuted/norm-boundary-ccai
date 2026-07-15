"""Qualitative check: random test windows per dataset, truth vs predictions
of RAW / RevIN / CondNorm (RLinear, h=24, tuned L). CPU-only so the running
grid is untouched. Output: paper/figures/sample_forecasts.png

Usage: uv run python -m experiments.show_samples [--seed 7]
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.g4_grid import DATASETS, build_frame, firststage, split_starts
from src.models.rlinear import NormWrapper, RLinear
from src.norms import build_norm
from src.theory.figstyle import INK, METHOD_COLORS, apply_paper_style

H = 24
ARMS = ("raw", "revin", "condnorm")
ARM_COLOR = {"raw": METHOD_COLORS["raw"], "revin": METHOD_COLORS["in"],
             "condnorm": METHOD_COLORS["cn"]}


def tuned_L(name):
    import csv

    path = os.path.join(os.path.dirname(__file__), "..", "results",
                        "g4_lookback.csv")
    with open(path) as f:
        for r in csv.DictReader(f):
            if (r["dataset"], r["backbone"], r["h"]) == (name, "rlinear", str(H)):
                return int(r["L"])
    return 336


def train_and_predict(frame, L, arm, level, starts_test, device="cpu",
                      epochs=8):
    torch.manual_seed(0)
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
    st = torch.tensor(series, dtype=torch.float32, device=device)
    sp = split_starts(frame, L, H)
    tr = torch.tensor(sp["train"], dtype=torch.long)
    norm = build_norm("revin" if arm == "revin" else "raw", num_features=C,
                      lookback=L, horizon=H)
    model = NormWrapper(RLinear(L, H, C), norm).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    ar_L, ar_h = torch.arange(L), torch.arange(H)
    batch = max(16, min(256, int(2_000_000 / (L * C))))
    for _ in range(epochs):
        perm = tr[torch.randperm(len(tr))]
        for i in range(0, len(perm), batch):
            j = perm[i : i + batch]
            x = st[j[:, None] + ar_L[None, :]]
            y = st[j[:, None] + L + ar_h[None, :]]
            opt.zero_grad()
            torch.nn.functional.mse_loss(model(x), y).backward()
            opt.step()
    model.eval()
    s = torch.tensor(starts_test, dtype=torch.long)
    with torch.no_grad():
        x = st[s[:, None] + ar_L[None, :]]
        pred = model(x).numpy()
    tgt = starts_test[:, None] + L + np.arange(H)[None, :]
    if arm == "condnorm":
        pred_y = pred * sd_r + mu_r + level[tgt]
    else:
        pred_y = pred * sd_g + mu_g
    return pred_y  # original scale (n, H, C)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-windows", type=int, default=3)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    apply_paper_style()
    names = list(DATASETS)
    fig, axes = plt.subplots(len(names), args.n_windows,
                             figsize=(11, 2.1 * len(names)),
                             constrained_layout=True)

    for i, name in enumerate(names):
        frame = build_frame(name)
        level = firststage(frame)
        L = tuned_L(name)
        sp = split_starts(frame, L, H)
        starts = rng.choice(sp["test"], size=args.n_windows, replace=False)
        preds = {arm: train_and_predict(frame, L, arm, level, starts)
                 for arm in ARMS}
        ch = frame["values"].shape[1] - 1  # OT/target channel (univ: 0)
        for j in range(args.n_windows):
            ax = axes[i, j]
            s = starts[j]
            hist = frame["values"][s + L - 72 : s + L, ch]
            true = frame["values"][s + L : s + L + H, ch]
            tx_h = np.arange(-72, 0)
            tx_f = np.arange(0, H)
            ax.plot(tx_h, hist, color=INK, lw=1.0, alpha=0.55)
            ax.plot(tx_f, true, color=INK, lw=1.6, label="truth")
            for arm in ARMS:
                ax.plot(tx_f, preds[arm][j, :, ch], color=ARM_COLOR[arm],
                        lw=1.3, label=arm if (i, j) == (0, 0) else None)
            ax.axvline(0, color="#d9d8d4", lw=0.8)
            idx_time = frame["index"][s + L]
            if j == 0:
                ax.set_ylabel(name, fontsize=8)
            ax.set_title(str(idx_time)[:16], fontsize=7)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, ["truth", "RAW", "RevIN", "CondNorm"],
               loc="outside upper center", ncol=4, frameon=False)
    out = os.path.join(os.path.dirname(__file__), "..", "paper", "figures",
                       "sample_forecasts.png")
    fig.savefig(out, dpi=150)
    print("saved:", out)


if __name__ == "__main__":
    main()
