"""Same test windows as show_samples.py, but the INFORMATION-FAIR arms:
raw+cov / revin+cov / condnorm+cov (CovMixLinear, identical features).
Exogenous datasets only. Output: paper/figures/sample_forecasts_covfair.png

Usage: uv run python -m experiments.show_samples_covfair
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.g4_covfair import ARMS, CovMixLinear, run_one  # noqa: F401
from experiments.g4_grid import build_frame, firststage, split_starts
from experiments.show_samples import tuned_L
from src.theory.figstyle import INK, METHOD_COLORS, apply_paper_style

H, L = 24, 336
EXOG = ("jeju_wind", "gefcom_wind", "gefcom_load", "gefcom_solar")
ARM_COLOR = {"raw+cov": METHOD_COLORS["raw"], "revin+cov": METHOD_COLORS["in"],
             "condnorm+cov": METHOD_COLORS["cn"]}


def train_predict_at(frame, arm, level, starts, device="cpu", epochs=10):
    """Train CovMixLinear (same recipe as g4_covfair) and predict at starts."""
    torch.manual_seed(0)
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
    st = torch.tensor(series, dtype=torch.float32)
    ct = torch.tensor(cov_z, dtype=torch.float32)
    sp = split_starts(frame, L, H)
    model = CovMixLinear(L, H, cov.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    ar_L, ar_Lh, ar_h = torch.arange(L), torch.arange(L + H), torch.arange(H)

    def gather(s_idx):
        x = st[s_idx[:, None] + ar_L[None, :]]
        cb = ct[s_idx[:, None] + ar_Lh[None, :]].reshape(len(s_idx), -1)
        tgt = st[s_idx[:, None] + L + ar_h[None, :]]
        if arm == "revin+cov":
            m = x.mean(1, keepdim=True)
            s = x.std(1, keepdim=True) + 1e-5
            return (x - m) / s, cb, (tgt - m) / s, (m, s)
        return x, cb, tgt, None

    tr = torch.tensor(sp["train"], dtype=torch.long)
    for _ in range(epochs):
        perm = tr[torch.randperm(len(tr))]
        for i in range(0, len(perm), 256):
            x, cb, tgt, _ = gather(perm[i : i + 256])
            opt.zero_grad()
            torch.nn.functional.mse_loss(model(x, cb), tgt).backward()
            opt.step()
    model.eval()
    s = torch.tensor(starts, dtype=torch.long)
    with torch.no_grad():
        x, cb, _, restore = gather(s)
        pred = model(x, cb).numpy()
    tgt = starts[:, None] + L + np.arange(H)[None, :]
    if arm == "revin+cov":
        m, sd = (t.numpy() for t in restore)
        return (pred * sd + m) * sd_g + mu_g
    if arm == "condnorm+cov":
        return pred * sd_r + mu_r + level[tgt, 0]
    return pred * sd_g + mu_g


def main():
    rng = np.random.default_rng(7)
    apply_paper_style()
    fig, axes = plt.subplots(len(EXOG), 3, figsize=(11, 2.2 * len(EXOG)),
                             constrained_layout=True)
    for i, name in enumerate(EXOG):
        frame = build_frame(name)
        level = firststage(frame)
        # same forecast ORIGINS as show_samples.py (regenerate its sampling)
        L_prev = tuned_L(name)
        prev_starts = rng.choice(split_starts(frame, L_prev, H)["test"],
                                 size=3, replace=False)
        valid = set(split_starts(frame, L, H)["test"])
        starts = np.array([s + L_prev - L if (s + L_prev - L) in valid
                           else min(valid, key=lambda v: abs(v - (s + L_prev - L)))
                           for s in prev_starts])
        preds = {arm: train_predict_at(frame, arm, level, starts)
                 for arm in ARMS}
        for j in range(3):
            ax = axes[i, j]
            s = starts[j]
            hist = frame["values"][s + L - 72 : s + L, 0]
            true = frame["values"][s + L : s + L + H, 0]
            ax.plot(np.arange(-72, 0), hist, color=INK, lw=1.0, alpha=0.55)
            ax.plot(np.arange(H), true, color=INK, lw=1.6)
            for arm in ARMS:
                ax.plot(np.arange(H), preds[arm][j], color=ARM_COLOR[arm], lw=1.3)
            ax.axvline(0, color="#d9d8d4", lw=0.8)
            ax.set_title(str(frame["index"][s + L])[:16], fontsize=7)
            if j == 0:
                ax.set_ylabel(name, fontsize=8)
    from matplotlib.lines import Line2D

    fig.legend(handles=[Line2D([], [], color=c, lw=2) for c in
                        [INK] + [ARM_COLOR[a] for a in ARMS]],
               labels=["truth", "RAW+cov", "RevIN+cov", "CondNorm+cov"],
               loc="outside upper center", ncol=4, frameon=False)
    out = os.path.join(os.path.dirname(__file__), "..", "paper", "figures",
                       "sample_forecasts_covfair.png")
    fig.savefig(out, dpi=150)
    print("saved:", out)


if __name__ == "__main__":
    main()
