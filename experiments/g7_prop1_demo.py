"""G7 item 5 — Proposition 1' nonlinear demo: does the RevIN constraint cost
what the (provisional) bound says, even for an MLP?

DGP (docs/theory_g1.md §3 Prop 1, M1 mapping; constants in configs/g7_prop1.yaml):
    g ~ N(0, lam*V), v ~ N(0, (1-lam)*V), window_j = g + v + e_j (e_j iid N(0, sigma_z^2))
    ybar = mean(window)  ->  Cov(ybar, g) = lam*V
    y = a*ybar + b*g + kappa*ybar*g + eps

Arms (identical MLP, identical inputs up to the normalization constraint):
    raw   : net([window ; g])                                  (full class, contains Bayes)
    revin : net([window - mean(window) ; g]) + mean(window)    (RevIN affine restoration --
            the centered window is independent of ybar, so the interaction
            kappa*ybar*g is unreachable; irreducible excess
            = ((a-1)^2 + kappa^2*lam*V) * Var(ybar|g), >= the bound below)
    cn    : net([window - g ; g]) + g                          (m_hat = g restoration;
            mean(window - g) recovers ybar - g, so the class contains Bayes)

Excess risk = mean((pred - Bayes)^2) on a large test set, Bayes predictor known
analytically from the DGP. PROVISIONAL bound (constant config-swappable):
    bound = bound_scale * kappa^2 * Var(ybar|g) * Var(g),
    Var(ybar|g) = (1-lam)*V + sigma_z^2/w  (DGP constants, not estimated).

CONTRADICTION PROTOCOL: if the RevIN arm's mean excess falls below the bound by
more than 2 SE at any lambda, no confirmation figure is written; the numbers go
to results/g7_prop1_CONTRADICTION.md and the script exits non-zero.

Output: results/g7_prop1.csv + paper/figures/figG_prop1_mlp.{pdf,png}
Usage:  uv run python -m experiments.g7_prop1_demo [--config configs/g7_prop1.yaml]
"""

import argparse
import csv
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import yaml

ROOT = os.path.join(os.path.dirname(__file__), "..")
CSV_PATH = os.path.join(ROOT, "results", "g7_prop1.csv")
FIG_BASE = os.path.join(ROOT, "paper", "figures", "figG_prop1_mlp")
CONTRA_PATH = os.path.join(ROOT, "results", "g7_prop1_CONTRADICTION.md")

ARMS = ("raw", "revin", "cn")
FIELDS = ["lam", "arm", "seed", "excess", "bound"]


def draw(dgp, lam, n, rng):
    """One synthetic sample: window (n, w), covariate g (n,), target y (n,)."""
    V, w = dgp["V"], dgp["w"]
    g = rng.normal(0.0, np.sqrt(lam * V), n)
    v = rng.normal(0.0, np.sqrt((1 - lam) * V), n)
    window = g[:, None] + v[:, None] + rng.normal(0.0, dgp["sigma_z"], (n, w))
    ybar = window.mean(axis=1)
    f_bayes = dgp["a"] * ybar + dgp["b"] * g + dgp["kappa"] * ybar * g
    y = f_bayes + rng.normal(0.0, dgp["noise_sd"], n)
    return window, g, y, f_bayes


def build_mlp(d_in, hidden):
    layers, d = [], d_in
    for d_h in hidden:
        layers += [torch.nn.Linear(d, d_h), torch.nn.ReLU()]
        d = d_h
    layers.append(torch.nn.Linear(d, 1))
    return torch.nn.Sequential(*layers)


def arm_features(arm, window, g):
    """Model input and additive restoration term for each arm."""
    if arm == "raw":
        return np.hstack([window, g[:, None]]), np.zeros_like(g)
    if arm == "revin":
        mu = window.mean(axis=1)
        return np.hstack([window - mu[:, None], g[:, None]]), mu
    if arm == "cn":
        return np.hstack([window - g[:, None], g[:, None]]), g
    raise KeyError(arm)


def run_arm(arm, train, test, cfg, seed, device):
    """Train the MLP for one arm, return excess risk over Bayes on the test set."""
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    m = cfg["model"]

    x_tr, base_tr = arm_features(arm, train[0], train[1])
    x_te, base_te = arm_features(arm, test[0], test[1])
    xt = torch.tensor(x_tr, dtype=torch.float32, device=device)
    bt = torch.tensor(base_tr, dtype=torch.float32, device=device)
    yt = torch.tensor(train[2], dtype=torch.float32, device=device)
    net = build_mlp(x_tr.shape[1], m["hidden"]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=m["lr"])
    batch = m["batch_size"]

    net.train()
    for _ in range(m["epochs"]):
        perm = torch.randperm(len(xt), device=device)
        for i in range(0, len(perm), batch):
            idx = perm[i : i + batch]
            opt.zero_grad()
            pred = net(xt[idx]).squeeze(-1) + bt[idx]
            torch.nn.functional.mse_loss(pred, yt[idx]).backward()
            opt.step()

    net.eval()
    xe = torch.tensor(x_te, dtype=torch.float32, device=device)
    be = torch.tensor(base_te, dtype=torch.float32, device=device)
    preds = []
    with torch.no_grad():
        for i in range(0, len(xe), 8192):
            preds.append((net(xe[i : i + 8192]).squeeze(-1)
                          + be[i : i + 8192]).cpu().numpy())
    pred = np.concatenate(preds)
    return float(np.mean((pred - test[3]) ** 2))


def provisional_bound(dgp, lam, bound_scale):
    """bound_scale * kappa^2 * Var(ybar|g) * Var(g), from DGP constants only."""
    var_ybar_g = (1 - lam) * dgp["V"] + dgp["sigma_z"] ** 2 / dgp["w"]
    var_g = lam * dgp["V"]
    return bound_scale * dgp["kappa"] ** 2 * var_ybar_g * var_g


def revin_irreducible(dgp, lam):
    """Exact irreducible excess of the RevIN class in this DGP (reference only):
    ((a-1)^2 + kappa^2 * Var(g)) * Var(ybar|g)."""
    var_ybar_g = (1 - lam) * dgp["V"] + dgp["sigma_z"] ** 2 / dgp["w"]
    return ((dgp["a"] - 1) ** 2 + dgp["kappa"] ** 2 * lam * dgp["V"]) * var_ybar_g


def make_figure(results, bounds, cfg):
    import matplotlib.pyplot as plt

    from src.theory.figstyle import (INK, INK_SECONDARY, METHOD_COLORS,
                                     apply_paper_style)

    apply_paper_style()
    arm_color = {"raw": METHOD_COLORS["raw"], "revin": METHOD_COLORS["in"],
                 "cn": METHOD_COLORS["cn"]}
    arm_label = {"raw": "RAW", "revin": "IN (RevIN)", "cn": "CN"}
    lambdas = cfg["lambdas"]

    fig, axes = plt.subplots(1, len(lambdas), figsize=(5.4, 2.6), sharey=True,
                             constrained_layout=True)
    rng = np.random.default_rng(0)  # jitter only
    for ax, lam in zip(np.atleast_1d(axes), lambdas):
        for j, arm in enumerate(ARMS):
            e = np.array(results[(lam, arm)])
            ax.scatter(j + rng.uniform(-0.12, 0.12, len(e)), e, s=10,
                       color=arm_color[arm], alpha=0.55, lw=0, zorder=3)
            ax.hlines(e.mean(), j - 0.22, j + 0.22, color=arm_color[arm],
                      lw=2.0, zorder=4)
        ax.axhline(bounds[lam], color=INK, ls="--", lw=1.0, zorder=2)
        ax.set_yscale("log")
        ax.set_xticks(range(len(ARMS)))
        ax.set_xticklabels([arm_label[a] for a in ARMS])
        ax.set_xlim(-0.5, len(ARMS) - 0.5)
        ax.set_title(rf"$\lambda = {lam}$")
    np.atleast_1d(axes)[0].set_ylabel("excess risk over Bayes")
    ax0 = np.atleast_1d(axes)[0]
    ax0.annotate(r"provisional bound $\kappa^2\,\mathrm{Var}(\bar y\mid g)\,\mathrm{Var}(g)$",
                 xy=(0.02, bounds[lambdas[0]]), xycoords=("axes fraction", "data"),
                 xytext=(0, -4), textcoords="offset points", va="top",
                 fontsize=7.5, color=INK)
    fig.suptitle("MLP arms: RevIN sits on/above the bound; RAW & CN fall below",
                 fontsize=8.5, color=INK_SECONDARY)
    os.makedirs(os.path.dirname(FIG_BASE), exist_ok=True)
    fig.savefig(FIG_BASE + ".pdf")
    fig.savefig(FIG_BASE + ".png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs",
                                                     "g7_prop1.yaml"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    args = ap.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    dgp = cfg["dgp"]
    torch.set_num_threads(10)

    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    results, bounds = {}, {}
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for li, lam in enumerate(cfg["lambdas"]):
            bounds[lam] = provisional_bound(dgp, lam, cfg["bound_scale"])
            print(f"lam={lam}: bound={bounds[lam]:.5f} "
                  f"(revin irreducible={revin_irreducible(dgp, lam):.5f})",
                  flush=True)
            for arm in ARMS:
                results[(lam, arm)] = []
            for seed in range(cfg["seeds"]):
                rng = np.random.default_rng([li, seed])
                train = draw(dgp, lam, dgp["n_train"], rng)
                test = draw(dgp, lam, dgp["n_test"], rng)
                for arm in ARMS:
                    ex = run_arm(arm, train, test, cfg, seed, args.device)
                    results[(lam, arm)].append(ex)
                    w.writerow({"lam": lam, "arm": arm, "seed": seed,
                                "excess": round(ex, 6),
                                "bound": round(bounds[lam], 6)})
                    f.flush()
                print(f"  seed {seed}: " + "  ".join(
                    f"{a}={results[(lam, a)][-1]:.5f}" for a in ARMS),
                    flush=True)

    # contradiction check: RevIN mean must not undercut the bound by > 2 SE
    contra = []
    for lam in cfg["lambdas"]:
        e = np.array(results[(lam, "revin")])
        mean, se = e.mean(), e.std(ddof=1) / np.sqrt(len(e))
        margin = mean - bounds[lam]
        print(f"lam={lam}: revin mean={mean:.5f} bound={bounds[lam]:.5f} "
              f"margin={margin:+.5f} (2SE={2 * se:.5f})", flush=True)
        if margin < -2 * se:
            contra.append((lam, mean, se, bounds[lam]))

    if contra:
        with open(CONTRA_PATH, "w") as fh:
            fh.write("# G7 Prop 1' demo — CONTRADICTION\n\n"
                     "RevIN arm mean excess risk fell below the provisional "
                     "bound by more than 2 SE.\nThe bound constant "
                     f"(bound_scale={cfg['bound_scale']}) is wrong or the "
                     "mapping is broken — do NOT use figG_prop1_mlp as "
                     "confirmation.\n\n"
                     "| lambda | revin mean excess | SE | bound | margin |\n"
                     "|---|---|---|---|---|\n")
            for lam, mean, se, b in contra:
                fh.write(f"| {lam} | {mean:.6f} | {se:.6f} | {b:.6f} "
                         f"| {mean - b:+.6f} |\n")
            fh.write(f"\nConfig: {args.config}; per-seed rows in "
                     f"results/g7_prop1.csv\n")
        print(f"CONTRADICTION -> {CONTRA_PATH}", flush=True)
        sys.exit(1)

    make_figure(results, bounds, cfg)
    print(f"saved: {FIG_BASE}.pdf / .png", flush=True)


if __name__ == "__main__":
    main()
