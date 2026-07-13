"""Fig 1 — dominance regions of RAW / IN / CN on the (lambda, drift) plane.

Spec: docs/theory_g1.md §6. Three panels (h = 24, 96, 336) share axes; the IN
region retreats as the horizon grows (Proposition 3 made visible).

Usage: uv run python -m src.theory.fig1
"""

import math
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from src.theory.closed_form import TheoryParams, dominance_grid, window_noise_var
from src.theory.figstyle import (
    INK,
    METHOD_COLORS,
    METHOD_LABELS,
    apply_paper_style,
    tint,
)

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "paper", "figures")

BASE = TheoryParams(V=1.0, w=96, sigma_z=1.0, sigma_u=math.sqrt(0.0036),
                    sigma_est=math.sqrt(0.02))
HORIZONS = (24, 96, 336)
# label order matches closed_form.METHODS: 0=raw, 1=in, 2=cn_est
FILLS = ListedColormap([tint(METHOD_COLORS["raw"]), tint(METHOD_COLORS["in"]),
                        tint(METHOD_COLORS["cn"])])


def region_label_positions(labels, lam_grid, drift2_grid):
    """Centroid of each region present, for direct labels."""
    out = {}
    for code in np.unique(labels):
        ys, xs = np.nonzero(labels == code)
        out[int(code)] = (float(lam_grid[xs].mean()), float(drift2_grid[ys].mean()))
    return out


def make_fig1(path_base: str | None = None) -> str:
    apply_paper_style()
    lam_grid = np.linspace(0, 1, 201)
    drift2_grid = np.linspace(0, 1.5, 151)

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), sharey=True,
                             constrained_layout=True)
    for ax, h in zip(axes, HORIZONS):
        p = TheoryParams(**{**BASE.__dict__, "h": h})
        labels = dominance_grid(lam_grid, drift2_grid, p)
        ax.pcolormesh(lam_grid, drift2_grid, labels, cmap=FILLS, vmin=0, vmax=2,
                      shading="nearest", rasterized=True)

        # analytic IN/CN boundary: sigma_delta^2 = drift+noise - est - (1-lam)V
        pen = h * p.sigma_u**2 + p.s_x2h + window_noise_var(p)
        boundary = pen - p.sigma_est**2 - (1 - lam_grid) * p.V
        mask = (boundary >= 0) & (boundary <= drift2_grid[-1])
        if mask.any():
            ax.plot(lam_grid[mask], boundary[mask], color=INK, lw=1.2)

        names = {0: "RAW", 1: "IN", 2: "CN"}
        for code, (cx, cy) in region_label_positions(
            labels, lam_grid, drift2_grid
        ).items():
            frac = np.mean(labels == code)
            if frac > 0.02:
                ax.text(cx, cy, names[code], ha="center", va="center",
                        fontsize=9, fontweight="bold", color=INK)
        ax.set_title(f"$h = {h}$")
        ax.set_xlabel(r"$\lambda$ (level predictability)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, drift2_grid[-1])
    axes[0].set_ylabel(r"$\sigma_\Delta^2 / V$ (covariate-orthogonal drift)")

    handles = [Patch(facecolor=tint(METHOD_COLORS[k]), edgecolor=INK, lw=0.4,
                     label=METHOD_LABELS[k]) for k in ("raw", "in", "cn")]
    fig.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False)

    os.makedirs(FIG_DIR, exist_ok=True)
    base = path_base or os.path.join(FIG_DIR, "fig1_dominance")
    fig.savefig(base + ".pdf")
    fig.savefig(base + ".png")
    plt.close(fig)
    return base


if __name__ == "__main__":
    out = make_fig1()
    print(f"saved: {out}.pdf / .png")
