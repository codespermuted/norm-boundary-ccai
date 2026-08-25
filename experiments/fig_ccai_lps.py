"""CCAI workshop figure: two panels, no in-plot statistics.

Left  — the eight aggregated datasets: RevIN-CondNorm gap vs pre-registered
        LPS (classifier view; the 0.28-0.58 interior is shaded as unobserved).
Right — the ten GEFCom-Wind zones (graded view; all above tau).

Values are computed from the frozen CSVs (g4_grid dedup'd, zonegaps), not
hardcoded. Output: paper/figures/fig_ccai_lps.{pdf,png}.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
FIGS = os.path.join(ROOT, "paper", "figures")
TAU = 0.3

LABEL = {"jeju_wind": "Jeju", "gefcom_wind": "GEF-Wind",
         "gefcom_load": "GEF-Load", "gefcom_solar": "GEF-Solar",
         "etth1": "ETTh1", "etth2": "ETTh2",
         "electricity": "Electricity", "weather": "Weather"}
EXO = {"jeju_wind", "gefcom_wind", "gefcom_load", "gefcom_solar"}


MAIN_BACKBONES = ["rlinear", "patchtst", "segrnn", "lgbm_dms"]
# Table 1 anchor cells (revin, condnorm) — the figure must reproduce them.
TABLE1 = {"jeju_wind": (0.6969, 0.2933), "gefcom_wind": (1.0039, 0.1629),
          "gefcom_load": (0.3432, 0.1066), "gefcom_solar": (0.1844, 0.0580),
          "etth1": (0.3771, 2.2016), "etth2": (0.2900, 6.6044),
          "electricity": (0.1458, 0.1726), "weather": (0.1719, 0.7274)}


def main():
    g = pd.read_csv(os.path.join(ROOT, "results", "g4_grid.csv")).drop_duplicates()
    g = g[g.backbone.isin(MAIN_BACKBONES)]
    piv = g.groupby(["dataset", "norm"])["mse"].mean().unstack("norm")
    for name, (rv, cn) in TABLE1.items():
        assert abs(piv.loc[name, "revin"] - rv) < 5e-4, (name, piv.loc[name, "revin"])
        assert abs(piv.loc[name, "condnorm"] - cn) < 5e-4, (name, piv.loc[name, "condnorm"])
    gap = (piv["revin"] - piv["condnorm"]).rename("gap")
    lps = pd.read_csv(os.path.join(ROOT, "results", "lps_official.csv"))
    lps = lps.set_index(lps.columns[0]).iloc[:, 0].rename("lps")
    ds = pd.concat([lps, gap], axis=1).dropna()

    z = pd.read_csv(os.path.join(ROOT, "results", "graded_lps_zonegaps.csv"))

    plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.4, 1.85))

    # the 0.28-0.58 interior band is explained in the caption ("shaded =
    # unobserved interior"); no in-plot rotated text — it read as a glitch
    ax.axvspan(0.283, 0.575, color="0.92", zorder=0)
    ax.axvline(TAU, ls="--", lw=0.9, color="0.3")
    ax.axhline(0, lw=0.7, color="0.6")
    ax.set_xlim(-0.9, 1.35)
    # (dx pt, dy pt, ha): exogenous labels sit right/below their dots (the
    # cluster is in the top-right), standard-group labels sit left — chosen
    # so no label crosses the frame, the tau line, another label, or a dot
    off = {"etth1": (5, 3, "left"), "etth2": (-5, 0, "right"),
           "electricity": (-5, -3, "right"), "weather": (-5, -3, "right"),
           "jeju_wind": (-5, -2, "right"), "gefcom_wind": (5, -2, "left"),
           "gefcom_load": (5, -2, "left"), "gefcom_solar": (0, -11, "center")}
    for name, r in ds.iterrows():
        exo = name in EXO
        ax.scatter(r["lps"], r["gap"], s=26, zorder=3,
                   color="#1f77b4" if exo else "#d62728",
                   marker="o" if exo else "s")
        dx, dy, ha = off.get(name, (4, 4, "left"))
        ax.annotate(LABEL.get(name, name), (r["lps"], r["gap"]),
                    textcoords="offset points", xytext=(dx, dy), ha=ha,
                    fontsize=7)
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_yticks([-10, -1, -0.1, 0, 0.1, 1])
    ax.set_yticklabels(["-10", "-1", "-0.1", "0", "0.1", "1"])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("LPS (pre-registered, computed before training)")
    # short label: the full "MSE gap: RevIN - CondNorm" overflows the
    # 1.48in panel height and gets clipped in the saved PDF
    ax.set_ylabel(r"RevIN $-$ CN (MSE)")
    ax.set_title("Eight datasets: sign predicted 8/8", fontsize=8.5)
    ax.text(TAU - 0.04, 0.90, r"$\tau=0.3$", ha="right", fontsize=7,
            transform=ax.get_xaxis_transform())

    bx.scatter(z["lps"], z["revin_cn"], s=26, color="#1f77b4", zorder=3)
    # near-coincident pairs (z8/z9 at lps~0.63, z1/z3 at ~0.70) and the
    # right-edge cluster (z5/z7/z4) get labels on opposite sides
    zoff = {2: (4, 2, "left"), 9: (-4, 0, "right"), 8: (-4, -9, "right"),
            10: (4, -3, "left"), 1: (4, 1, "left"), 3: (5, -4, "left"),
            6: (4, -3, "left"), 5: (-4, -2, "right"), 7: (-4, -1, "right"),
            4: (4, -3, "left")}
    for _, r in z.iterrows():
        dx, dy, ha = zoff.get(int(r["zone"]), (3, 3, "left"))
        bx.annotate(f"z{int(r['zone'])}", (r["lps"], r["revin_cn"]),
                    textcoords="offset points", xytext=(dx, dy), ha=ha,
                    fontsize=7)
    bx.axhline(0, lw=0.7, color="0.6")
    bx.set_xlabel("per-zone LPS (pre-registered)")
    bx.set_ylabel(r"RevIN $-$ CN (MSE)")
    bx.set_ylim(-0.05, 1.1)
    bx.set_xlim(0.563, 0.785)
    bx.set_title("Ten wind zones: positive in 10/10", fontsize=8.5)

    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"fig_ccai_lps.{ext}"), dpi=200,
                    bbox_inches="tight")
    print("panel A:", ds.round(3).to_dict("index"))
    print("panel B zones:", len(z))


if __name__ == "__main__":
    main()
