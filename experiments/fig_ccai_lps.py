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

    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8.5,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.4, 1.48))

    ax.axvspan(0.283, 0.575, color="0.92", zorder=0)
    ax.text(0.429, 0.5, "unobserved", ha="center", va="center", fontsize=7,
            color="0.45", transform=ax.get_xaxis_transform(), rotation=90)
    ax.axvline(TAU, ls="--", lw=0.9, color="0.3")
    ax.axhline(0, lw=0.7, color="0.6")
    ax.set_xlim(-0.9, 1.13)
    off = {"etth1": (4, 4), "etth2": (4, 4), "electricity": (4, 2),
           "weather": (-16, -12), "jeju_wind": (0, -13), "gefcom_wind": (4, 2),
           "gefcom_load": (4, 2), "gefcom_solar": (4, -10)}
    for name, r in ds.iterrows():
        exo = name in EXO
        ax.scatter(r["lps"], r["gap"], s=26, zorder=3,
                   color="#1f77b4" if exo else "#d62728",
                   marker="o" if exo else "s")
        ax.annotate(LABEL.get(name, name), (r["lps"], r["gap"]),
                    textcoords="offset points", xytext=off.get(name, (4, 4)),
                    fontsize=7)
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_yticks([-10, -1, -0.1, 0, 0.1, 1])
    ax.set_yticklabels(["-10", "-1", "-0.1", "0", "0.1", "1"])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("LPS (pre-registered, computed before training)")
    ax.set_ylabel(r"MSE gap: RevIN $-$ CondNorm")
    ax.set_title("Eight datasets: sign predicted 8/8", fontsize=8.5)
    ax.text(TAU - 0.04, 0.90, r"$\tau=0.3$", ha="right", fontsize=7,
            transform=ax.get_xaxis_transform())

    bx.scatter(z["lps"], z["revin_cn"], s=26, color="#1f77b4", zorder=3)
    zoff = {8: (3, -11), 9: (3, 4)}
    for _, r in z.iterrows():
        bx.annotate(f"z{int(r['zone'])}", (r["lps"], r["revin_cn"]),
                    textcoords="offset points",
                    xytext=zoff.get(int(r["zone"]), (3, 3)), fontsize=7)
    bx.axhline(0, lw=0.7, color="0.6")
    bx.set_xlabel("per-zone LPS (pre-registered)")
    bx.set_ylabel(r"MSE gap: RevIN $-$ CondNorm")
    bx.set_ylim(-0.05, 1.1)
    bx.set_title("Ten wind zones: positive in 10/10", fontsize=8.5)

    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"fig_ccai_lps.{ext}"), dpi=200,
                    bbox_inches="tight")
    print("panel A:", ds.round(3).to_dict("index"))
    print("panel B zones:", len(z))


if __name__ == "__main__":
    main()
