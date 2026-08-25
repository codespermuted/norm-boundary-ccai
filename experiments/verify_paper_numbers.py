"""Re-derive every number the CCAI body prints, from the frozen CSVs.

Countermeasure for the failure mode three review rounds each surfaced: prose
edited under a hard page limit re-scopes a sentence without re-checking the base
its number came from. Run this after any edit that touches a number.

Usage: uv run python -m experiments.verify_paper_numbers
"""
import os
import re

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
R = lambda n: pd.read_csv(os.path.join(ROOT, "results", n))
B = 100_000


def boot(v, seed=0):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    rng = np.random.default_rng(seed)
    return float(v[rng.integers(0, len(v), size=(B, len(v)))].mean(1).mean())


def boot_ci(v, seed=0):
    """Percentile bootstrap bounds, same recipe as the paper's intervals."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    rng = np.random.default_rng(seed)
    m = v[rng.integers(0, len(v), size=(B, len(v)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    checks = []

    def chk(name, got, want, tol=0.0015):
        checks.append((name, abs(got - want) <= tol, got, want))

    f = R("g11_footing.csv")
    cell = lambda bb: (f[f.backbone == bb]
                       .groupby(["dataset", "h", "arm", "footing"]).mse.mean()
                       .unstack(["arm", "footing"]))
    c, m = cell("linmix"), cell("mlpmix")
    par = R("g4_covfair_full.csv")
    parl = par[par.backbone == "linmix"]
    cn = parl[parl.arm == "condnorm"].groupby(["dataset", "h"]).mse.mean()

    for ft, want in [("global", 0.026), ("window_floor", -0.075), ("scale", -0.030)]:
        chk(f"toggle @ {ft}", boot(c[("revin", ft)] - c[("raw", ft)]), want)
    chk("RevIN footing sensitivity, linmix",
        boot(c[("revin", "global")] - c[("revin", "scale")]), 0.032)
    chk("RevIN footing sensitivity, mlpmix",
        boot(m[("revin", "global")] - m[("revin", "scale")]), 0.024)
    chk("interaction",
        boot((c[("revin", "global")] - c[("raw", "global")])
             - (c[("revin", "scale")] - c[("raw", "scale")])), 0.056)
    chk("RAW share of the interaction",
        boot(c[("raw", "global")] - c[("raw", "scale")]), -0.024)
    chk("restricted min endpoint",
        boot(np.minimum(c[("revin", "global")], c[("revin", "scale")])
             - c[("raw", "global")]), -0.006)
    dr = c[("raw", "center_global")] - c[("raw", "global")]
    dv = c[("revin", "center_scale")] - c[("revin", "scale")]
    chk("level-isolating difference", boot((dr - dv).dropna()), 0.014)

    rev = [x for x in c.columns if x[0] == "revin"]
    gaps = [min(c[("raw", "global")][k], c.loc[k, rev].min()) - cn.get(k, np.nan)
            for k in c.index]
    chk("boundary gap, min", min(gaps), 0.051)
    chk("boundary gap, max", max(gaps), 0.147)

    # Boundary scope (2026-07-31): per-backbone Raw/RevIN-to-CN gaps that the
    # rescoped boundary claim prints, over the 11 exogenous parity cells.
    pc = par.groupby(["backbone", "dataset", "h", "arm"]).mse.mean().unstack("arm")
    pc = pc[pc.index.get_level_values("dataset").str.startswith(("jeju", "gefcom"))]
    for bb, wraw, wrev in [("linmix", 0.127, 0.147), ("mlpmix", 0.052, 0.076),
                           ("patchtstcov", 0.142, 0.145), ("segrnncov", -0.004, 0.007),
                           ("lgbmcov", -0.009, 0.010)]:
        b = pc.xs(bb, level="backbone")
        chk(f"raw-to-CN gap, {bb}", (b.raw - b.condnorm).mean(), wraw)
        chk(f"revin-to-CN gap, {bb}", (b.revin - b.condnorm).mean(), wrev)
    six = pc.xs("lgbmcov", level="backbone")
    six = six[six.index.get_level_values("dataset").isin(["gefcom_wind", "gefcom_solar"])]
    chk("lgbmcov raw<=CN on footing-grid cells (of 6)",
        float(((six.raw - six.condnorm) <= 0).sum()), 4, 0.1)
    mm = cell("mlpmix")
    mcn = (par[par.backbone == "mlpmix"][lambda d: d.arm == "condnorm"]
           .groupby(["dataset", "h"]).mse.mean())
    mrev = [x for x in mm.columns if x[0] == "revin"]
    mgaps = [min(mm[("raw", "global")][k], mm.loc[k, mrev].min()) - mcn.get(k, np.nan)
             for k in mm.index]
    chk("mlpmix boundary margin, min", min(mgaps), 0.025)
    chk("mlpmix boundary margin, max", max(mgaps), 0.128)

    g = R("g13_stats.csv")
    base = parl.groupby(["dataset", "h", "arm"]).mse.mean().unstack("arm")
    st = g.groupby(["dataset", "h", "arm"]).mse.mean().unstack("arm")
    j = base.join(st, rsuffix="_st", how="inner")
    chk("gap closed by feeding back ybar and s",
        float(((j.revin - j.revin_st) / (j.revin - j.condnorm)).mean()), 0.065, 0.004)

    w = R("g15_width.csv")
    w = w[w.variant == "pooled"].groupby(["dataset", "arm"])[["width80", "cov80"]].mean()
    chk("GEFCom-Wind RevIN width", w.loc[("gefcom_wind", "revin"), "width80"], 2.22, 0.006)
    chk("GEFCom-Wind CN width", w.loc[("gefcom_wind", "condnorm"), "width80"], 1.12, 0.006)
    chk("GEFCom-Load CN coverage", w.loc[("gefcom_load", "condnorm"), "cov80"], 0.79, 0.003)

    r = R("g12_ramp_cells.csv")
    r = r[(r.backbone == "linmix") & r.clean]
    chk("ramp bottom tercile", r.cost_lo.mean(), 0.0139)
    chk("ramp top tercile", r.cost_hi.mean(), 0.0335)
    chk("ramp top/bottom ratio (abstract's 2.4x)",
        r.cost_hi.mean() / r.cost_lo.mean(), 2.4, 0.05)

    gr = R("g4_grid.csv")
    for bb, want, wcn in [("timexer_ms", 0.011, 0.41), ("itransformer_ms", 0.019, 0.43)]:
        q = gr[gr.backbone == bb].groupby(["dataset", "h", "norm"]).mse.mean().unstack("norm")
        exo = [i for i in q.index if i[0].startswith(("jeju", "gefcom"))]
        chk(f"{bb} isolated toggle",
            float((q.loc[exo, "revin"] - q.loc[exo, "raw"]).dropna().mean()), want)
        chk(f"{bb} revin-to-CN gap (Fig 1c)",
            float((q.loc[exo, "revin"] - q.loc[exo, "condnorm"]).dropna().mean()),
            wcn, 0.005)

    # ---- 2026-08-25 review revision -------------------------------------
    # Priority 4: the body may never print the pooled +0.016 without the
    # bit-reproducible linear-mixer value beside it, so both are checked here.
    lm = pc.xs("linmix", level="backbone")
    d = (lm.revin - lm.raw).dropna()
    chk("linmix isolated toggle, 11 exogenous cells", float(d.mean()), 0.020)
    lo, hi = boot_ci(d)
    chk("linmix toggle CI lower", lo, 0.006, 0.002)
    chk("linmix toggle CI upper", hi, 0.037, 0.002)
    chk("pooled isolated toggle, 5 backbones x 11 cells",
        float((pc.revin - pc.raw).dropna().mean()), 0.016)

    # Priority 3: S1 now prints the SOTA sign split, so the counts are checked.
    for bb, want in [("timexer_ms", 5.0), ("itransformer_ms", 5.0)]:
        q = gr[gr.backbone == bb].groupby(["dataset", "h", "norm"]).mse.mean().unstack("norm")
        exo = [i for i in q.index if i[0].startswith(("jeju", "gefcom"))]
        t = (q.loc[exo, "revin"] - q.loc[exo, "raw"]).dropna()
        chk(f"{bb} exogenous cells", float(len(t)), 11.0, 0.1)
        chk(f"{bb} cells with a positive toggle", float((t > 0).sum()), want, 0.1)

    # Priority 2 / 6: the LPS values S2 now names, and the tau-to-Electricity
    # distance the body prints as 0.017.
    lp = R("lps_official.csv").set_index("dataset").lps
    chk("Electricity LPS", float(lp["electricity"]), 0.283, 0.0006)
    chk("GEFCom-Load LPS (panel max)", float(lp["gefcom_load"]), 0.894, 0.0006)
    chk("GEFCom-Load is the panel maximum", float(lp.max()), float(lp["gefcom_load"]), 1e-9)
    chk("tau minus Electricity LPS", 0.30 - float(lp["electricity"]), 0.017, 0.0006)
    zg = R("graded_lps_zonegaps.csv")
    chk("zones with a positive RevIN-CN gap", float((zg.revin_cn > 0).sum()), 10.0, 0.1)
    chk("zones below the tau interval's upper end (0.70)",
        float((zg.lps < 0.70).sum()), 5.0, 0.1)
    chk("lowest zone LPS", float(zg.lps.min()), 0.575, 0.0006)
    chk("gap between Electricity and the lowest zone is unoccupied",
        float(((lp > lp["electricity"]) & (lp < zg.lps.min())).sum()), 0.0, 0.1)
    chk("energy group LPS min", float(lp[["jeju_wind", "gefcom_wind", "gefcom_load",
                                          "gefcom_solar"]].min()), 0.744, 0.0006)
    chk("benchmark group LPS min", float(lp[["etth1", "etth2", "electricity",
                                             "weather"]].min()), -0.717, 0.0006)

    # ---- structural re-check (2026-08-25): counts and printed tables ------
    # The review checklist asks for Table 1's column means, the 4,202 run total
    # and the 11/7 cell counts to be re-derived, not trusted.
    MAIN_BB = ("rlinear", "patchtst", "segrnn", "lgbm_dms")
    SOTA_BB = ("timexer", "timexer_ms", "itransformer", "itransformer_ms")
    md = gr[gr.backbone.isin(MAIN_BB)].drop_duplicates(
        ["dataset", "norm", "backbone", "h", "seed"])
    n_main, n_sota, n_par = len(md), int(gr.backbone.isin(SOTA_BB).sum()), len(par)
    chk("main-grid runs (deduplicated)", float(n_main), 1794.0, 0.1)
    chk("SOTA replication runs", float(n_sota), 1275.0, 0.1)
    chk("information-parity runs", float(n_par), 1133.0, 0.1)
    chk("run total", float(n_main + n_sota + n_par), 4202.0, 0.1)
    chk("wall-clock-logged runs reported", float(n_main + n_sota), 3069.0, 0.1)
    chk("superseded main-grid re-run rows",
        float(len(gr[gr.backbone.isin(MAIN_BB)]) - n_main), 48.0, 0.1)
    zn = R("graded_lps.csv")
    chk("zone-study RLinear runs", float((zn.backbone == "rlinear").sum()), 450.0, 0.1)
    chk("footing-block runs (four footings)",
        float(len(f[f.footing.isin(["global", "window", "window_floor", "scale"])])),
        480.0, 0.1)
    chk("level-isolating runs",
        float(len(f[f.footing.isin(["center_global", "center_scale"])])), 120.0, 0.1)
    chk("mean-only runs", float(len(R("g9_meanonly.csv"))), 180.0, 0.1)
    chk("fed-back-statistics runs", float(len(g)), 70.0, 0.1)

    lm_cells = pc.xs("linmix", level="backbone").index
    clean = [c for c in lm_cells
             if c[0] != "gefcom_load" and not (c[0] == "jeju_wind" and c[1] == 48)]
    chk("exogenous parity cells", float(len(lm_cells)), 11.0, 0.1)
    chk("clean exogenous parity cells", float(len(clean)), 7.0, 0.1)

    # Table 2 (access versus layer), exactly as experiments/g4_table1.py builds it.
    t2 = md.groupby(["dataset", "norm"]).mse.mean().unstack("norm")
    PAPER_T2 = {
        "jeju_wind": (0.6887, 0.6969, 0.7030, 0.7724, 0.2933),
        "gefcom_wind": (1.1153, 1.0039, 0.9755, 1.1875, 0.1629),
        "gefcom_solar": (0.1828, 0.1844, 0.2116, 0.1833, 0.0580),
        "gefcom_load": (0.3592, 0.3432, 0.3330, 0.3678, 0.1066),
        "etth1": (0.3962, 0.3771, 0.3910, 0.4114, 2.2016),
        "etth2": (0.3868, 0.2900, 0.2869, 0.3253, 6.6044),
        "electricity": (0.1500, 0.1458, 0.1443, 0.1375, 0.1726),
        "weather": (0.1815, 0.1719, 0.1666, 0.1747, 0.7274)}
    n_bad_t2 = sum(abs(t2.loc[d, c] - v) > 6e-5
                   for d, vals in PAPER_T2.items()
                   for c, v in zip(("raw", "revin", "san", "fan", "condnorm"), vals))
    chk("Table 2 cells disagreeing with the frozen grid", float(n_bad_t2), 0.0, 0.1)

    # Table 1 column means, re-added from the emitted tables/tab_footing.tex.
    tex = open(os.path.join(ROOT, "paper", "tables", "tab_footing.tex")).read()
    body = [l for l in tex.splitlines() if l.strip().startswith(("GEFCom", "mean"))]
    num = re.compile(r"-?\d+\.\d+")
    vals = [[float(x) for x in num.findall(l.split("&", 1)[1])] for l in body]
    cells, printed = np.array(vals[:6]), np.array(vals[6])
    chk("Table 1 column means recomputed from its own rows",
        float(np.abs(cells.mean(0) - printed).max()), 0.0, 0.0006)

    # ---- deployment-variant LPS (run 2026-08-25) --------------------------
    if os.path.exists(os.path.join(ROOT, "results", "lps_deployment.csv")):
        dp = R("lps_deployment.csv").set_index("dataset")
        dz = R("lps_deployment_zones.csv")
        chk("deployment sides retained, datasets", float(dp.side_agrees.sum()), 8.0, 0.1)
        chk("deployment sides retained, zones", float(dz.side_agrees.sum()), 10.0, 0.1)
        chk("Electricity deployment LPS", float(dp.loc["electricity", "lps_deployment"]), 0.2997, 0.0006)
        chk("tau minus Electricity deployment LPS",
            0.30 - float(dp.loc["electricity", "lps_deployment"]), 0.0003, 0.0006)
        chk("Weather deployment LPS", float(dp.loc["weather", "lps_deployment"]), -1.347, 0.001)
        chk("ETTh1 deployment LPS", float(dp.loc["etth1", "lps_deployment"]), -0.186, 0.001)
        chk("ETTh2 deployment LPS", float(dp.loc["etth2", "lps_deployment"]), 0.188, 0.001)
        chk("GEFCom-Load deployment LPS", float(dp.loc["gefcom_load", "lps_deployment"]), 0.883, 0.001)
        chk("GEFCom-Solar deployment LPS", float(dp.loc["gefcom_solar", "lps_deployment"]), 0.849, 0.001)
        for ds in ("jeju_wind", "gefcom_wind"):
            chk(f"{ds} variants coincide", float(dp.loc[ds, "lps_full"] == dp.loc[ds, "lps_deployment"]), 1.0, 0.1)
        chk("zone deployment LPS min", float(dz.lps_deployment.min()), 0.471, 0.001)
        chk("zone deployment LPS max", float(dz.lps_deployment.max()), 0.750, 0.001)
        chk("largest zone move (z09)", float(dz.delta.min()), -0.161, 0.001)

    for name, good, got, want in checks:
        print(f"{'OK ' if good else 'XX '}{name:44s} paper {want:+.4f}  data {got:+.4f}")
    n_ok = sum(g for _, g, _, _ in checks)
    print(f"\n{n_ok}/{len(checks)} body numbers reproduce from the frozen CSVs")
    return 0 if n_ok == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
