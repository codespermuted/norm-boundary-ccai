"""G7 Block E — LPS inference: permutation p, MBB CI, lambda*_hat.

POST-HOC REFINEMENT of the pre-registered LPS gate — same protocol as
compute_lps_official.py (w=96, LightGBM first stage, expanding CV, exact
covariate sets, per-channel mean for multivariate). Does NOT alter the
pre-registered tau rule; supporting inference only (results/gate1.md stands).

Per dataset:
  * permutation p (all circular window shifts, B<=999)
  * season-aligned permutation p (shifts restricted to multiples of the
    number of windows per week -> daily/weekly phase preserved under null)
  * 90% moving-block bootstrap CI (B=499)
  * lambda*_hat plug-in at h=96 (= w, the pre-registration window scale)

Outputs: results/lps_inference.csv + results/lps_inference.md.

Usage: uv run python -m experiments.g7_lps_inference
Env:   LPS_JOBS (default 12), LPS_PERM_B / LPS_BOOT_B (smoke overrides),
       LPS_INF_DATASETS (comma list to restrict, e.g. smoke on one dataset)
"""

import math
import os

import lightgbm as _lgb
import numpy as np
import pandas as pd

# Force single-threaded LightGBM: the window-mean datasets are tiny
# (hundreds of rows), where the wrapper's all-cores default thrashes on
# thread sync (minutes per fit instead of milliseconds). Patching the
# module attribute is picked up by src.theory.lps._fit_predict.
_OrigLGBM = _lgb.LGBMRegressor


class _SingleThreadLGBM(_OrigLGBM):
    def __init__(self, **kw):
        kw.setdefault("n_jobs", 1)
        super().__init__(**kw)


_lgb.LGBMRegressor = _SingleThreadLGBM

from joblib import Parallel, delayed

from src.data.covariate import longest_contiguous
from src.data.curation import BUILDERS
from src.data.etth import load_etth_frame
from src.data.ltsf import load_ltsf_frame
from src.theory.lps import calendar_features
from src.theory.lps_inference import lambda_star_hat, mbb_ci, permutation_test

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT_CSV = os.path.join(ROOT, "results", "lps_inference.csv")
OUT_MD = os.path.join(ROOT, "results", "lps_inference.md")
W = 96
H_STAR = 96  # lambda*_hat horizon anchor: h = w (pre-registration scale)
B_PERM = int(os.environ.get("LPS_PERM_B", "999"))
B_BOOT = int(os.environ.get("LPS_BOOT_B", "499"))
N_JOBS = int(os.environ.get("LPS_JOBS", "12"))

MULTIVARIATE = {
    "etth1": lambda: load_etth_frame("ETTh1"),
    "etth2": lambda: load_etth_frame("ETTh2"),
    "weather": lambda: load_ltsf_frame("weather"),
    "electricity": lambda: load_ltsf_frame("electricity"),
}
CURATED = ("jeju_wind", "gefcom_wind", "gefcom_load", "gefcom_solar")


def season_align(index, w):
    """Smallest window shift preserving weekly (hence daily) phase:
    lcm(window span, one week) in windows. Hourly w=96 -> 7; 10-min -> 21."""
    step = int(np.median(np.diff(index.values) / np.timedelta64(1, "s")))
    win = w * step
    week = 7 * 24 * 3600
    return math.lcm(win, week) // win


def dataset_arrays():
    """(name, y, X, index) in cheap-first order; covariates exactly as in
    compute_lps_official.py (exog+cal curated, cal-only multivariate)."""
    for name in CURATED:
        df = longest_contiguous(BUILDERS[name]())
        cal = calendar_features(df.index)
        cov = df.drop(columns=["y"]).values if df.shape[1] > 1 else None
        X = cal if cov is None else np.column_stack([cov, cal])
        yield name, df["y"].values, X, df.index
    for name, loader in MULTIVARIATE.items():
        df = loader()
        yield name, df.values, calendar_features(df.index), df.index


def _flush(rows):
    pd.DataFrame(rows).round(4).to_csv(OUT_CSV, index=False)


def _report(df):
    lines = [
        "# LPS inference (G7 Block E)",
        "",
        "**Post-hoc refinement — does not alter the pre-registered rule.** "
        "The G4 gate decision (results/gate1.md) was taken on the absolute "
        "LPS vs the pre-registered tau threshold; the permutation p-values, "
        "bootstrap CIs and lambda*_hat below are supporting inference "
        "computed afterwards under the identical LPS protocol "
        "(w=96, LightGBM, expanding CV, per-channel mean).",
        "",
        f"- Permutation: circular window shifts, B<={B_PERM} "
        "(exact enumeration when fewer shifts exist); aligned variant "
        "restricts shifts to whole weeks (daily/weekly phase kept under "
        "the null).",
        f"- CI: 90% moving-block bootstrap, B={B_BOOT}, "
        "block_len=ceil(n_windows^(1/3)).",
        f"- lambda*_hat: plug-in at h={H_STAR} with sigma_Delta^2=0 "
        "(lower bound; lam > lambda*_hat necessary, not sufficient, for "
        "CN — see src/theory/lps_inference.py docstring for proxy biases).",
        "",
        "```",
        df.to_string(index=False),
        "```",
        "",
    ]
    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines))


def main():
    only = os.environ.get("LPS_INF_DATASETS")
    only = set(only.split(",")) if only else None
    rows = []
    for name, y, X, index in dataset_arrays():
        if only is not None and name not in only:
            continue
        perm = permutation_test(y, X, W, B=B_PERM, seed=0, n_jobs=N_JOBS)
        align = season_align(index, W)
        if align < perm["n_windows"] - 1:
            perm_al = permutation_test(y, X, W, B=B_PERM,
                                       align_windows=align, seed=0,
                                       n_jobs=N_JOBS)
            p_al, b_al = perm_al["p_value"], perm_al["B_effective"]
        else:  # too few whole-week shifts for a meaningful null
            p_al, b_al = float("nan"), 0
        boot = mbb_ci(y, X, W, B=B_BOOT, seed=0, n_jobs=N_JOBS)
        Y = y if y.ndim == 2 else y[:, None]
        lam_ch = Parallel(n_jobs=N_JOBS)(
            delayed(lambda_star_hat)(Y[:, c], X, W, H_STAR)
            for c in range(Y.shape[1]))
        rows.append({
            "dataset": name,
            "lps": perm["lps"],
            "p_perm": perm["p_value"],
            "p_perm_aligned": p_al,
            "ci_lo": boot["ci_lo"],
            "ci_hi": boot["ci_hi"],
            "lambda_star_hat": float(np.mean(
                [r["lambda_star_hat"] for r in lam_ch])),
            "n_windows": perm["n_windows"],
            "b_perm": perm["B_effective"],
            "b_perm_aligned": b_al,
            "align_windows": align,
            "block_len": boot["block_len"],
        })
        print(rows[-1], flush=True)
        _flush(rows)
    df = pd.read_csv(OUT_CSV)
    _report(df)
    print(df.to_string(index=False))
    print(f"wrote {OUT_CSV} and {OUT_MD}")


if __name__ == "__main__":
    main()
