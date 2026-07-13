"""LGBM-DMS sanity on ETTh1 h=96 (CI pooled windows, window z-norm arm).

Literature context: GBM/regression baselines on ETTh1-96 multivariate sit
around 0.40-0.48 std-scale MSE (TFB, position papers). Bound: <= 0.50.

Usage: uv run python -m experiments.g3_lgbm_sanity
"""

import numpy as np

from src.data import build_dataset
from src.models.lgbm_dms import LgbmDMS, window_znorm

L, H = 336, 96


def pooled(ds, split):
    w = ds.windows(split)
    seg = getattr(ds.splits, split)  # (T, C)
    n = len(w)
    C = seg.shape[1]
    xs = np.stack([seg[i : i + L] for i in range(n)])          # (n, L, C)
    ys = np.stack([seg[i + L : i + L + H] for i in range(n)])  # (n, H, C)
    x = xs.transpose(0, 2, 1).reshape(n * C, L)
    y = ys.transpose(0, 2, 1).reshape(n * C, H)
    return x, y


def main():
    ds = build_dataset("etth1", lookback=L, horizon=H)
    xtr, ytr = pooled(ds, "train")
    xte, yte = pooled(ds, "test")

    xtr_n, (mtr, str_) = window_znorm(xtr)
    xte_n, (mte, ste) = window_znorm(xte)
    model = LgbmDMS(horizon=H, n_estimators=120).fit(xtr_n, (ytr - mtr) / str_)
    pred = model.predict(xte_n) * ste + mte
    mse = float(np.mean((pred - yte) ** 2))
    mae = float(np.mean(np.abs(pred - yte)))
    print(f"SANITY lgbm_dms(win-znorm): mse={mse:.4f} mae={mae:.4f} "
          f"(bound 0.50) {'OK' if mse <= 0.50 else 'OUT-OF-RANGE'}")


if __name__ == "__main__":
    main()
