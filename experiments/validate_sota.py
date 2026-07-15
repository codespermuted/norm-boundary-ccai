"""Paper-number reproduction for the ported SOTA backbones (G3-style AC).

ETTh1, L=96, h=96, standard multivariate protocol, built-in instance norm ON
(faithful to the official pipelines):
  iTransformer (ICLR 2024): paper MSE 0.386  -> accept 0.36..0.42
  TimeXer 'M'  (NeurIPS 24): paper MSE 0.382 -> accept 0.36..0.42

Usage: uv run python -m experiments.validate_sota
"""

import numpy as np
import torch

from src.data import build_dataset
from src.models.itransformer import ITransformer
from src.models.timexer import TimeXer

L, H = 96, 96
DEVICE = "cpu"  # keep GPUs untouched for the running grid


def windows(seg, L_, h):
    n = len(seg) - L_ - h + 1
    x = np.stack([seg[i : i + L_] for i in range(n)])
    y = np.stack([seg[i + L_ : i + L_ + h] for i in range(n)])
    return (torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32))


def train_eval(model, forward, ds, lr=1e-4, epochs=10, patience=3, batch=32):
    tr_x, tr_y = windows(ds.splits.train, L, H)
    va_x, va_y = windows(ds.splits.val, L, H)
    te_x, te_y = windows(ds.splits.test, L, H)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(tr_x))
        for i in range(0, len(perm), batch):
            j = perm[i : i + batch]
            opt.zero_grad()
            torch.nn.functional.mse_loss(forward(tr_x[j]), tr_y[j]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = np.mean([torch.mean((forward(va_x[i : i + 256])
                                     - va_y[i : i + 256]) ** 2).item()
                         for i in range(0, len(va_x), 256)])
        print(f"  epoch {ep}: val={v:.4f}", flush=True)
        if v < best:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        mse = np.mean([torch.mean((forward(te_x[i : i + 256])
                                   - te_y[i : i + 256]) ** 2).item()
                       for i in range(0, len(te_x), 256)])
    return float(mse)


def main():
    torch.manual_seed(0)
    torch.set_num_threads(12)
    ds = build_dataset("etth1", lookback=L, horizon=H)

    it = ITransformer(L, H, ds.num_features, d_model=256, n_heads=8,
                      e_layers=2, d_ff=256, dropout=0.1, use_norm=True)
    mse_it = train_eval(it, lambda x: it(x), ds)
    ok_it = 0.36 <= mse_it <= 0.42
    print(f"iTransformer ETTh1-96: MSE={mse_it:.4f} (paper 0.386) "
          f"{'OK' if ok_it else 'OUT-OF-RANGE'}", flush=True)

    torch.manual_seed(0)
    tx = TimeXer(L, H, patch_len=16, d_model=256, n_heads=8, e_layers=1,
                 d_ff=1024, dropout=0.1, use_norm=True)
    mse_tx = train_eval(tx, lambda x: tx.forward_multi(x), ds, batch=32)
    ok_tx = 0.36 <= mse_tx <= 0.42
    print(f"TimeXer(M) ETTh1-96: MSE={mse_tx:.4f} (paper 0.382) "
          f"{'OK' if ok_tx else 'OUT-OF-RANGE'}", flush=True)


if __name__ == "__main__":
    main()
