"""Model Confidence Set (Hansen, Lunde & Nason 2011), T_max statistic,
stationary-bootstrap p-values. Returns the alpha-level surviving set and
elimination p-values (plan §5.4, alpha = 0.10).
"""

from __future__ import annotations

import numpy as np


def _stationary_bootstrap_indices(n: int, B: int, mean_block: float,
                                  rng: np.random.Generator) -> np.ndarray:
    p = 1.0 / mean_block
    starts = rng.integers(0, n, size=(B, n))
    restart = rng.random((B, n)) < p
    restart[:, 0] = True
    idx = np.zeros((B, n), dtype=int)
    for t in range(n):
        if t == 0:
            idx[:, 0] = starts[:, 0]
        else:
            cont = (idx[:, t - 1] + 1) % n
            idx[:, t] = np.where(restart[:, t], starts[:, t], cont)
    return idx


def mcs(losses: np.ndarray, names: list[str], alpha: float = 0.10,
        B: int = 1000, seed: int = 0) -> dict:
    """losses: (n_origins, m_models). Returns surviving set + p-values."""
    rng = np.random.default_rng(seed)
    n, m = losses.shape
    idx = _stationary_bootstrap_indices(n, B, mean_block=max(2.0, n ** (1 / 3)), rng=rng)

    included = list(range(m))
    pvals: dict[str, float] = {}
    p_running = 0.0
    while len(included) > 1:
        L = losses[:, included]
        dbar = L.mean(axis=0)[:, None] - L.mean(axis=0)[None, :]  # (k,k)
        boot_means = L[idx].mean(axis=1)                          # (B, k)
        bd = boot_means[:, :, None] - boot_means[:, None, :]      # (B, k, k)
        var = np.mean((bd - dbar[None]) ** 2, axis=0) + 1e-300
        tstat = dbar / np.sqrt(var)
        TR = np.max(tstat)
        TR_boot = np.max((bd - dbar[None]) / np.sqrt(var[None]), axis=(1, 2))
        p = float(np.mean(TR_boot >= TR))
        p_running = max(p_running, p)
        worst = int(np.argmax(tstat.max(axis=1)))
        pvals[names[included[worst]]] = p_running
        if p >= alpha:
            break
        included.pop(worst)
    for i in included:
        pvals.setdefault(names[i], 1.0)
    return {"included": [names[i] for i in included], "pvalues": pvals}
