import numpy as np

from src.eval.dm_test import dm_test
from src.eval.mcs import mcs


def test_dm_detects_clear_difference():
    rng = np.random.default_rng(0)
    e2 = rng.chisquare(1, 500)
    e1 = e2 + 0.5 + rng.normal(0, 0.1, 500)  # model 1 clearly worse
    r = dm_test(e1, e2, h=24)
    assert r["stat"] > 0 and r["p"] < 0.01


def test_dm_no_difference():
    rng = np.random.default_rng(1)
    base = rng.chisquare(1, 500)
    r = dm_test(base + rng.normal(0, 0.01, 500),
                base + rng.normal(0, 0.01, 500), h=24)
    assert r["p"] > 0.05


def test_mcs_keeps_good_drops_bad():
    """MCS coverage: the truly-bad arm must be eliminated with tiny p; a tied
    twin survives with prob >= 1-alpha, so check across seeds (>=4/5)."""
    kept_twin = 0
    for seed in range(5):
        rng = np.random.default_rng(seed)
        n = 400
        common = rng.chisquare(1, n)
        losses = np.stack([
            common + rng.normal(0, 0.05, n),
            common + rng.normal(0, 0.05, n),
            common + 1.0 + rng.normal(0, 0.05, n),
        ], axis=1)
        res = mcs(losses, ["A", "B", "bad"], alpha=0.10, B=300, seed=seed)
        assert "bad" not in res["included"]
        assert res["pvalues"]["bad"] < 0.05
        kept_twin += int({"A", "B"} <= set(res["included"]))
    assert kept_twin >= 4, f"tied twin kept only {kept_twin}/5"
