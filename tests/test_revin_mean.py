"""Invariants for the mean-only ablation arm (G9).

The point of this arm is that it differs from RAW in exactly one channel — the
lookback window mean — so these tests pin that it does that and nothing else.
"""
import torch

from src.norms import build_norm


def test_revin_mean_invertibility():
    """denorm(norm(x)) must reconstruct x (plan §5.2 invariant, all norms)."""
    torch.manual_seed(0)
    nm = build_norm("revin_mean", num_features=7)
    x = torch.randn(32, 336, 7) * 5 + 3
    assert torch.allclose(nm(nm(x, "norm"), "denorm"), x, atol=1e-6)


def test_revin_mean_centres_but_does_not_rescale():
    """Window mean removed; window scale left exactly as it was."""
    torch.manual_seed(0)
    nm = build_norm("revin_mean", num_features=3)
    x = torch.randn(16, 96, 3) * 10 - 7
    xn = nm(x, "norm")
    assert torch.allclose(xn.mean(dim=1), torch.zeros(16, 3), atol=1e-5)
    # RevIN would force unit variance here; this arm must not touch it
    assert torch.allclose(xn.std(dim=1, unbiased=False),
                          x.std(dim=1, unbiased=False), atol=1e-6)


def test_revin_mean_has_no_learnable_parameters():
    """No affine: the arm must add zero parameters to the model."""
    nm = build_norm("revin_mean", num_features=5)
    assert list(nm.parameters()) == []


def test_revin_mean_horizon_shape_change():
    """denorm is applied to the horizon (h != L), reusing the stored mean."""
    nm = build_norm("revin_mean", num_features=7)
    nm(torch.randn(8, 336, 7), "norm")
    assert nm(torch.randn(8, 96, 7), "denorm").shape == (8, 96, 7)


def test_revin_mean_restores_the_stale_level():
    """The restored level is the lookback mean, not the horizon's own mean —
    this staleness is the mechanism under test, so pin it explicitly."""
    x = torch.zeros(1, 100, 1)
    x[0, :, 0] = 5.0                      # lookback sits at level 5
    nm = build_norm("revin_mean", num_features=1)
    nm(x, "norm")
    out = nm(torch.zeros(1, 24, 1), "denorm")   # backbone predicts zero residual
    assert torch.allclose(out, torch.full((1, 24, 1), 5.0))


def test_revin_mean_differs_from_revin_only_by_scale_and_affine():
    """RevIN's normalized window equals this arm's, rescaled — the decomposition
    RevIN = mean + scale (+ affine) that the ablation relies on."""
    torch.manual_seed(0)
    x = torch.randn(8, 96, 4) * 3 + 1
    mean_only = build_norm("revin_mean", num_features=4)(x, "norm")
    revin = build_norm("revin", num_features=4)(x, "norm")   # affine is identity at init
    sd = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
    assert torch.allclose(revin, mean_only / sd, atol=1e-5)
