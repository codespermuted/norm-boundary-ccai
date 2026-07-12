"""G0 acceptance: minimal end-to-end pipeline runs and learns."""

import torch

from src.train import run


def test_gpu_one_step():
    """Tiny tensor, one optimizer step on the training device (GPU if present)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.nn.Linear(8, 1).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    x, y = torch.randn(16, 8, device=device), torch.randn(16, 1, device=device)
    loss0 = torch.nn.functional.mse_loss(model(x), y)
    loss0.backward()
    opt.step()
    loss1 = torch.nn.functional.mse_loss(model(x), y)
    assert torch.isfinite(loss1)
    assert loss1.item() < loss0.item()


def test_end_to_end_etth1_rlinear_revin():
    """ETTh1 + RLinear + RevIN trains a few steps end-to-end and yields finite metrics."""
    cfg = {
        "dataset": "etth1",
        "norm": "revin",
        "backbone": "rlinear",
        "lookback": 336,
        "horizon": 96,
        "seed": 0,
        "batch_size": 64,
        "lr": 0.005,
        "epochs": 1,
        "max_steps": 5,
        "patience": 3,
        "device": "cuda",
    }
    result = run(cfg, log_mlflow=False)
    assert result["steps"] == 5
    assert 0 < result["test_mse"] < 100
    assert 0 < result["test_mae"] < 100
