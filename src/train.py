"""Single training entry point: config -> run -> MLflow logging.

Usage:
    uv run python -m src.train --config configs/smoke_etth1.yaml

Environment invariants (see CONTRIBUTING notes):
  - GPU1 only (GPU0 is reserved for Xorg) -> CUDA_VISIBLE_DEVICES defaults to "1"
  - deterministic algorithms where possible
  - MLflow experiment "norm-boundary", run name {dataset}_{norm}_{backbone}_{h}_{seed}
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import random
import subprocess
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data import build_dataset
from src.models import build_backbone
from src.models.rlinear import NormWrapper
from src.norms import build_norm

MLFLOW_EXPERIMENT = "norm-boundary"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=os.path.dirname(__file__)
        ).strip()
    except Exception:
        return "unknown"


def evaluate(model, loader, device) -> dict:
    model.eval()
    se_sum, ae_sum, count = 0.0, 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            se_sum += torch.sum((pred - y) ** 2).item()
            ae_sum += torch.sum(torch.abs(pred - y)).item()
            count += y.numel()
    return {"mse": se_sum / count, "mae": ae_sum / count}


def run(cfg: dict, log_mlflow: bool = True) -> dict:
    set_seed(cfg["seed"])
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")

    ds = build_dataset(cfg["dataset"], lookback=cfg["lookback"], horizon=cfg["horizon"])
    loaders = {
        split: DataLoader(
            ds.windows(split),
            batch_size=cfg["batch_size"],
            shuffle=(split == "train"),
            drop_last=False,  # TFB: drop_last biases test metrics
        )
        for split in ("train", "val", "test")
    }

    norm = build_norm(cfg["norm"], num_features=ds.num_features)
    backbone = build_backbone(
        cfg["backbone"], lookback=cfg["lookback"], horizon=cfg["horizon"],
        num_features=ds.num_features,
    )
    model = NormWrapper(backbone, norm).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = torch.nn.MSELoss()

    max_steps = cfg.get("max_steps") or float("inf")
    patience = cfg.get("patience", 3)
    best_val, best_state, bad_epochs, global_step = float("inf"), None, 0, 0

    t0 = time.time()
    for epoch in range(cfg["epochs"]):
        model.train()
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            global_step += 1
            if global_step >= max_steps:
                break
        val = evaluate(model, loaders["val"], device)
        if val["mse"] < best_val:
            best_val = val["mse"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        print(f"epoch {epoch}: val_mse={val['mse']:.6f} (best={best_val:.6f})")
        if bad_epochs >= patience or global_step >= max_steps:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    test = evaluate(model, loaders["test"], device)
    wall = time.time() - t0

    result = {
        "test_mse": test["mse"], "test_mae": test["mae"],
        "best_val_mse": best_val, "steps": global_step, "wall_seconds": wall,
    }

    if log_mlflow:
        import mlflow

        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        run_name = f"{cfg['dataset']}_{cfg['norm']}_{cfg['backbone']}_{cfg['horizon']}_{cfg['seed']}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(cfg)
            mlflow.log_metrics(result)
            mlflow.set_tags({"git_commit": git_commit_hash(), "device": str(device)})
    print(f"result: {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run(cfg, log_mlflow=not args.no_mlflow)


if __name__ == "__main__":
    main()
