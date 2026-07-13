"""G3 sanity: new components on ETTh1 h=96, literature-range check.

Expected ranges (multivariate, std-scale MSE; generous bounds — full tuning
happens in G4): PatchTST/RevIN <= 0.45, SegRNN/RevIN <= 0.47,
SAN/RLinear <= 0.45, FAN/RLinear <= 0.50. Literature references:
PatchTST 0.370-0.414, SegRNN ~0.35 (L=720), SAN-Linear ~0.37.

Usage: uv run python -m experiments.g3_sanity
"""

from src.train import run

CONFIGS = [
    dict(dataset="etth1", norm="revin", backbone="patchtst", lookback=336,
         horizon=96, seed=0, batch_size=64, lr=1e-4, epochs=12, patience=3,
         device="cuda"),
    dict(dataset="etth1", norm="revin", backbone="segrnn", lookback=720,
         horizon=96, seed=0, batch_size=64, lr=1e-3, epochs=12, patience=3,
         device="cuda", backbone_kwargs={"seg_len": 48}),
    dict(dataset="etth1", norm="san", backbone="rlinear", lookback=336,
         horizon=96, seed=0, batch_size=64, lr=5e-3, epochs=12, patience=3,
         device="cuda", norm_kwargs={"period_len": 24}),
    dict(dataset="etth1", norm="fan", backbone="rlinear", lookback=336,
         horizon=96, seed=0, batch_size=64, lr=5e-3, epochs=12, patience=3,
         device="cuda", norm_kwargs={"freq_topk": 20}),
]

BOUNDS = {"patchtst": 0.45, "segrnn": 0.47, "san": 0.45, "fan": 0.50}

if __name__ == "__main__":
    rows = []
    for cfg in CONFIGS:
        res = run(cfg, log_mlflow=True)
        tag = cfg["backbone"] if cfg["norm"] == "revin" else cfg["norm"]
        ok = res["test_mse"] <= BOUNDS[tag]
        rows.append((tag, res["test_mse"], res["test_mae"], ok))
        print(f"SANITY {tag}: mse={res['test_mse']:.4f} "
              f"(bound {BOUNDS[tag]}) {'OK' if ok else 'OUT-OF-RANGE'}", flush=True)
    print("\nSummary:")
    for tag, mse, mae, ok in rows:
        print(f"  {tag:10s} mse={mse:.4f} mae={mae:.4f} {'✓' if ok else '✗'}")
