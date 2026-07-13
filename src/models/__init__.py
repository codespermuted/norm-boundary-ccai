from src.models.patchtst import PatchTST
from src.models.rlinear import RLinear
from src.models.segrnn import SegRNN

BACKBONE_REGISTRY = {
    "rlinear": RLinear,
    "patchtst": PatchTST,
    "segrnn": SegRNN,
    # 'lgbm_dms' is not a torch module: src/train.py routes it separately
    # (src/models/lgbm_dms.py)
}


def build_backbone(name: str, lookback: int, horizon: int, num_features: int, **kwargs):
    if name not in BACKBONE_REGISTRY:
        raise KeyError(f"unknown backbone '{name}', available: {sorted(BACKBONE_REGISTRY)}")
    return BACKBONE_REGISTRY[name](
        lookback=lookback, horizon=horizon, num_features=num_features, **kwargs
    )
