from src.models.rlinear import RLinear

BACKBONE_REGISTRY = {
    "rlinear": RLinear,
}


def build_backbone(name: str, lookback: int, horizon: int, num_features: int, **kwargs):
    if name not in BACKBONE_REGISTRY:
        raise KeyError(f"unknown backbone '{name}', available: {sorted(BACKBONE_REGISTRY)}")
    return BACKBONE_REGISTRY[name](
        lookback=lookback, horizon=horizon, num_features=num_features, **kwargs
    )
