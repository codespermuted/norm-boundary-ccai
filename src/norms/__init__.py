from src.norms.base import NoNorm
from src.norms.fan import FAN
from src.norms.revin import RevIN
from src.norms.san import SAN

NORM_REGISTRY = {
    "raw": NoNorm,
    "revin": RevIN,
    "san": SAN,
    "fan": FAN,
}

_NEEDS_SHAPE = {"san", "fan"}


def build_norm(name: str, num_features: int, lookback: int | None = None,
               horizon: int | None = None, **kwargs):
    """CondNorm is a data-space transform (src/norms/condnorm.py), not a
    model-side module, so it is not built here."""
    if name not in NORM_REGISTRY:
        raise KeyError(f"unknown norm '{name}', available: {sorted(NORM_REGISTRY)}")
    if name in _NEEDS_SHAPE:
        if lookback is None or horizon is None:
            raise ValueError(f"{name} needs lookback and horizon")
        return NORM_REGISTRY[name](num_features=num_features, lookback=lookback,
                                   horizon=horizon, **kwargs)
    return NORM_REGISTRY[name](num_features=num_features, **kwargs)
