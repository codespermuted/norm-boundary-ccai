from src.norms.base import NoNorm
from src.norms.revin import RevIN

NORM_REGISTRY = {
    "raw": NoNorm,
    "revin": RevIN,
}


def build_norm(name: str, num_features: int, **kwargs):
    if name not in NORM_REGISTRY:
        raise KeyError(f"unknown norm '{name}', available: {sorted(NORM_REGISTRY)}")
    return NORM_REGISTRY[name](num_features=num_features, **kwargs)
