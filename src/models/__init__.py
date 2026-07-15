from src.models.itransformer import ITransformer
from src.models.patchtst import PatchTST
from src.models.rlinear import RLinear
from src.models.segrnn import SegRNN
from src.models.timexer import TimeXer


class TimeXerM(TimeXer):
    """'M'-mode adapter: standard (B,L,C)->(B,h,C) backbone interface."""

    def __init__(self, lookback, horizon, num_features, **kw):
        super().__init__(lookback, horizon, **kw)

    def forward(self, x):  # noqa: D102
        return self.forward_multi(x)


BACKBONE_REGISTRY = {
    "rlinear": RLinear,
    "patchtst": PatchTST,
    "segrnn": SegRNN,
    "itransformer": ITransformer,
    "timexer": TimeXerM,   # multivariate mode; exog 'MS' mode is routed in
                           # experiments/g4_grid.py (needs the covariate feed)
    # 'lgbm_dms' is not a torch module: src/train.py routes it separately
    # (src/models/lgbm_dms.py)
}


def build_backbone(name: str, lookback: int, horizon: int, num_features: int, **kwargs):
    if name not in BACKBONE_REGISTRY:
        raise KeyError(f"unknown backbone '{name}', available: {sorted(BACKBONE_REGISTRY)}")
    return BACKBONE_REGISTRY[name](
        lookback=lookback, horizon=horizon, num_features=num_features, **kwargs
    )
