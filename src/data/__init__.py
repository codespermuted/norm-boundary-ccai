from src.data.etth import ETTHourly, load_etth_frame
from src.data.ltsf import LTSFDataset

DATASET_REGISTRY = {
    "etth1": lambda **kw: ETTHourly(name="ETTh1", **kw),
    "etth2": lambda **kw: ETTHourly(name="ETTh2", **kw),
    "electricity": lambda **kw: LTSFDataset(name="electricity", **kw),
    "weather": lambda **kw: LTSFDataset(name="weather", **kw),
}


def build_covariate_dataset(name: str, **kwargs):
    """Curated exogenous-driven datasets (jeju_wind, gefcom_*, kpx_demand_*)."""
    from src.data.covariate import CovariateSeries

    return CovariateSeries(name, **kwargs)


def build_dataset(name: str, **kwargs):
    if name not in DATASET_REGISTRY:
        raise KeyError(f"unknown dataset '{name}', available: {sorted(DATASET_REGISTRY)}")
    return DATASET_REGISTRY[name](**kwargs)
