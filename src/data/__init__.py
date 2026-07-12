from src.data.etth import ETTHourly, load_etth_frame

DATASET_REGISTRY = {
    "etth1": lambda **kw: ETTHourly(name="ETTh1", **kw),
    "etth2": lambda **kw: ETTHourly(name="ETTh2", **kw),
}


def build_dataset(name: str, **kwargs):
    if name not in DATASET_REGISTRY:
        raise KeyError(f"unknown dataset '{name}', available: {sorted(DATASET_REGISTRY)}")
    return DATASET_REGISTRY[name](**kwargs)
