import torch
import torch.nn as nn


class NoNorm(nn.Module):
    """RAW: identity. Global z-scoring is handled at the dataset level,
    so the model-side normalization is a no-op."""

    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode not in ("norm", "denorm"):
            raise ValueError(f"mode must be 'norm' or 'denorm', got {mode!r}")
        return x
