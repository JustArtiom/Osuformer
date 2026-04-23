from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import Tensor


@runtime_checkable
class AudioEncoder(Protocol):
    output_dim: int
    feature_rate_hz: float

    def __call__(self, mel: Tensor, key_padding_mask: Tensor | None = None) -> Tensor: ...
