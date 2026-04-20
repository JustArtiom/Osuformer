from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
    max_steps: int,
    min_lr: float,
    base_lr: float,
) -> LambdaLR:
    min_ratio = max(0.0, min_lr / base_lr)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        if step >= max_steps:
            return min_ratio
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)
