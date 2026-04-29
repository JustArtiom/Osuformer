from __future__ import annotations

from torch import Tensor, nn


class AdaLNModulation(nn.Module):
    def __init__(self, d_model: int, cond_dim: int, num_outputs: int):
        super().__init__()
        self.num_outputs = num_outputs
        self.d_model = d_model
        self.proj = nn.Linear(cond_dim, num_outputs * d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, cond: Tensor) -> tuple[Tensor, ...]:
        out = self.proj(cond)
        return tuple(out.chunk(self.num_outputs, dim=-1))


def modulate(x: Tensor, scale: Tensor, shift: Tensor) -> Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def gated_residual(x: Tensor, residual_addition: Tensor, gate: Tensor) -> Tensor:
    return x + gate.unsqueeze(1) * residual_addition
