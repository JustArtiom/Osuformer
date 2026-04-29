from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn


@dataclass
class AuxOutputs:
    star: Tensor
    descriptor_logits: Tensor
    density: Tensor


class AuxHeads(nn.Module):
    def __init__(self, encoder_dim: int, descriptor_count: int, hidden_dim: int = 256):
        super().__init__()
        self.norm = nn.LayerNorm(encoder_dim)
        self.shared = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.SiLU(),
        )
        self.star_head = nn.Linear(hidden_dim, 1)
        self.descriptor_head = nn.Linear(hidden_dim, descriptor_count)
        self.density_head = nn.Linear(hidden_dim, 1)

    def forward(self, encoder_pool: Tensor) -> AuxOutputs:
        h = self.shared(self.norm(encoder_pool))
        star = self.star_head(h).squeeze(-1)
        descriptor_logits = self.descriptor_head(h)
        density = self.density_head(h).squeeze(-1)
        return AuxOutputs(star=star, descriptor_logits=descriptor_logits, density=density)


def pool_encoder_output(memory: Tensor, key_padding_mask: Tensor | None) -> Tensor:
    if key_padding_mask is None:
        return memory.mean(dim=1)
    valid = (~key_padding_mask).to(memory.dtype).unsqueeze(-1)
    summed = (memory * valid).sum(dim=1)
    counts = valid.sum(dim=1).clamp(min=1.0)
    return summed / counts
