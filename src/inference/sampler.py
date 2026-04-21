from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from src.osu_tokenizer import EventType


@dataclass
class SamplingConfig:
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    time_temperature: float = 0.6
    event_bias: dict[EventType, float] = field(default_factory=dict)


def sample_next_token(
    logits: Tensor,
    config: SamplingConfig,
    is_time_token: bool = False,
) -> int:
    temperature = config.time_temperature if is_time_token else config.temperature
    if temperature <= 0:
        return int(logits.argmax().item())
    scaled = logits / max(1e-6, temperature)
    if config.top_k > 0 and config.top_k < scaled.numel():
        topk_vals, _ = torch.topk(scaled, config.top_k)
        threshold = topk_vals[-1]
        scaled = torch.where(scaled < threshold, torch.full_like(scaled, float("-inf")), scaled)
    if 0.0 < config.top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cum = sorted_probs.cumsum(dim=-1)
        mask = cum > config.top_p
        mask[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
        scaled = torch.full_like(scaled, float("-inf")).scatter(dim=-1, index=sorted_indices, src=sorted_logits)
    probs = torch.softmax(scaled, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())
