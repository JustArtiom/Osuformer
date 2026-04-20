from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from src.osu_tokenizer import EventType, SpecialToken, Vocab

from .dataset import OsuSample


@dataclass
class Batch:
    mel: Tensor
    input_ids: Tensor
    target_ids: Tensor
    loss_mask: Tensor
    loss_weights: Tensor
    token_pad_mask: Tensor


class Collator:
    def __init__(
        self,
        vocab: Vocab,
        abs_time_weight: float = 3.0,
        default_weight: float = 1.0,
        pad_id: int = int(SpecialToken.PAD),
    ):
        self.pad_id = pad_id
        self.abs_time_weight = abs_time_weight
        self.default_weight = default_weight
        start, end = vocab.token_range(EventType.ABS_TIME)
        self._abs_time_start = start
        self._abs_time_end = end

    def __call__(self, samples: list[OsuSample]) -> Batch:
        mel = torch.stack([s.mel for s in samples], dim=0)
        max_len = max(s.input_ids.shape[0] for s in samples)
        b = len(samples)
        input_ids = torch.full((b, max_len), self.pad_id, dtype=torch.long)
        target_ids = torch.full((b, max_len), self.pad_id, dtype=torch.long)
        loss_mask = torch.zeros((b, max_len), dtype=torch.bool)
        token_pad_mask = torch.ones((b, max_len), dtype=torch.bool)
        for i, s in enumerate(samples):
            n = s.input_ids.shape[0]
            input_ids[i, :n] = s.input_ids
            target_ids[i, :n] = s.target_ids
            loss_mask[i, :n] = s.loss_mask
            token_pad_mask[i, :n] = False
        is_abs_time = (target_ids >= self._abs_time_start) & (target_ids < self._abs_time_end)
        loss_weights = torch.where(
            is_abs_time,
            torch.full_like(target_ids, fill_value=0, dtype=torch.float32).fill_(self.abs_time_weight),
            torch.full_like(target_ids, fill_value=0, dtype=torch.float32).fill_(self.default_weight),
        )
        return Batch(
            mel=mel,
            input_ids=input_ids,
            target_ids=target_ids,
            loss_mask=loss_mask,
            loss_weights=loss_weights,
            token_pad_mask=token_pad_mask,
        )
