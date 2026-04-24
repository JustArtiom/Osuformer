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
        rhythm_weight: float = 3.0,
        slider_weight: float = 1.0,
        default_weight: float = 1.0,
        pad_id: int = int(SpecialToken.PAD),
    ):
        self.pad_id = pad_id
        self.rhythm_weight = rhythm_weight
        self.slider_weight = slider_weight
        self.default_weight = default_weight
        self._rhythm_ranges: list[tuple[int, int]] = [
            vocab.token_range(t)
            for t in (EventType.ABS_TIME, EventType.BEAT, EventType.MEASURE, EventType.TIMING_POINT)
        ]
        self._slider_ranges: list[tuple[int, int]] = [
            vocab.token_range(t)
            for t in (
                EventType.SLIDER_END,
                EventType.DURATION,
                EventType.SLIDER_SLIDES,
                EventType.SCROLL_SPEED,
            )
        ]

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
        is_rhythm = torch.zeros_like(target_ids, dtype=torch.bool)
        for start, end in self._rhythm_ranges:
            is_rhythm = is_rhythm | ((target_ids >= start) & (target_ids < end))
        is_slider = torch.zeros_like(target_ids, dtype=torch.bool)
        for start, end in self._slider_ranges:
            is_slider = is_slider | ((target_ids >= start) & (target_ids < end))
        loss_weights = torch.full_like(target_ids, fill_value=0, dtype=torch.float32).fill_(self.default_weight)
        loss_weights = torch.where(
            is_slider,
            torch.full_like(loss_weights, self.slider_weight),
            loss_weights,
        )
        loss_weights = torch.where(
            is_rhythm,
            torch.full_like(loss_weights, self.rhythm_weight),
            loss_weights,
        )
        return Batch(
            mel=mel,
            input_ids=input_ids,
            target_ids=target_ids,
            loss_mask=loss_mask,
            loss_weights=loss_weights,
            token_pad_mask=token_pad_mask,
        )
