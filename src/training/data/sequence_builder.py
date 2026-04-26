from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch import Tensor

from src.cache.metadata import MetadataRecord
from src.config.schemas.tokenizer import TokenizerConfig
from src.osu_tokenizer import Event, EventType, SpecialToken, Vocab


@dataclass
class SequenceSample:
    input_ids: Tensor
    target_ids: Tensor
    loss_mask: Tensor
    length: int


class SequenceBuilder:
    def __init__(
        self,
        vocab: Vocab,
        tokenizer_cfg: TokenizerConfig,
        max_len: int,
        history_event_count: int,
        timing_jitter_bins: int = 0,
        cfg_dropout_prob: float = 0.0,
    ):
        self.vocab = vocab
        self.cfg = tokenizer_cfg
        self.max_len = max_len
        self.history_event_count = history_event_count
        self.timing_jitter_bins = max(0, timing_jitter_bins)
        self.cfg_dropout_prob = max(0.0, min(1.0, cfg_dropout_prob))
        self._type_order = [er.type for er in vocab.output_ranges] + [er.type for er in vocab.input_ranges]
        self._abs_type_idx = self._type_order.index(EventType.ABS_TIME)
        total_ms = tokenizer_cfg.context_ms + tokenizer_cfg.generate_ms + tokenizer_cfg.lookahead_ms
        self._window_bin_count = total_ms // tokenizer_cfg.dt_bin_ms
        self._context_bin = tokenizer_cfg.context_ms // tokenizer_cfg.dt_bin_ms
        self._max_rel_bin = vocab.range_for(EventType.REL_TIME).max_value
        self._abs_token_start, self._abs_token_end = vocab.token_range(EventType.ABS_TIME)
        self._rel_token_start, self._rel_token_end = vocab.token_range(EventType.REL_TIME)

    def build(
        self,
        event_types: list[int],
        event_values: list[int],
        map_record: dict,
        metadata: MetadataRecord | None,
        window_start_ms: float,
    ) -> SequenceSample:
        window_start_bin = int(round(window_start_ms / self.cfg.dt_bin_ms))
        history_groups, window_groups = self._slice_groups(event_types, event_values, window_start_bin)

        tokens: list[int] = [int(SpecialToken.SOS_SEQ)]
        if self.cfg_dropout_prob <= 0.0 or random.random() >= self.cfg_dropout_prob:
            for ev in self._conditioning(map_record, metadata):
                tokens.append(self.vocab.encode_event(ev))
        tokens.append(int(SpecialToken.MAP_START))

        history_trimmed = history_groups[-self.history_event_count :] if self.history_event_count > 0 else []
        last_raw_bin: int | None = None
        for group in history_trimmed:
            raw_abs, events = group
            rel = self._compute_rel(last_raw_bin, raw_abs)
            tokens.append(int(SpecialToken.TIME_ABS_NULL))
            tokens.append(self.vocab.encode_event(Event(EventType.REL_TIME, rel)))
            last_raw_bin = raw_abs
            for ev in events:
                tokens.append(self.vocab.encode_event(ev))

        tokens.append(int(SpecialToken.HISTORY_END))

        sos_idx: int | None = None
        for group in window_groups:
            raw_abs, events = group
            window_local = raw_abs - window_start_bin
            if sos_idx is None and window_local >= self._context_bin:
                tokens.append(int(SpecialToken.SOS))
                sos_idx = len(tokens) - 1
            rel = self._compute_rel(last_raw_bin, raw_abs)
            tokens.append(self.vocab.encode_event(Event(EventType.ABS_TIME, window_local)))
            tokens.append(self.vocab.encode_event(Event(EventType.REL_TIME, rel)))
            last_raw_bin = raw_abs
            for ev in events:
                tokens.append(self.vocab.encode_event(ev))

        if sos_idx is None:
            tokens.append(int(SpecialToken.SOS))
            sos_idx = len(tokens) - 1

        tokens.append(int(SpecialToken.EOS))
        eos_idx = len(tokens) - 1

        if len(tokens) > self.max_len:
            tokens = tokens[: self.max_len]
            eos_idx = min(eos_idx, self.max_len - 1)

        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        loss_mask = torch.zeros(input_ids.shape[0], dtype=torch.bool)
        if sos_idx < eos_idx:
            start = min(sos_idx, loss_mask.shape[0])
            end = min(eos_idx, loss_mask.shape[0])
            loss_mask[start:end] = True
        if self.timing_jitter_bins > 0:
            input_ids = self._jitter_time_tokens(input_ids)
        return SequenceSample(
            input_ids=input_ids,
            target_ids=target_ids,
            loss_mask=loss_mask,
            length=input_ids.shape[0],
        )

    def _jitter_time_tokens(self, input_ids: Tensor) -> Tensor:
        jitter_hi = self.timing_jitter_bins + 1
        offsets = torch.randint(-self.timing_jitter_bins, jitter_hi, input_ids.shape, dtype=input_ids.dtype)
        is_abs = (input_ids >= self._abs_token_start) & (input_ids < self._abs_token_end)
        is_rel = (input_ids >= self._rel_token_start) & (input_ids < self._rel_token_end)
        out = input_ids
        if bool(is_abs.any().item()):
            shifted_abs = torch.clamp(out + offsets, self._abs_token_start, self._abs_token_end - 1)
            out = torch.where(is_abs, shifted_abs, out)
        if bool(is_rel.any().item()):
            shifted_rel = torch.clamp(out + offsets, self._rel_token_start, self._rel_token_end - 1)
            out = torch.where(is_rel, shifted_rel, out)
        return out

    def _slice_groups(
        self,
        event_types: list[int],
        event_values: list[int],
        window_start_bin: int,
    ) -> tuple[list[tuple[int, list[Event]]], list[tuple[int, list[Event]]]]:
        window_end_bin = window_start_bin + self._window_bin_count
        history: list[tuple[int, list[Event]]] = []
        window: list[tuple[int, list[Event]]] = []
        current_abs: int | None = None
        current_events: list[Event] = []
        for type_idx, value in zip(event_types, event_values):
            if type_idx == self._abs_type_idx:
                self._flush_group(current_abs, current_events, window_start_bin, window_end_bin, history, window)
                current_abs = value
                current_events = []
            else:
                current_events.append(Event(type=self._type_order[type_idx], value=value))
        self._flush_group(current_abs, current_events, window_start_bin, window_end_bin, history, window)
        return history, window

    @staticmethod
    def _flush_group(
        abs_bin: int | None,
        events: list[Event],
        window_start_bin: int,
        window_end_bin: int,
        history: list[tuple[int, list[Event]]],
        window: list[tuple[int, list[Event]]],
    ) -> None:
        if abs_bin is None:
            return
        if abs_bin < window_start_bin:
            history.append((abs_bin, events))
        elif abs_bin < window_end_bin:
            window.append((abs_bin, events))

    def _compute_rel(self, last: int | None, current: int) -> int:
        if last is None:
            return 0
        delta = current - last
        if delta < 0 or delta > self._max_rel_bin:
            return self._max_rel_bin
        return delta

    def _conditioning(self, map_record: dict, metadata: MetadataRecord | None) -> list[Event]:
        events: list[Event] = []
        events.append(Event(EventType.HITSOUNDED, 1 if map_record.get("hitsounded", False) else 0))
        events.append(self._cs_event(map_record.get("circle_size", 5.0)))
        events.append(self._stat_event(EventType.AR, map_record.get("approach_rate", 5.0), self.cfg.ar_step))
        events.append(self._stat_event(EventType.OD, map_record.get("overall_difficulty", 5.0), self.cfg.od_step))
        events.append(self._stat_event(EventType.HP, map_record.get("hp_drain_rate", 5.0), self.cfg.hp_step))
        sv = int(round(float(map_record.get("slider_multiplier", 1.4)) * 100))
        sv = max(self.cfg.global_sv_min, min(self.cfg.global_sv_max, sv))
        events.append(Event(EventType.GLOBAL_SV, sv))
        duration_ms = float(map_record.get("duration_ms", 0.0))
        bucket = int(duration_ms / (self.cfg.song_length_bucket_s * 1000.0))
        bucket = max(0, min(self.cfg.song_length_buckets - 1, bucket))
        events.append(Event(EventType.SONG_LENGTH, bucket))

        if metadata is not None:
            star = metadata.star_rating
            bins = self.cfg.difficulty_bins
            idx = int(round(star * (bins - 1) / self.cfg.difficulty_max_star))
            idx = max(0, min(bins - 1, idx))
            events.append(Event(EventType.DIFFICULTY, idx))
            year = metadata.ranked_year
            if year >= self.cfg.year_min and year <= self.cfg.year_max:
                events.append(Event(EventType.YEAR, year))
            for descriptor_idx in metadata.descriptor_indices:
                events.append(Event(EventType.DESCRIPTOR, descriptor_idx))
        return events

    def _cs_event(self, value: float) -> Event:
        return self._stat_event(EventType.CS, value, self.cfg.cs_step)

    def _stat_event(self, event_type: EventType, value: float, step: float) -> Event:
        idx = int(round(float(value) / step))
        er = self.vocab.range_for(event_type)
        idx = max(er.min_value, min(er.max_value, idx))
        return Event(type=event_type, value=idx)
