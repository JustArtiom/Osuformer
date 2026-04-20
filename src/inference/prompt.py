from __future__ import annotations

from dataclasses import dataclass, field

from src.config.schemas.tokenizer import TokenizerConfig
from src.osu_tokenizer import DESCRIPTOR_TO_INDEX, Event, EventType, SpecialToken, Vocab


@dataclass
class GenerationPrompt:
    star_rating: float | None = None
    descriptors: list[str] = field(default_factory=list)
    year: int | None = None
    hitsounded: bool = True
    cs: float = 4.0
    ar: float = 9.0
    od: float = 8.0
    hp: float = 6.0
    slider_multiplier: float = 1.4
    song_length_ms: float = 0.0


def build_conditioning_tokens(
    prompt: GenerationPrompt,
    vocab: Vocab,
    tokenizer_cfg: TokenizerConfig,
) -> list[int]:
    out: list[int] = [int(SpecialToken.SOS_SEQ)]
    for ev in _conditioning_events(prompt, vocab, tokenizer_cfg):
        out.append(vocab.encode_event(ev))
    out.append(int(SpecialToken.MAP_START))
    return out


def _conditioning_events(prompt: GenerationPrompt, vocab: Vocab, cfg: TokenizerConfig) -> list[Event]:
    events: list[Event] = []
    events.append(Event(EventType.HITSOUNDED, 1 if prompt.hitsounded else 0))
    events.append(_stat_event(vocab, EventType.CS, prompt.cs, cfg.cs_step))
    events.append(_stat_event(vocab, EventType.AR, prompt.ar, cfg.ar_step))
    events.append(_stat_event(vocab, EventType.OD, prompt.od, cfg.od_step))
    events.append(_stat_event(vocab, EventType.HP, prompt.hp, cfg.hp_step))
    sv = int(round(prompt.slider_multiplier * 100))
    sv = max(cfg.global_sv_min, min(cfg.global_sv_max, sv))
    events.append(Event(EventType.GLOBAL_SV, sv))
    bucket = int(prompt.song_length_ms / (cfg.song_length_bucket_s * 1000.0))
    bucket = max(0, min(cfg.song_length_buckets - 1, bucket))
    events.append(Event(EventType.SONG_LENGTH, bucket))
    if prompt.star_rating is not None:
        bins = cfg.difficulty_bins
        idx = int(round(prompt.star_rating * (bins - 1) / cfg.difficulty_max_star))
        idx = max(0, min(bins - 1, idx))
        events.append(Event(EventType.DIFFICULTY, idx))
    if prompt.year is not None and cfg.year_min <= prompt.year <= cfg.year_max:
        events.append(Event(EventType.YEAR, prompt.year))
    for tag in prompt.descriptors:
        idx = DESCRIPTOR_TO_INDEX.get(tag)
        if idx is None:
            continue
        events.append(Event(EventType.DESCRIPTOR, idx))
    return events


def _stat_event(vocab: Vocab, event_type: EventType, value: float, step: float) -> Event:
    idx = int(round(value / step))
    er = vocab.range_for(event_type)
    idx = max(er.min_value, min(er.max_value, idx))
    return Event(type=event_type, value=idx)
