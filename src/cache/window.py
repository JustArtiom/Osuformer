from __future__ import annotations

from src.config.schemas.tokenizer import TokenizerConfig
from src.osu_tokenizer import Event, EventType, Vocab, attach_rel_times


def type_index_map(vocab: Vocab) -> dict[EventType, int]:
    order = [er.type for er in vocab.output_ranges] + [er.type for er in vocab.input_ranges]
    return {t: i for i, t in enumerate(order)}


def type_from_index(vocab: Vocab, idx: int) -> EventType:
    order = [er.type for er in vocab.output_ranges] + [er.type for er in vocab.input_ranges]
    return order[idx]


def slice_window_events(
    event_types: list[int],
    event_values: list[int],
    vocab: Vocab,
    config: TokenizerConfig,
    window_start_ms: float,
    clamp_to_window: bool = True,
) -> list[Event]:
    dt_bin = config.dt_bin_ms
    window_start_bin = int(round(window_start_ms / dt_bin))
    total_ms = config.context_ms + config.generate_ms + config.lookahead_ms
    window_end_bin = window_start_bin + int(round(total_ms / dt_bin))

    abs_type_idx = type_index_map(vocab)[EventType.ABS_TIME]
    order = [er.type for er in vocab.output_ranges] + [er.type for er in vocab.input_ranges]

    out: list[Event] = []
    keep_group = False
    for type_idx, value in zip(event_types, event_values):
        if type_idx == abs_type_idx:
            keep_group = window_start_bin <= value < window_end_bin
            if keep_group:
                shifted = value - window_start_bin
                if clamp_to_window:
                    shifted = max(0, shifted)
                out.append(Event(type=EventType.ABS_TIME, value=shifted))
            continue
        if not keep_group:
            continue
        out.append(Event(type=order[type_idx], value=value))

    return attach_rel_times(out, vocab, config)
