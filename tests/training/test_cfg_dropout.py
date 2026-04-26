import random

from src.config.loader import load_config
from src.osu_tokenizer import SpecialToken, Vocab, beatmap_to_events, collect_timing_events, merge_by_time
from src.training.data.sequence_builder import SequenceBuilder

from tests.osu_tokenizer.fixtures import build_beatmap, make_circle, make_timing_point


def _make_inputs() -> tuple[list[int], list[int], dict, Vocab]:
    cfg = load_config("config/config.yaml").tokenizer
    vocab = Vocab(cfg)
    beatmap = build_beatmap(
        hit_objects=[make_circle(time=500.0 + i * 500.0) for i in range(8)],
        timing_points=[make_timing_point(0.0, 120.0)],
    )
    stream = beatmap_to_events(beatmap, 0.0, vocab, cfg, clamp_abs_time=False)
    timing = collect_timing_events(beatmap, 0.0, vocab, cfg, clamp_abs_time=False)
    merged = merge_by_time(stream.events, timing)
    type_order = [er.type for er in vocab.output_ranges] + [er.type for er in vocab.input_ranges]
    type_to_idx = {t: i for i, t in enumerate(type_order)}
    event_types = [type_to_idx[e.type] for e in merged]
    event_values = [e.value for e in merged]
    map_record = {
        "hitsounded": True,
        "circle_size": 4.0,
        "approach_rate": 9.0,
        "overall_difficulty": 8.0,
        "hp_drain_rate": 6.0,
        "slider_multiplier": 1.4,
        "duration_ms": 5000.0,
    }
    return event_types, event_values, map_record, vocab


def _build(prob: float, seed: int) -> tuple[list[int], int, int]:
    cfg = load_config("config/config.yaml").tokenizer
    event_types, event_values, map_record, vocab = _make_inputs()
    builder = SequenceBuilder(
        vocab=vocab,
        tokenizer_cfg=cfg,
        max_len=2048,
        history_event_count=32,
        timing_jitter_bins=0,
        cfg_dropout_prob=prob,
    )
    random.seed(seed)
    seq = builder.build(
        event_types=event_types,
        event_values=event_values,
        map_record=map_record,
        metadata=None,
        window_start_ms=0.0,
    )
    ids = [int(x) for x in seq.input_ids.tolist()]
    sos_seq = ids.index(int(SpecialToken.SOS_SEQ))
    map_start = ids.index(int(SpecialToken.MAP_START))
    return ids, sos_seq, map_start


def test_cfg_dropout_zero_keeps_full_conditioning() -> None:
    _, sos_seq, map_start = _build(prob=0.0, seed=0)
    assert map_start - sos_seq > 1


def test_cfg_dropout_one_drops_all_conditioning() -> None:
    _, sos_seq, map_start = _build(prob=1.0, seed=0)
    assert map_start - sos_seq == 1


def test_cfg_dropout_partial_produces_both_modes() -> None:
    full_count = 0
    null_count = 0
    for seed in range(64):
        _, sos_seq, map_start = _build(prob=0.5, seed=seed)
        if map_start - sos_seq == 1:
            null_count += 1
        else:
            full_count += 1
    assert full_count > 0
    assert null_count > 0
