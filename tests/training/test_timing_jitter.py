import torch

from src.config.loader import load_config
from src.osu_tokenizer import EventType, Vocab, beatmap_to_events, collect_timing_events, merge_by_time
from src.training.data.sequence_builder import SequenceBuilder

from tests.osu_tokenizer.fixtures import build_beatmap, make_circle, make_timing_point


def _build_sample(jitter_bins: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
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
    builder = SequenceBuilder(
        vocab=vocab,
        tokenizer_cfg=cfg,
        max_len=2048,
        history_event_count=32,
        timing_jitter_bins=jitter_bins,
    )
    seq = builder.build(
        event_types=event_types,
        event_values=event_values,
        map_record={"hitsounded": True, "circle_size": 4.0, "approach_rate": 9.0, "overall_difficulty": 8.0, "hp_drain_rate": 6.0, "slider_multiplier": 1.4, "duration_ms": 5000.0},
        metadata=None,
        window_start_ms=0.0,
    )
    return seq.input_ids


def test_jitter_zero_produces_stable_output() -> None:
    a = _build_sample(0, seed=42)
    b = _build_sample(0, seed=7)
    assert bool((a == b).all())


def test_jitter_nonzero_changes_input_ids() -> None:
    unjittered = _build_sample(0, seed=42)
    jittered = _build_sample(2, seed=42)
    assert not bool((unjittered == jittered).all())


def test_jitter_only_affects_time_tokens() -> None:
    cfg = load_config("config/config.yaml").tokenizer
    vocab = Vocab(cfg)
    abs_start, abs_end = vocab.token_range(EventType.ABS_TIME)
    rel_start, rel_end = vocab.token_range(EventType.REL_TIME)
    unjittered = _build_sample(0, seed=42)
    jittered = _build_sample(2, seed=42)
    diff_positions = (unjittered != jittered).nonzero(as_tuple=True)[0]
    assert len(diff_positions) > 0
    for pos in diff_positions:
        original_id = int(unjittered[pos].item())
        is_abs = abs_start <= original_id < abs_end
        is_rel = rel_start <= original_id < rel_end
        assert is_abs or is_rel, f"position {pos} was not time-token originally (id {original_id})"


def test_jitter_respects_max_bins_and_ranges() -> None:
    cfg = load_config("config/config.yaml").tokenizer
    vocab = Vocab(cfg)
    abs_start, abs_end = vocab.token_range(EventType.ABS_TIME)
    rel_start, rel_end = vocab.token_range(EventType.REL_TIME)
    unjittered = _build_sample(0, seed=42)
    jittered = _build_sample(2, seed=42)
    diff_positions = (unjittered != jittered).nonzero(as_tuple=True)[0]
    for pos in diff_positions:
        orig = int(unjittered[pos].item())
        jitt = int(jittered[pos].item())
        offset = jitt - orig
        assert -2 <= offset <= 2
        if abs_start <= orig < abs_end:
            assert abs_start <= jitt < abs_end
        elif rel_start <= orig < rel_end:
            assert rel_start <= jitt < rel_end
        else:
            raise AssertionError(f"jittered a non-time token at pos {pos} (orig id {orig})")


def test_jitter_leaves_targets_unchanged() -> None:
    from src.training.data.sequence_builder import SequenceBuilder as SB

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

    torch.manual_seed(0)
    clean = SB(vocab=vocab, tokenizer_cfg=cfg, max_len=2048, history_event_count=32, timing_jitter_bins=0).build(
        event_types=event_types, event_values=event_values,
        map_record={"hitsounded": True, "circle_size": 4.0, "approach_rate": 9.0, "overall_difficulty": 8.0, "hp_drain_rate": 6.0, "slider_multiplier": 1.4, "duration_ms": 5000.0},
        metadata=None, window_start_ms=0.0,
    )
    torch.manual_seed(0)
    jittered = SB(vocab=vocab, tokenizer_cfg=cfg, max_len=2048, history_event_count=32, timing_jitter_bins=2).build(
        event_types=event_types, event_values=event_values,
        map_record={"hitsounded": True, "circle_size": 4.0, "approach_rate": 9.0, "overall_difficulty": 8.0, "hp_drain_rate": 6.0, "slider_multiplier": 1.4, "duration_ms": 5000.0},
        metadata=None, window_start_ms=0.0,
    )
    assert bool((clean.target_ids == jittered.target_ids).all())
    assert bool((clean.loss_mask == jittered.loss_mask).all())
    assert not bool((clean.input_ids == jittered.input_ids).all())
