from src.osu_tokenizer import EventType, Vocab, collect_timing_events

from .fixtures import build_beatmap, make_circle, make_config, make_timing_point


def test_beat_scaffold_emits_beats_between_timing_points_4_4() -> None:
    cfg = make_config()
    vocab = Vocab(cfg)
    tp = make_timing_point(time=0.0, bpm=120.0)
    beatmap = build_beatmap(
        hit_objects=[make_circle(time=1_000.0), make_circle(time=4_500.0)],
        timing_points=[tp],
    )
    events = collect_timing_events(beatmap, window_start_ms=0.0, vocab=vocab, config=cfg, clamp_abs_time=False)
    beat_count = sum(1 for e in events if e.type == EventType.BEAT)
    measure_count = sum(1 for e in events if e.type == EventType.MEASURE)
    timing_point_count = sum(1 for e in events if e.type == EventType.TIMING_POINT)
    assert timing_point_count == 1
    assert measure_count >= 1
    assert beat_count >= 3


def test_beat_scaffold_respects_meter_3_4() -> None:
    cfg = make_config()
    vocab = Vocab(cfg)
    tp = make_timing_point(time=0.0, bpm=120.0)
    tp.meter = 3
    beatmap = build_beatmap(
        hit_objects=[make_circle(time=1_000.0), make_circle(time=6_500.0)],
        timing_points=[tp],
    )
    events = collect_timing_events(beatmap, window_start_ms=0.0, vocab=vocab, config=cfg, clamp_abs_time=False)
    beat_count = sum(1 for e in events if e.type == EventType.BEAT)
    measure_count = sum(1 for e in events if e.type == EventType.MEASURE)
    assert measure_count >= 1
    assert beat_count >= 2 * measure_count


def test_snapping_emitted_per_hit_object() -> None:
    from src.osu_tokenizer import beatmap_to_events

    cfg = make_config()
    vocab = Vocab(cfg)
    tp = make_timing_point(time=0.0, bpm=120.0)
    beatmap = build_beatmap(
        hit_objects=[make_circle(time=500.0), make_circle(time=1000.0)],
        timing_points=[tp],
    )
    stream = beatmap_to_events(beatmap, window_start_ms=0.0, vocab=vocab, config=cfg, clamp_abs_time=False)
    snap_events = [e for e in stream.events if e.type == EventType.SNAPPING]
    assert len(snap_events) == 2
    for ev in snap_events:
        assert 1 <= ev.value <= cfg.snap_max
