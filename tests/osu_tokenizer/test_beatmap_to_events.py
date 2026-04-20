from src.osu.enums import CurveType
from src.osu_tokenizer import EventType, Vocab, attach_rel_times, beatmap_to_events

from .fixtures import build_beatmap, make_circle, make_config, make_slider, make_timing_point


def _types(events: list) -> list[EventType]:
    return [e.type for e in events]


def test_circle_emits_expected_event_group() -> None:
    cfg = make_config()
    vocab = Vocab(cfg)
    beatmap = build_beatmap([make_circle(time=100.0)], [make_timing_point()])
    stream = beatmap_to_events(beatmap, window_start_ms=100.0, vocab=vocab, config=cfg)
    assert _types(stream.events) == [
        EventType.ABS_TIME,
        EventType.DISTANCE,
        EventType.POS,
        EventType.HITSOUND,
        EventType.VOLUME,
        EventType.CIRCLE,
    ]


def test_slider_emits_head_anchors_last_anchor_end() -> None:
    cfg = make_config()
    vocab = Vocab(cfg)
    slider = make_slider(
        time=1000.0,
        head=(100.0, 100.0),
        anchors=[(200.0, 200.0), (300.0, 300.0)],
        duration=500.0,
        curve_type=CurveType.BEZIER,
    )
    beatmap = build_beatmap([slider], [make_timing_point()])
    stream = beatmap_to_events(beatmap, window_start_ms=1000.0, vocab=vocab, config=cfg)
    types = _types(stream.events)
    assert types[0] == EventType.ABS_TIME
    assert EventType.SLIDER_HEAD in types
    assert types.count(EventType.BEZIER_ANCHOR) == 1
    assert EventType.LAST_ANCHOR in types
    assert types[-1] == EventType.SLIDER_END


def test_slider_with_slides_emits_slides_token() -> None:
    cfg = make_config()
    vocab = Vocab(cfg)
    slider = make_slider(
        time=1000.0,
        anchors=[(200.0, 200.0)],
        duration=500.0,
        slides=3,
    )
    beatmap = build_beatmap([slider])
    stream = beatmap_to_events(beatmap, window_start_ms=1000.0, vocab=vocab, config=cfg)
    slides_events = [e for e in stream.events if e.type == EventType.SLIDER_SLIDES]
    assert len(slides_events) == 1
    assert slides_events[0].value == 3


def test_rel_time_is_delta_from_previous_abs() -> None:
    cfg = make_config()
    vocab = Vocab(cfg)
    beatmap = build_beatmap([make_circle(100.0), make_circle(350.0, x=300.0)])
    stream = beatmap_to_events(beatmap, window_start_ms=100.0, vocab=vocab, config=cfg)
    with_rel = attach_rel_times(stream.events, vocab, cfg)
    abs_values = [e.value for e in with_rel if e.type == EventType.ABS_TIME]
    rel_values = [e.value for e in with_rel if e.type == EventType.REL_TIME]
    assert len(abs_values) == len(rel_values)
    assert rel_values[0] == 0
    assert rel_values[1] == abs_values[1] - abs_values[0]
