from src.config.loader import load_config
from src.inference.detokenizer import events_to_beatmap
from src.osu.enums import CurveType
from src.osu.hit_object import Circle, Slider
from src.osu_tokenizer import Vocab, beatmap_to_events, collect_timing_events, merge_by_time

from tests.osu_tokenizer.fixtures import build_beatmap, make_circle, make_slider, make_timing_point


def _roundtrip(hit_objects: list) -> list:
    cfg = load_config("config/config.yaml").tokenizer
    vocab = Vocab(cfg)
    bm = build_beatmap(hit_objects=hit_objects, timing_points=[make_timing_point(0.0, 180.0)])
    stream = beatmap_to_events(bm, 0.0, vocab, cfg, clamp_abs_time=False)
    timing = collect_timing_events(bm, 0.0, vocab, cfg, clamp_abs_time=False)
    merged = merge_by_time(stream.events, timing)
    out = events_to_beatmap(merged, vocab=vocab, tokenizer_cfg=cfg, audio_filename="x.mp3", bpm=180.0)
    return list(out.hit_objects)


def test_circles_only_roundtrip() -> None:
    inputs = [make_circle(time=1000.0), make_circle(time=2000.0, x=300.0, y=200.0)]
    out = _roundtrip(inputs)
    assert len(out) == 2
    assert all(isinstance(h, Circle) for h in out)


def test_single_slider_roundtrip() -> None:
    inputs = [make_slider(time=1000.0, head=(100.0, 100.0), anchors=[(200.0, 200.0)])]
    out = _roundtrip(inputs)
    assert len(out) == 1
    assert isinstance(out[0], Slider)


def test_multi_anchor_slider_roundtrip() -> None:
    inputs = [make_slider(time=1000.0, head=(100.0, 100.0), anchors=[(200.0, 200.0), (300.0, 150.0), (400.0, 100.0)])]
    out = _roundtrip(inputs)
    assert len(out) == 1
    assert isinstance(out[0], Slider)


def test_unclosed_slider_does_not_swallow_next_object() -> None:
    from src.config.loader import load_config
    from src.osu_tokenizer import Event, EventType, SpecialToken, Vocab
    from src.inference.detokenizer import events_to_beatmap

    cfg = load_config("config/config.yaml").tokenizer
    vocab = Vocab(cfg)

    def enc(t: EventType, v: int = 0) -> Event:
        return Event(type=t, value=v)

    events: list[Event] = [
        enc(EventType.SLIDER_HEAD, 0),
        enc(EventType.ABS_TIME, 100),
        enc(EventType.DISTANCE, 0),
        enc(EventType.POS, 10),
        enc(EventType.HITSOUND, 0),
        enc(EventType.VOLUME, 80),
        enc(EventType.BEZIER_ANCHOR, 0),
        enc(EventType.POS, 20),
        enc(EventType.CIRCLE, 0),
        enc(EventType.ABS_TIME, 200),
        enc(EventType.DISTANCE, 0),
        enc(EventType.POS, 30),
        enc(EventType.HITSOUND, 0),
        enc(EventType.VOLUME, 80),
    ]
    beatmap = events_to_beatmap(events, vocab=vocab, tokenizer_cfg=cfg, audio_filename="x.mp3", bpm=180.0)
    from src.osu.hit_object import Circle, Slider

    assert sum(1 for h in beatmap.hit_objects if isinstance(h, Circle)) == 1, "circle after unclosed slider must survive"
    _ = SpecialToken


def test_mixed_roundtrip_preserves_counts() -> None:
    inputs = [
        make_circle(time=1000.0),
        make_slider(time=2000.0, head=(100.0, 100.0), anchors=[(200.0, 200.0)]),
        make_circle(time=3500.0),
        make_slider(time=4500.0, head=(300.0, 100.0), anchors=[(400.0, 200.0), (450.0, 100.0)]),
        make_slider(time=5500.0, head=(100.0, 200.0), anchors=[(200.0, 100.0)], curve_type=CurveType.BEZIER),
        make_circle(time=7000.0),
    ]
    out = _roundtrip(inputs)
    n_sliders = sum(1 for h in out if isinstance(h, Slider))
    n_circles = sum(1 for h in out if isinstance(h, Circle))
    assert n_sliders == 3, f"expected 3 sliders after roundtrip, got {n_sliders}"
    assert n_circles == 3, f"expected 3 circles after roundtrip, got {n_circles}"


def test_slider_duration_survives_roundtrip_within_7_percent() -> None:
    inputs = [
        make_slider(time=1000.0, head=(100.0, 100.0), anchors=[(200.0, 150.0)], duration=500.0),
        make_slider(time=3000.0, head=(300.0, 100.0), anchors=[(400.0, 200.0), (450.0, 100.0)], duration=1000.0),
    ]
    out = _roundtrip(inputs)
    sliders = [h for h in out if isinstance(h, Slider)]
    assert len(sliders) == 2
    for orig, round_tripped in zip(inputs, sliders):
        orig_dur = orig.object_params.duration
        new_dur = round_tripped.object_params.duration
        err_pct = abs(new_dur - orig_dur) / max(orig_dur, 1e-6) * 100
        assert err_pct < 7.0, f"slider duration drift {err_pct:.1f}% (orig {orig_dur}ms, got {new_dur}ms)"


def test_slider_time_and_position_survive_roundtrip() -> None:
    inputs = [
        make_slider(time=1500.0, head=(100.0, 100.0), anchors=[(250.0, 150.0)]),
    ]
    out = _roundtrip(inputs)
    assert len(out) == 1
    s = out[0]
    assert isinstance(s, Slider)
    assert abs(float(s.time) - 1500.0) <= 20.0
    assert abs(s.x - 100.0) <= 32.0
    assert abs(s.y - 100.0) <= 32.0
