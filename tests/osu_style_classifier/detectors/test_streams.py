from src.osu_style_classifier import classify

from ..fixtures import build_beatmap, make_timing_point, stream_circles


def test_detects_streams_at_high_bpm() -> None:
    tp = make_timing_point(bpm=200.0)
    interval = 60000.0 / 200.0 / 4
    circles = stream_circles(count=60, start_ms=1000.0, interval_ms=interval)
    beatmap = build_beatmap(circles, timing_points=[tp])

    tags = set(classify(beatmap).tags)
    assert "skillset/streams" in tags


def test_does_not_detect_streams_at_low_bpm() -> None:
    tp = make_timing_point(bpm=120.0)
    interval = 60000.0 / 120.0 / 4
    circles = stream_circles(count=20, start_ms=1000.0, interval_ms=interval)
    beatmap = build_beatmap(circles, timing_points=[tp])

    tags = set(classify(beatmap).tags)
    assert "skillset/streams" not in tags


def test_detects_spaced_streams_when_distance_large() -> None:
    tp = make_timing_point(bpm=200.0)
    interval = 60000.0 / 200.0 / 4
    circles = stream_circles(count=60, start_ms=1000.0, interval_ms=interval, dx=140.0)
    beatmap = build_beatmap(circles, timing_points=[tp])

    tags = set(classify(beatmap).tags)
    assert "streams/spaced streams" in tags
